"""
Enhanced Text Processor for Semantic Co-occurrence Network Analysis

This module implements a linguistically and statistically grounded text processing pipeline
specifically designed for academic NLP practice in semantic co-occurrence network construction.

The pipeline follows 6 steps:
1. Linguistic Preprocessing (NO stopword removal yet)
2. Phrase/Keyphrase Candidate Extraction
3. Static Stopword Filtering (Lightweight)
4. Corpus-level Statistics
5. Dynamic Stopword Identification
6. Final Phrase Filtering

Technology Stack:
- spaCy for tokenization, POS tagging, dependency parsing
- spaCy Matcher for rule-based phrase extraction
- TF-IDF for dynamic stopword identification
- Static stopword lists (English + Chinese)
"""

import re
import math
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass
from pathlib import Path
import json

# spaCy imports
try:
    import spacy
    from spacy.matcher import Matcher
    from spacy.tokens import Doc, Token, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available, falling back to basic processing")
    # Create dummy classes for type hints when spaCy is not available
    class Doc:
        pass
    class Token:
        pass
    class Span:
        pass

from ..core.data_models import TOCDocument, ProcessedDocument, Window
from ..core.config import Config
from ..core.error_handler import ErrorHandler


@dataclass
class PhraseCandidate:
    """Phrase candidate with linguistic and statistical information"""
    text: str
    tokens: List[str]
    pos_tags: List[str]
    dependency_relations: List[str]
    segment_id: str
    state: str
    frequency: int = 1
    
    def __post_init__(self):
        if not self.tokens:
            self.tokens = self.text.split()


@dataclass
class CorpusStatistics:
    """Corpus-level statistics for phrases"""
    phrase: str
    tf: float  # Total frequency across corpus
    df: int    # Number of segments containing the phrase
    idf: float # Inverse document frequency
    tfidf: float # TF-IDF score
    segments: Set[str]  # Segments containing this phrase
    states: Set[str]    # States containing this phrase
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'phrase': self.phrase,
            'tf': self.tf,
            'df': self.df,
            'idf': self.idf,
            'tfidf': self.tfidf,
            'segments': list(self.segments),
            'states': list(self.states)
        }


class EnhancedLanguageProcessor:
    """
    Enhanced language processor using spaCy for linguistic analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load spaCy models
        self.nlp_en = None
        self.nlp_zh = None
        
        if SPACY_AVAILABLE:
            try:
                # Load English model
                self.nlp_en = spacy.load("en_core_web_sm")
                self.logger.info("Loaded English spaCy model: en_core_web_sm")
            except OSError:
                self.logger.warning("English spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            
            try:
                # Load Chinese model
                self.nlp_zh = spacy.load("zh_core_web_sm")
                self.logger.info("Loaded Chinese spaCy model: zh_core_web_sm")
            except OSError:
                self.logger.warning("Chinese spaCy model not found. Install with: python -m spacy download zh_core_web_sm")
        
        # Initialize matchers for phrase extraction
        self.en_matcher = None
        self.zh_matcher = None
        self._setup_matchers()
    
    def _setup_matchers(self):
        """Setup spaCy matchers for phrase extraction"""
        if not SPACY_AVAILABLE:
            return
        
        # English matcher patterns
        if self.nlp_en:
            self.en_matcher = Matcher(self.nlp_en.vocab)
            
            # Pattern 1: (ADJ)*(NOUN)+
            adj_noun_pattern = [
                {"POS": {"IN": ["ADJ"]}, "OP": "*"},  # Zero or more adjectives
                {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}  # One or more nouns
            ]
            
            # Pattern 2: NOUN + NOUN (compound nouns)
            noun_noun_pattern = [
                {"POS": {"IN": ["NOUN", "PROPN"]}},
                {"POS": {"IN": ["NOUN", "PROPN"]}}
            ]
            
            # Pattern 3: ADJ + NOUN
            adj_noun_simple = [
                {"POS": "ADJ"},
                {"POS": {"IN": ["NOUN", "PROPN"]}}
            ]
            
            self.en_matcher.add("ADJ_NOUN", [adj_noun_pattern])
            self.en_matcher.add("NOUN_NOUN", [noun_noun_pattern])
            self.en_matcher.add("ADJ_NOUN_SIMPLE", [adj_noun_simple])
        
        # Chinese matcher patterns
        if self.nlp_zh:
            self.zh_matcher = Matcher(self.nlp_zh.vocab)
            
            # Chinese noun phrases
            zh_noun_pattern = [
                {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}
            ]
            
            # Adjective + Noun in Chinese
            zh_adj_noun = [
                {"POS": "ADJ"},
                {"POS": {"IN": ["NOUN", "PROPN"]}}
            ]
            
            self.zh_matcher.add("ZH_NOUN", [zh_noun_pattern])
            self.zh_matcher.add("ZH_ADJ_NOUN", [zh_adj_noun])
    
    def detect_language(self, text: str) -> str:
        """
        Detect text language (English or Chinese)
        
        Args:
            text: Input text
            
        Returns:
            Language code ('en' or 'zh')
        """
        if not text or not text.strip():
            return 'en'  # Default to English
        
        # Count Chinese characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.strip())
        
        if total_chars == 0:
            return 'en'
        
        chinese_ratio = chinese_chars / total_chars
        
        # If more than 10% Chinese characters, consider it Chinese
        return 'zh' if chinese_ratio > 0.1 else 'en'
    
    def linguistic_preprocessing(self, text: str, language: str) -> Dict[str, Any]:
        """
        Step 1: Linguistic Preprocessing (NO stopword removal yet)
        
        Args:
            text: Input text
            language: Language code ('en' or 'zh')
            
        Returns:
            Dictionary with preprocessing results
        """
        # Clean text (HTML, special chars)
        cleaned_text = self._clean_text(text)
        
        if not SPACY_AVAILABLE:
            return self._fallback_preprocessing(cleaned_text, language)
        
        # Choose appropriate spaCy model
        nlp = self.nlp_en if language == 'en' else self.nlp_zh
        if nlp is None:
            return self._fallback_preprocessing(cleaned_text, language)
        
        try:
            # Process with spaCy
            doc = nlp(cleaned_text)
            
            # Extract linguistic information
            sentences = [sent.text for sent in doc.sents]
            tokens = []
            pos_tags = []
            dep_relations = []
            
            for token in doc:
                if not token.is_space:  # Skip whitespace tokens
                    tokens.append(token.text)
                    pos_tags.append(token.pos_)
                    dep_relations.append(token.dep_)
            
            return {
                'cleaned_text': cleaned_text,
                'sentences': sentences,
                'tokens': tokens,
                'pos_tags': pos_tags,
                'dependency_relations': dep_relations,
                'spacy_doc': doc  # Keep for phrase extraction
            }
            
        except Exception as e:
            self.logger.warning(f"spaCy preprocessing failed: {e}, falling back to basic processing")
            return self._fallback_preprocessing(cleaned_text, language)
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing HTML tags and normalizing special characters"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[^\w\s\u4e00-\u9fff.,;:!?()-]', ' ', text)
        
        return text.strip()
    
    def _fallback_preprocessing(self, text: str, language: str) -> Dict[str, Any]:
        """Fallback preprocessing when spaCy is not available"""
        # Simple sentence segmentation
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Simple tokenization
        if language == 'zh':
            # For Chinese, split by common punctuation and whitespace
            tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', text)
        else:
            # For English, split by whitespace and punctuation
            tokens = re.findall(r'\b\w+\b', text.lower())
        
        return {
            'cleaned_text': text,
            'sentences': sentences,
            'tokens': tokens,
            'pos_tags': ['UNKNOWN'] * len(tokens),
            'dependency_relations': ['UNKNOWN'] * len(tokens),
            'spacy_doc': None
        }


class PhraseCandidateExtractor:
    """
    Step 2: Phrase/Keyphrase Candidate Extraction using spaCy Matcher
    """
    
    def __init__(self, config: Dict[str, Any], language_processor: EnhancedLanguageProcessor):
        self.config = config
        self.language_processor = language_processor
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.min_phrase_length = config.get('min_phrase_length', 2)
        self.max_phrase_length = config.get('max_phrase_length', 4)
    
    def extract_phrase_candidates(self, preprocessing_result: Dict[str, Any], 
                                segment_id: str, state: str, language: str) -> List[PhraseCandidate]:
        """
        Extract phrase candidates using spaCy Matcher rules
        
        Args:
            preprocessing_result: Result from linguistic preprocessing
            segment_id: Segment identifier
            state: State/region identifier
            language: Language code
            
        Returns:
            List of phrase candidates
        """
        spacy_doc = preprocessing_result.get('spacy_doc')
        
        if spacy_doc is not None and SPACY_AVAILABLE:
            return self._extract_with_spacy_matcher(spacy_doc, segment_id, state, language)
        else:
            return self._extract_with_fallback(preprocessing_result, segment_id, state, language)
    
    def _extract_with_spacy_matcher(self, doc: Union[Doc, Any], segment_id: str, state: str, language: str) -> List[PhraseCandidate]:
        """Extract phrases using spaCy Matcher"""
        candidates = []
        
        # Choose appropriate matcher
        matcher = (self.language_processor.en_matcher if language == 'en' 
                  else self.language_processor.zh_matcher)
        
        if matcher is None:
            return self._extract_with_fallback_from_doc(doc, segment_id, state, language)
        
        try:
            # Find matches
            matches = matcher(doc)
            
            for match_id, start, end in matches:
                span = doc[start:end]
                phrase_text = span.text.strip()
                
                # Filter by length
                if len(phrase_text) < self.min_phrase_length:
                    continue
                
                # Extract tokens and linguistic features
                tokens = [token.text for token in span]
                pos_tags = [token.pos_ for token in span]
                dep_relations = [token.dep_ for token in span]
                
                # Additional filtering
                if self._should_keep_phrase(phrase_text, tokens, pos_tags, language):
                    candidate = PhraseCandidate(
                        text=phrase_text,
                        tokens=tokens,
                        pos_tags=pos_tags,
                        dependency_relations=dep_relations,
                        segment_id=segment_id,
                        state=state
                    )
                    candidates.append(candidate)
            
            # Also extract dependency-based phrases
            dep_candidates = self._extract_dependency_phrases(doc, segment_id, state, language)
            candidates.extend(dep_candidates)
            
        except Exception as e:
            self.logger.warning(f"spaCy matcher extraction failed: {e}")
            return self._extract_with_fallback_from_doc(doc, segment_id, state, language)
        
        return candidates
    
    def _extract_dependency_phrases(self, doc: Union[Doc, Any], segment_id: str, state: str, language: str) -> List[PhraseCandidate]:
        """Extract phrases based on dependency relations (amod, compound)"""
        candidates = []
        
        for token in doc:
            # Look for amod (adjectival modifier) relations
            if token.dep_ == "amod" and token.head.pos_ in ["NOUN", "PROPN"]:
                phrase_text = f"{token.text} {token.head.text}"
                if self._should_keep_phrase(phrase_text, [token.text, token.head.text], 
                                          [token.pos_, token.head.pos_], language):
                    candidate = PhraseCandidate(
                        text=phrase_text,
                        tokens=[token.text, token.head.text],
                        pos_tags=[token.pos_, token.head.pos_],
                        dependency_relations=[token.dep_, token.head.dep_],
                        segment_id=segment_id,
                        state=state
                    )
                    candidates.append(candidate)
            
            # Look for compound relations
            elif token.dep_ == "compound" and token.head.pos_ in ["NOUN", "PROPN"]:
                phrase_text = f"{token.text} {token.head.text}"
                if self._should_keep_phrase(phrase_text, [token.text, token.head.text],
                                          [token.pos_, token.head.pos_], language):
                    candidate = PhraseCandidate(
                        text=phrase_text,
                        tokens=[token.text, token.head.text],
                        pos_tags=[token.pos_, token.head.pos_],
                        dependency_relations=[token.dep_, token.head.dep_],
                        segment_id=segment_id,
                        state=state
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _extract_with_fallback(self, preprocessing_result: Dict[str, Any], 
                             segment_id: str, state: str, language: str) -> List[PhraseCandidate]:
        """Fallback phrase extraction without spaCy"""
        candidates = []
        tokens = preprocessing_result.get('tokens', [])
        pos_tags = preprocessing_result.get('pos_tags', [])
        
        if len(tokens) < 2:
            return candidates
        
        # Generate n-grams (2 to max_phrase_length)
        for n in range(2, min(self.max_phrase_length + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                phrase_tokens = tokens[i:i+n]
                
                # Skip if any token is too short or all punctuation
                if any(len(token) < 2 and not token.isalnum() for token in phrase_tokens):
                    continue
                
                phrase_text = ' '.join(phrase_tokens) if language == 'en' else ''.join(phrase_tokens)
                phrase_pos = pos_tags[i:i+n] if i+n <= len(pos_tags) else ['UNKNOWN'] * n
                
                if self._should_keep_phrase(phrase_text, phrase_tokens, phrase_pos, language):
                    candidate = PhraseCandidate(
                        text=phrase_text,
                        tokens=phrase_tokens,
                        pos_tags=phrase_pos,
                        dependency_relations=['UNKNOWN'] * n,
                        segment_id=segment_id,
                        state=state
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _extract_with_fallback_from_doc(self, doc: Union[Doc, Any], segment_id: str, state: str, language: str) -> List[PhraseCandidate]:
        """Fallback extraction from spaCy doc when matcher fails"""
        candidates = []
        tokens = [token.text for token in doc if not token.is_space]
        pos_tags = [token.pos_ for token in doc if not token.is_space]
        
        # Generate bigrams and trigrams
        for n in range(2, min(4, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                phrase_tokens = tokens[i:i+n]
                phrase_text = ' '.join(phrase_tokens) if language == 'en' else ''.join(phrase_tokens)
                phrase_pos = pos_tags[i:i+n]
                
                if self._should_keep_phrase(phrase_text, phrase_tokens, phrase_pos, language):
                    candidate = PhraseCandidate(
                        text=phrase_text,
                        tokens=phrase_tokens,
                        pos_tags=phrase_pos,
                        dependency_relations=['UNKNOWN'] * n,
                        segment_id=segment_id,
                        state=state
                    )
                    candidates.append(candidate)
        
        return candidates
    
    def _should_keep_phrase(self, phrase_text: str, tokens: List[str], pos_tags: List[str], language: str) -> bool:
        """
        Determine if a phrase candidate should be kept
        
        Filtering rules:
        - No single stopwords
        - No pure function phrases
        - No single punctuation or numeric tokens
        - Must be linguistically meaningful
        """
        # Basic length check
        if len(tokens) < 2:
            return False
        
        # Check for pure punctuation or numbers
        if all(re.match(r'^[\W\d]+$', token) for token in tokens):
            return False
        
        # Language-specific filtering
        if language == 'en':
            return self._should_keep_english_phrase(phrase_text, tokens, pos_tags)
        else:
            return self._should_keep_chinese_phrase(phrase_text, tokens, pos_tags)
    
    def _should_keep_english_phrase(self, phrase_text: str, tokens: List[str], pos_tags: List[str]) -> bool:
        """English-specific phrase filtering"""
        # Basic length and content checks
        if len(tokens) < 2:
            return False
        
        # Skip if all tokens are too short (except common short words)
        short_ok = {'is', 'of', 'to', 'in', 'on', 'at', 'by', 'or', 'it', 'we', 'he', 'me'}
        if all(len(token) < 2 and token.lower() not in short_ok for token in tokens):
            return False
        
        # Avoid phrases that are entirely function words
        function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        if all(token.lower() in function_words for token in tokens):
            return False
        
        # Avoid phrases starting and ending with function words for 2-word phrases
        if len(tokens) == 2 and tokens[0].lower() in function_words and tokens[1].lower() in function_words:
            return False
        
        # Must contain at least one substantial word (length >= 3 or known good short words)
        substantial_words = {'is', 'of', 'to', 'in', 'on', 'at', 'by', 'or', 'it', 'we', 'he', 'me', 'my', 'us'}
        has_substantial = any(len(token) >= 3 or token.lower() in substantial_words for token in tokens)
        if not has_substantial:
            return False
        
        return True
    
    def _should_keep_chinese_phrase(self, phrase_text: str, tokens: List[str], pos_tags: List[str]) -> bool:
        """Chinese-specific phrase filtering"""
        # Must contain at least some Chinese characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', phrase_text))
        if chinese_chars < len(phrase_text) * 0.5:
            return False
        
        # Avoid pure function words
        function_words = {'的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '也', '很', '到', '说', '要', '去', '会', '着'}
        if all(token in function_words for token in tokens):
            return False
        
        return True


class StaticStopwordFilter:
    """
    Step 3: Static Stopword Filtering (Lightweight)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load static stopword lists
        self.english_stopwords = self._load_english_stopwords()
        self.chinese_stopwords = self._load_chinese_stopwords()
    
    def _load_english_stopwords(self) -> Set[str]:
        """Load English stopwords"""
        stopwords = {
            # Articles
            'the', 'a', 'an',
            # Prepositions
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'into', 'onto', 'upon',
            # Conjunctions
            'and', 'or', 'but', 'nor', 'yet', 'so',
            # Pronouns
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            # Auxiliary verbs
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            # Modal verbs
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
            # Common adverbs
            'not', 'no', 'yes', 'very', 'too', 'also', 'just', 'only', 'even', 'still'
        }
        
        # Load from file if specified
        stopwords_file = self.config.get('english_stopwords_file')
        if stopwords_file and Path(stopwords_file).exists():
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    file_stopwords = {line.strip().lower() for line in f if line.strip() and not line.startswith('#')}
                stopwords.update(file_stopwords)
                self.logger.info(f"Loaded {len(file_stopwords)} English stopwords from file")
            except Exception as e:
                self.logger.warning(f"Failed to load English stopwords from file: {e}")
        
        return stopwords
    
    def _load_chinese_stopwords(self) -> Set[str]:
        """Load Chinese stopwords"""
        stopwords = {
            # Common function words
            '的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '也', '很', '到', '说', '要', '去', '会', '着',
            '个', '上', '来', '下', '对', '从', '把', '被', '让', '使', '给', '为', '与', '及', '或', '但', '而', '且',
            # Pronouns
            '这', '那', '些', '此', '其', '他', '她', '它', '我', '你', '您', '们', '自己',
            # Question words
            '什么', '怎么', '为什么', '哪里', '哪个', '多少', '几个',
            # Time and quantity
            '现在', '以前', '以后', '今天', '昨天', '明天', '一些', '很多', '少数'
        }
        
        # Load from file if specified
        stopwords_file = self.config.get('chinese_stopwords_file')
        if stopwords_file and Path(stopwords_file).exists():
            try:
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    file_stopwords = {line.strip() for line in f if line.strip() and not line.startswith('#')}
                stopwords.update(file_stopwords)
                self.logger.info(f"Loaded {len(file_stopwords)} Chinese stopwords from file")
            except Exception as e:
                self.logger.warning(f"Failed to load Chinese stopwords from file: {e}")
        
        return stopwords
    
    def filter_candidates(self, candidates: List[PhraseCandidate], language: str) -> List[PhraseCandidate]:
        """
        Apply static stopword filtering
        
        Rules:
        - Remove phrases consisting entirely of stopwords
        - Remove phrases shorter than 2 tokens (unless justified)
        - Do NOT over-filter at this stage
        """
        stopwords = self.english_stopwords if language == 'en' else self.chinese_stopwords
        filtered_candidates = []
        
        for candidate in candidates:
            # Check if phrase consists entirely of stopwords
            if language == 'en':
                tokens_to_check = [token.lower() for token in candidate.tokens]
            else:
                tokens_to_check = candidate.tokens
            
            if all(token in stopwords for token in tokens_to_check):
                continue  # Skip phrases that are entirely stopwords
            
            # Check minimum length (conservative)
            if len(candidate.tokens) < 2:
                continue
            
            filtered_candidates.append(candidate)
        
        removed_count = len(candidates) - len(filtered_candidates)
        self.logger.debug(f"Static stopword filtering: removed {removed_count} candidates")
        
        return filtered_candidates


class CorpusStatisticsCalculator:
    """
    Step 4: Corpus-level Statistics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_statistics(self, all_candidates: List[PhraseCandidate]) -> Dict[str, CorpusStatistics]:
        """
        Calculate corpus-level statistics for all phrases
        
        For each phrase, compute:
        - TF: total frequency across corpus
        - DF: number of segments containing the phrase
        - IDF: log(N / (DF + 1))
        - TF-IDF
        """
        if not all_candidates:
            return {}
        
        # Group candidates by phrase text
        phrase_groups = defaultdict(list)
        for candidate in all_candidates:
            phrase_groups[candidate.text].extend([candidate])
        
        # Calculate statistics
        total_segments = len(set(candidate.segment_id for candidate in all_candidates))
        statistics = {}
        
        for phrase_text, candidates in phrase_groups.items():
            # Calculate TF (total frequency)
            tf = len(candidates)
            
            # Calculate DF (document frequency - number of segments)
            segments = set(candidate.segment_id for candidate in candidates)
            df = len(segments)
            
            # Calculate IDF
            idf = math.log(total_segments / (df + 1)) if df > 0 else 0
            
            # Calculate TF-IDF
            tfidf = tf * idf
            
            # Collect states
            states = set(candidate.state for candidate in candidates)
            
            statistics[phrase_text] = CorpusStatistics(
                phrase=phrase_text,
                tf=tf,
                df=df,
                idf=idf,
                tfidf=tfidf,
                segments=segments,
                states=states
            )
        
        self.logger.info(f"Calculated statistics for {len(statistics)} unique phrases")
        return statistics


class DynamicStopwordIdentifier:
    """
    Step 5: Dynamic Stopword Identification (Core Requirement)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Thresholds (configurable)
        self.df_threshold_ratio = config.get('df_threshold_ratio', 0.8)  # DF / N >= threshold
        self.tfidf_threshold = config.get('tfidf_threshold', 0.1)  # Low TF-IDF threshold
        self.min_frequency = config.get('min_frequency_for_stopword', 5)  # Minimum frequency to consider
    
    def identify_dynamic_stopwords(self, statistics: Dict[str, CorpusStatistics]) -> Tuple[Set[str], Dict[str, Dict[str, Any]]]:
        """
        Identify dynamic stopwords using TF-IDF
        
        A phrase is considered a dynamic stopword if:
        - Appears in a large proportion of segments (DF / N >= threshold)
        - Has low TF-IDF relative to corpus distribution
        
        Returns:
            Tuple of (dynamic_stopwords_set, explanation_dict)
        """
        if not statistics:
            return set(), {}
        
        dynamic_stopwords = set()
        explanations = {}
        
        # Calculate total number of segments
        all_segments = set()
        for stat in statistics.values():
            all_segments.update(stat.segments)
        total_segments = len(all_segments)
        
        if total_segments == 0:
            return set(), {}
        
        # Calculate thresholds
        df_cutoff = total_segments * self.df_threshold_ratio
        
        # Analyze each phrase
        for phrase, stat in statistics.items():
            explanation = {
                'phrase': phrase,
                'tf': stat.tf,
                'df': stat.df,
                'idf': stat.idf,
                'tfidf': stat.tfidf,
                'df_ratio': stat.df / total_segments,
                'is_dynamic_stopword': False,
                'reasons': []
            }
            
            # Check conditions
            high_frequency = stat.df >= df_cutoff
            low_tfidf = stat.tfidf < self.tfidf_threshold
            sufficient_frequency = stat.tf >= self.min_frequency
            
            if high_frequency:
                explanation['reasons'].append(f"High document frequency: {stat.df}/{total_segments} = {stat.df/total_segments:.3f} >= {self.df_threshold_ratio}")
            
            if low_tfidf:
                explanation['reasons'].append(f"Low TF-IDF score: {stat.tfidf:.4f} < {self.tfidf_threshold}")
            
            if not sufficient_frequency:
                explanation['reasons'].append(f"Insufficient frequency: {stat.tf} < {self.min_frequency}")
            
            # A phrase is a dynamic stopword if it meets all conditions
            if high_frequency and low_tfidf and sufficient_frequency:
                dynamic_stopwords.add(phrase)
                explanation['is_dynamic_stopword'] = True
                self.logger.debug(f"Dynamic stopword identified: '{phrase}' (TF-IDF: {stat.tfidf:.4f}, DF: {stat.df}/{total_segments})")
            
            explanations[phrase] = explanation
        
        self.logger.info(f"Identified {len(dynamic_stopwords)} dynamic stopwords from {len(statistics)} phrases")
        return dynamic_stopwords, explanations
    
    def save_dynamic_stopwords(self, dynamic_stopwords: Set[str], explanations: Dict[str, Dict[str, Any]], output_path: str):
        """Save dynamic stopwords with statistics for traceability"""
        try:
            # Prepare output data
            output_data = {
                'dynamic_stopwords': list(dynamic_stopwords),
                'total_count': len(dynamic_stopwords),
                'configuration': {
                    'df_threshold_ratio': self.df_threshold_ratio,
                    'tfidf_threshold': self.tfidf_threshold,
                    'min_frequency': self.min_frequency
                },
                'explanations': explanations
            }
            
            # Save to JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            # Also save simple text file
            txt_path = output_path.replace('.json', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("# Dynamic Stopwords\n")
                f.write(f"# Total: {len(dynamic_stopwords)}\n")
                f.write(f"# Configuration: DF_ratio>={self.df_threshold_ratio}, TF-IDF<{self.tfidf_threshold}, min_freq>={self.min_frequency}\n\n")
                for stopword in sorted(dynamic_stopwords):
                    f.write(f"{stopword}\n")
            
            self.logger.info(f"Saved dynamic stopwords to {output_path} and {txt_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save dynamic stopwords: {e}")
            raise


class FinalPhraseFilter:
    """
    Step 6: Final Phrase Filtering
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def apply_final_filtering(self, candidates: List[PhraseCandidate], 
                            static_stopwords: Set[str], dynamic_stopwords: Set[str],
                            language: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Combine static and dynamic stopwords and filter phrase list
        
        Returns:
            Tuple of (cleaned_phrase_list, phrase_to_segments_mapping)
        """
        # Combine stopwords
        all_stopwords = static_stopwords.union(dynamic_stopwords)
        
        # Filter candidates
        filtered_phrases = []
        phrase_to_segments = defaultdict(list)
        
        for candidate in candidates:
            phrase_text = candidate.text
            
            # Check against combined stopwords
            if language == 'en':
                check_phrase = phrase_text.lower()
            else:
                check_phrase = phrase_text
            
            if check_phrase not in all_stopwords:
                filtered_phrases.append(phrase_text)
                phrase_to_segments[phrase_text].append(candidate.segment_id)
        
        # Remove duplicates while preserving order
        unique_phrases = []
        seen = set()
        for phrase in filtered_phrases:
            if phrase not in seen:
                unique_phrases.append(phrase)
                seen.add(phrase)
        
        # Convert defaultdict to regular dict
        phrase_mappings = dict(phrase_to_segments)
        
        removed_count = len(candidates) - len(unique_phrases)
        self.logger.info(f"Final filtering: {len(unique_phrases)} phrases kept, {removed_count} removed")
        
        return unique_phrases, phrase_mappings


class EnhancedTextProcessor:
    """
    Main Enhanced Text Processor implementing the 6-step pipeline
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the enhanced text processor
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.error_handler = ErrorHandler(self.config.get_section('error_handling'))
        self.logger = logging.getLogger(__name__)
        
        # Get processing configuration
        processing_config = self.config.get_section('enhanced_text_processing')
        
        # Initialize components
        self.language_processor = EnhancedLanguageProcessor(processing_config)
        self.phrase_extractor = PhraseCandidateExtractor(processing_config, self.language_processor)
        self.static_filter = StaticStopwordFilter(processing_config)
        self.statistics_calculator = CorpusStatisticsCalculator(processing_config)
        self.dynamic_identifier = DynamicStopwordIdentifier(processing_config)
        self.final_filter = FinalPhraseFilter(processing_config)
        
        self.logger.info("EnhancedTextProcessor initialized successfully")
    
    def process_documents(self, documents: List[TOCDocument]) -> Dict[str, Any]:
        """
        Process documents through the complete 6-step pipeline
        
        Args:
            documents: List of TOC documents
            
        Returns:
            Dictionary containing all processing results
        """
        self.logger.info(f"Starting enhanced text processing for {len(documents)} documents")
        
        # Step 1: Linguistic Preprocessing for all documents
        self.logger.info("Step 1: Linguistic Preprocessing")
        preprocessing_results = []
        for doc in documents:
            language = self.language_processor.detect_language(doc.text)
            result = self.language_processor.linguistic_preprocessing(doc.text, language)
            result['document'] = doc
            result['language'] = language
            preprocessing_results.append(result)
        
        # Step 2: Phrase Candidate Extraction
        self.logger.info("Step 2: Phrase Candidate Extraction")
        all_candidates = []
        for result in preprocessing_results:
            doc = result['document']
            language = result['language']
            candidates = self.phrase_extractor.extract_phrase_candidates(
                result, doc.segment_id, doc.state or 'unknown', language
            )
            all_candidates.extend(candidates)
        
        self.logger.info(f"Extracted {len(all_candidates)} phrase candidates")
        
        # Step 3: Static Stopword Filtering
        self.logger.info("Step 3: Static Stopword Filtering")
        # Group by language for filtering
        en_candidates = [c for c in all_candidates if self.language_processor.detect_language(c.text) == 'en']
        zh_candidates = [c for c in all_candidates if self.language_processor.detect_language(c.text) == 'zh']
        
        filtered_en = self.static_filter.filter_candidates(en_candidates, 'en') if en_candidates else []
        filtered_zh = self.static_filter.filter_candidates(zh_candidates, 'zh') if zh_candidates else []
        
        filtered_candidates = filtered_en + filtered_zh
        self.logger.info(f"After static filtering: {len(filtered_candidates)} candidates")
        
        # Step 4: Corpus-level Statistics
        self.logger.info("Step 4: Corpus-level Statistics")
        statistics = self.statistics_calculator.calculate_statistics(filtered_candidates)
        
        # Step 5: Dynamic Stopword Identification
        self.logger.info("Step 5: Dynamic Stopword Identification")
        dynamic_stopwords, explanations = self.dynamic_identifier.identify_dynamic_stopwords(statistics)
        
        # Step 6: Final Phrase Filtering
        self.logger.info("Step 6: Final Phrase Filtering")
        # Combine static stopwords from both languages
        all_static_stopwords = self.static_filter.english_stopwords.union(self.static_filter.chinese_stopwords)
        
        final_phrases, phrase_mappings = self.final_filter.apply_final_filtering(
            filtered_candidates, all_static_stopwords, dynamic_stopwords, 'mixed'
        )
        
        # Compile results
        results = {
            'preprocessing_results': preprocessing_results,
            'phrase_candidates': all_candidates,
            'filtered_candidates': filtered_candidates,
            'corpus_statistics': statistics,
            'dynamic_stopwords': dynamic_stopwords,
            'dynamic_stopword_explanations': explanations,
            'static_stopwords': {
                'english': self.static_filter.english_stopwords,
                'chinese': self.static_filter.chinese_stopwords
            },
            'final_phrases': final_phrases,
            'phrase_to_segments': phrase_mappings,
            'processing_metadata': {
                'total_documents': len(documents),
                'total_candidates': len(all_candidates),
                'after_static_filtering': len(filtered_candidates),
                'dynamic_stopwords_count': len(dynamic_stopwords),
                'final_phrases_count': len(final_phrases),
                'languages_detected': list(set(r['language'] for r in preprocessing_results))
            }
        }
        
        self.logger.info(f"Enhanced text processing completed: {len(final_phrases)} final phrases")
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save processing results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save dynamic stopwords
        dynamic_stopwords_path = output_path / "dynamic_stopwords.json"
        self.dynamic_identifier.save_dynamic_stopwords(
            results['dynamic_stopwords'],
            results['dynamic_stopword_explanations'],
            str(dynamic_stopwords_path)
        )
        
        # Save corpus statistics
        stats_path = output_path / "corpus_statistics.json"
        stats_data = {phrase: stat.to_dict() for phrase, stat in results['corpus_statistics'].items()}
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        # Save final phrases
        phrases_path = output_path / "final_phrases.json"
        phrases_data = {
            'phrases': results['final_phrases'],
            'phrase_to_segments': results['phrase_to_segments'],
            'metadata': results['processing_metadata']
        }
        with open(phrases_path, 'w', encoding='utf-8') as f:
            json.dump(phrases_data, f, ensure_ascii=False, indent=2)
        
        # Save processing summary
        summary_path = output_path / "processing_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Enhanced Text Processing Summary\n")
            f.write("=" * 40 + "\n\n")
            
            metadata = results['processing_metadata']
            f.write(f"Total documents processed: {metadata['total_documents']}\n")
            f.write(f"Languages detected: {', '.join(metadata['languages_detected'])}\n")
            f.write(f"Initial phrase candidates: {metadata['total_candidates']}\n")
            f.write(f"After static filtering: {metadata['after_static_filtering']}\n")
            f.write(f"Dynamic stopwords identified: {metadata['dynamic_stopwords_count']}\n")
            f.write(f"Final phrases: {metadata['final_phrases_count']}\n\n")
            
            f.write("Dynamic Stopwords:\n")
            f.write("-" * 20 + "\n")
            for stopword in sorted(results['dynamic_stopwords']):
                f.write(f"  {stopword}\n")
        
        self.logger.info(f"Results saved to {output_dir}")