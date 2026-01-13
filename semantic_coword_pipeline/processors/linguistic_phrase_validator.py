"""
Linguistic Phrase Validator for Semantic Co-occurrence Network Analysis

This module implements strict POS-based phrase gating and dependency-based phrase construction
to ensure only linguistically valid noun phrases become graph nodes.

CORE REQUIREMENTS:
1. POS-based phrase gating (MANDATORY)
2. Dependency-based phrase construction (MANDATORY) 
3. Strict filtering order: Linguistic → Lexical → TF-IDF
4. Validation rules to prevent invalid phrases

Author: Semantic Co-word Network Analysis Research Team
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass

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
    class Matcher:
        pass


@dataclass
class ValidatedPhrase:
    """Linguistically validated phrase candidate"""
    text: str
    tokens: List[str]
    pos_tags: List[str]
    head_token: str
    head_pos: str
    dependency_relations: List[str]
    segment_id: str
    state: str
    validation_passed: bool = True
    validation_reasons: List[str] = None
    
    def __post_init__(self):
        if self.validation_reasons is None:
            self.validation_reasons = []


class POSBasedPhraseGate:
    """
    MANDATORY: POS-based phrase gating
    
    Ensures that a phrase is kept only if ALL of the following conditions are satisfied:
    1. The syntactic head of the phrase MUST have POS ∈ {NOUN, PROPN}
    2. The phrase MUST contain at least one token with POS ∈ {NOUN, PROPN}
    3. The phrase MUST NOT have its head token with POS ∈ {PRON, ADV, VERB}
    4. Single-token phrases are allowed ONLY if the token POS ∈ {NOUN, PROPN}
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define valid and invalid POS tags
        self.valid_head_pos = {'NOUN', 'PROPN'}
        self.valid_content_pos = {'NOUN', 'PROPN', 'ADJ'}  # ADJ allowed as modifiers
        self.invalid_head_pos = {'PRON', 'ADV', 'VERB', 'AUX'}
        
        # Specific rejected POS patterns
        self.rejected_patterns = {
            'pronouns': {'PRON'},
            'adverbs': {'ADV'},
            'verbs': {'VERB', 'AUX'},
            'gerunds': {'VERB'}  # Will be caught by dependency analysis
        }
    
    def validate_phrase(self, phrase_text: str, tokens: List[str], pos_tags: List[str], 
                       head_idx: int = -1) -> Tuple[bool, List[str]]:
        """
        Validate phrase against POS-based gating rules
        
        Args:
            phrase_text: The phrase text
            tokens: List of tokens in the phrase
            pos_tags: List of POS tags for each token
            head_idx: Index of the head token (-1 for last token as default)
            
        Returns:
            Tuple of (is_valid, reasons_for_rejection)
        """
        if not tokens or not pos_tags or len(tokens) != len(pos_tags):
            return False, ["Invalid token/POS structure"]
        
        reasons = []
        
        # Determine head token (default to last token for noun phrases)
        if head_idx == -1:
            head_idx = len(tokens) - 1
        
        if head_idx >= len(tokens):
            return False, ["Invalid head token index"]
        
        head_pos = pos_tags[head_idx]
        head_token = tokens[head_idx]
        
        # Rule 1: Head must be NOUN or PROPN
        if head_pos not in self.valid_head_pos:
            reasons.append(f"Head token '{head_token}' has invalid POS '{head_pos}' (must be NOUN/PROPN)")
        
        # Rule 2: Phrase must contain at least one NOUN/PROPN
        has_noun = any(pos in self.valid_head_pos for pos in pos_tags)
        if not has_noun:
            reasons.append("Phrase contains no NOUN/PROPN tokens")
        
        # Rule 3: Head must NOT be PRON, ADV, VERB
        if head_pos in self.invalid_head_pos:
            reasons.append(f"Head token '{head_token}' has forbidden POS '{head_pos}'")
        
        # Rule 4: Single-token phrases must be NOUN/PROPN
        if len(tokens) == 1 and pos_tags[0] not in self.valid_head_pos:
            reasons.append(f"Single token '{tokens[0]}' has invalid POS '{pos_tags[0]}' (must be NOUN/PROPN)")
        
        # Additional validation: Check for specific rejected patterns
        self._check_rejected_patterns(tokens, pos_tags, reasons)
        
        is_valid = len(reasons) == 0
        
        if not is_valid:
            self.logger.debug(f"Phrase '{phrase_text}' rejected: {'; '.join(reasons)}")
        
        return is_valid, reasons
    
    def _check_rejected_patterns(self, tokens: List[str], pos_tags: List[str], reasons: List[str]):
        """Check for specific patterns that should be rejected"""
        
        # Check for pronouns
        pronouns = [token for token, pos in zip(tokens, pos_tags) if pos == 'PRON']
        if pronouns:
            reasons.append(f"Contains pronouns: {', '.join(pronouns)}")
        
        # Check for standalone adjectives (single ADJ tokens)
        if len(tokens) == 1 and pos_tags[0] == 'ADJ':
            reasons.append(f"Standalone adjective: {tokens[0]}")
        
        # Check for adverbs
        adverbs = [token for token, pos in zip(tokens, pos_tags) if pos == 'ADV']
        if adverbs:
            reasons.append(f"Contains adverbs: {', '.join(adverbs)}")
        
        # Check for verb-only phrases
        if all(pos in {'VERB', 'AUX'} for pos in pos_tags):
            reasons.append("Verb-only phrase")
        
        # Check for gerunds (VERB tokens ending in -ing that are not part of noun phrases)
        gerunds = [token for token, pos in zip(tokens, pos_tags) 
                  if pos == 'VERB' and token.lower().endswith('ing')]
        if gerunds and not any(pos in {'NOUN', 'PROPN'} for pos in pos_tags):
            reasons.append(f"Gerund without noun context: {', '.join(gerunds)}")


class DependencyBasedPhraseConstructor:
    """
    MANDATORY: Dependency-based phrase construction
    
    Constructs phrases ONLY from linguistically valid structures using spaCy dependency parsing.
    Implements both noun chunks and dependency merges strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_phrase_length = config.get('max_phrase_length', 4)
        self.min_phrase_length = config.get('min_phrase_length', 2)
        
        # Valid dependency relations for phrase construction
        self.valid_compound_deps = {'compound', 'amod', 'nmod', 'nummod'}
        self.invalid_deps = {'advmod', 'nsubj', 'dobj', 'prep'}  # Relations to avoid
    
    def extract_noun_chunks(self, doc: Doc, segment_id: str, state: str) -> List[ValidatedPhrase]:
        """
        Strategy A: Extract noun chunks using spaCy's noun_chunks
        
        Keep a noun_chunk only if:
        - noun_chunk.root.pos_ ∈ {NOUN, PROPN}
        - noun_chunk does NOT contain any PRON tokens
        """
        phrases = []
        
        if not SPACY_AVAILABLE or doc is None:
            return phrases
        
        try:
            for chunk in doc.noun_chunks:
                # Check root POS
                if chunk.root.pos_ not in {'NOUN', 'PROPN'}:
                    continue
                
                # Check for pronouns
                has_pronoun = any(token.pos_ == 'PRON' for token in chunk)
                if has_pronoun:
                    continue
                
                # Extract phrase information
                tokens = [token.text for token in chunk]
                pos_tags = [token.pos_ for token in chunk]
                dep_relations = [token.dep_ for token in chunk]
                
                # Length filtering
                if len(tokens) < self.min_phrase_length or len(tokens) > self.max_phrase_length:
                    continue
                
                phrase_text = chunk.text.strip()
                if not phrase_text:
                    continue
                
                # Find head token index
                head_idx = -1
                for i, token in enumerate(chunk):
                    if token == chunk.root:
                        head_idx = i
                        break
                
                validated_phrase = ValidatedPhrase(
                    text=phrase_text,
                    tokens=tokens,
                    pos_tags=pos_tags,
                    head_token=chunk.root.text,
                    head_pos=chunk.root.pos_,
                    dependency_relations=dep_relations,
                    segment_id=segment_id,
                    state=state
                )
                
                phrases.append(validated_phrase)
                
        except Exception as e:
            self.logger.warning(f"Noun chunk extraction failed: {e}")
        
        return phrases
    
    def extract_dependency_merges(self, doc: Doc, segment_id: str, state: str) -> List[ValidatedPhrase]:
        """
        Strategy B: Construct phrases by merging tokens connected via valid dependency relations
        
        Merge tokens connected via:
        - compound relations (e.g., "student" → "discipline" → "student discipline")
        - amod relations (e.g., "digital" → "storage" → "digital storage")
        
        Do NOT construct phrases from:
        - advmod, nsubj/dobj without noun head
        - detached POS sequences without dependency grounding
        """
        phrases = []
        
        if not SPACY_AVAILABLE or doc is None:
            return phrases
        
        try:
            # Find valid dependency-based phrases
            processed_tokens = set()
            
            for token in doc:
                if token.i in processed_tokens:
                    continue
                
                # Only start from NOUN/PROPN tokens
                if token.pos_ not in {'NOUN', 'PROPN'}:
                    continue
                
                # Build phrase by following valid dependencies
                phrase_tokens = [token]
                phrase_indices = {token.i}
                
                # Look for modifiers (children with valid dependencies)
                for child in token.children:
                    if (child.dep_ in self.valid_compound_deps and 
                        child.pos_ in {'NOUN', 'PROPN', 'ADJ', 'NUM'} and
                        child.i not in processed_tokens):
                        phrase_tokens.append(child)
                        phrase_indices.add(child.i)
                
                # Look for head relationships (if this token modifies another noun)
                if (token.head != token and 
                    token.dep_ in self.valid_compound_deps and
                    token.head.pos_ in {'NOUN', 'PROPN'} and
                    token.head.i not in processed_tokens):
                    phrase_tokens.append(token.head)
                    phrase_indices.add(token.head.i)
                
                # Sort tokens by position in text
                phrase_tokens.sort(key=lambda t: t.i)
                
                # Length filtering
                if len(phrase_tokens) < self.min_phrase_length or len(phrase_tokens) > self.max_phrase_length:
                    continue
                
                # Check for invalid dependencies
                has_invalid_dep = any(t.dep_ in self.invalid_deps for t in phrase_tokens)
                if has_invalid_dep:
                    continue
                
                # Extract phrase information
                tokens = [t.text for t in phrase_tokens]
                pos_tags = [t.pos_ for t in phrase_tokens]
                dep_relations = [t.dep_ for t in phrase_tokens]
                phrase_text = ' '.join(tokens)
                
                # Find head token (usually the rightmost noun)
                head_token = phrase_tokens[-1]  # Default to last token
                for t in reversed(phrase_tokens):
                    if t.pos_ in {'NOUN', 'PROPN'}:
                        head_token = t
                        break
                
                validated_phrase = ValidatedPhrase(
                    text=phrase_text,
                    tokens=tokens,
                    pos_tags=pos_tags,
                    head_token=head_token.text,
                    head_pos=head_token.pos_,
                    dependency_relations=dep_relations,
                    segment_id=segment_id,
                    state=state
                )
                
                phrases.append(validated_phrase)
                
                # Mark tokens as processed
                processed_tokens.update(phrase_indices)
                
        except Exception as e:
            self.logger.warning(f"Dependency merge extraction failed: {e}")
        
        return phrases
    
    def extract_single_nouns(self, doc: Doc, segment_id: str, state: str) -> List[ValidatedPhrase]:
        """Extract single noun tokens that weren't captured in phrases"""
        phrases = []
        
        if not SPACY_AVAILABLE or doc is None:
            return phrases
        
        try:
            for token in doc:
                # Only single NOUN/PROPN tokens
                if token.pos_ not in {'NOUN', 'PROPN'}:
                    continue
                
                # Skip if token is part of a compound or has modifiers
                has_compound_children = any(child.dep_ in self.valid_compound_deps for child in token.children)
                is_compound_child = token.dep_ in self.valid_compound_deps
                
                if has_compound_children or is_compound_child:
                    continue  # Will be captured by phrase extraction
                
                # Skip very short tokens
                if len(token.text) < 2:
                    continue
                
                validated_phrase = ValidatedPhrase(
                    text=token.text,
                    tokens=[token.text],
                    pos_tags=[token.pos_],
                    head_token=token.text,
                    head_pos=token.pos_,
                    dependency_relations=[token.dep_],
                    segment_id=segment_id,
                    state=state
                )
                
                phrases.append(validated_phrase)
                
        except Exception as e:
            self.logger.warning(f"Single noun extraction failed: {e}")
        
        return phrases


class LinguisticPhraseValidator:
    """
    Main linguistic phrase validator that enforces POS-based gating and dependency-based construction
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.pos_gate = POSBasedPhraseGate()
        self.dependency_constructor = DependencyBasedPhraseConstructor(config)
        
        # Load spaCy models
        self.nlp_en = None
        self.nlp_zh = None
        
        if SPACY_AVAILABLE:
            try:
                self.nlp_en = spacy.load("en_core_web_sm")
                self.logger.info("Loaded English spaCy model for linguistic validation")
            except OSError:
                self.logger.warning("English spaCy model not found")
            
            try:
                self.nlp_zh = spacy.load("zh_core_web_sm")
                self.logger.info("Loaded Chinese spaCy model for linguistic validation")
            except OSError:
                self.logger.warning("Chinese spaCy model not found")
    
    def validate_and_extract_phrases(self, text: str, segment_id: str, state: str, language: str) -> List[ValidatedPhrase]:
        """
        Main method: Extract and validate phrases using strict linguistic rules
        
        Args:
            text: Input text
            segment_id: Segment identifier
            state: State identifier
            language: Language code ('en' or 'zh')
            
        Returns:
            List of linguistically validated phrases
        """
        if not SPACY_AVAILABLE:
            self.logger.warning("spaCy not available, using fallback validation")
            return self._fallback_validation(text, segment_id, state, language)
        
        # Choose appropriate spaCy model
        nlp = self.nlp_en if language == 'en' else self.nlp_zh
        if nlp is None:
            self.logger.warning(f"No spaCy model available for language: {language}")
            return self._fallback_validation(text, segment_id, state, language)
        
        try:
            # Process text with spaCy
            doc = nlp(text)
            
            # Extract phrases using dependency-based construction
            all_phrases = []
            
            # Strategy A: Noun chunks
            noun_chunk_phrases = self.dependency_constructor.extract_noun_chunks(doc, segment_id, state)
            all_phrases.extend(noun_chunk_phrases)
            
            # Strategy B: Dependency merges
            dependency_phrases = self.dependency_constructor.extract_dependency_merges(doc, segment_id, state)
            all_phrases.extend(dependency_phrases)
            
            # Strategy C: Single nouns (as fallback)
            single_noun_phrases = self.dependency_constructor.extract_single_nouns(doc, segment_id, state)
            all_phrases.extend(single_noun_phrases)
            
            # Apply POS-based gating to all extracted phrases
            validated_phrases = []
            for phrase in all_phrases:
                is_valid, reasons = self.pos_gate.validate_phrase(
                    phrase.text, phrase.tokens, phrase.pos_tags
                )
                
                if is_valid:
                    phrase.validation_passed = True
                    validated_phrases.append(phrase)
                else:
                    phrase.validation_passed = False
                    phrase.validation_reasons = reasons
                    self.logger.debug(f"Phrase rejected: '{phrase.text}' - {'; '.join(reasons)}")
            
            # Remove duplicates while preserving order
            unique_phrases = []
            seen_texts = set()
            for phrase in validated_phrases:
                if phrase.text not in seen_texts:
                    unique_phrases.append(phrase)
                    seen_texts.add(phrase.text)
            
            self.logger.info(f"Extracted {len(unique_phrases)} validated phrases from {len(all_phrases)} candidates")
            return unique_phrases
            
        except Exception as e:
            self.logger.error(f"Phrase validation failed: {e}")
            return self._fallback_validation(text, segment_id, state, language)
    
    def _fallback_validation(self, text: str, segment_id: str, state: str, language: str) -> List[ValidatedPhrase]:
        """
        Fallback validation when spaCy is not available
        
        Uses simple heuristics to approximate POS-based validation
        """
        phrases = []
        
        # Simple tokenization
        tokens = text.lower().split()
        
        # Generate bigrams and trigrams with basic validation
        for n in range(2, min(5, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                phrase_tokens = tokens[i:i+n]
                phrase_text = ' '.join(phrase_tokens)
                
                # Basic heuristic validation (without POS tags)
                if self._basic_phrase_validation(phrase_text, phrase_tokens):
                    validated_phrase = ValidatedPhrase(
                        text=phrase_text,
                        tokens=phrase_tokens,
                        pos_tags=['UNKNOWN'] * len(phrase_tokens),
                        head_token=phrase_tokens[-1],
                        head_pos='UNKNOWN',
                        dependency_relations=['UNKNOWN'] * len(phrase_tokens),
                        segment_id=segment_id,
                        state=state,
                        validation_passed=True
                    )
                    phrases.append(validated_phrase)
        
        # Add single tokens that look like nouns
        for token in tokens:
            if len(token) > 3 and not self._is_likely_function_word(token):
                validated_phrase = ValidatedPhrase(
                    text=token,
                    tokens=[token],
                    pos_tags=['NOUN'],  # Assume it's a noun
                    head_token=token,
                    head_pos='NOUN',
                    dependency_relations=['ROOT'],
                    segment_id=segment_id,
                    state=state,
                    validation_passed=True
                )
                phrases.append(validated_phrase)
        
        # Remove duplicates
        unique_phrases = []
        seen_texts = set()
        for phrase in phrases:
            if phrase.text not in seen_texts:
                unique_phrases.append(phrase)
                seen_texts.add(phrase.text)
        
        self.logger.info(f"Fallback validation: extracted {len(unique_phrases)} phrases")
        return unique_phrases
    
    def _basic_phrase_validation(self, phrase_text: str, tokens: List[str]) -> bool:
        """Basic validation without POS tags"""
        
        # Reject obvious invalid patterns
        invalid_patterns = [
            'someone', 'what you', 'you know', 'he said', 'she said',
            'quickly', 'frequently', 'currently', 'recently',
            'paying', 'running', 'operating', 'working'
        ]
        
        if phrase_text in invalid_patterns:
            return False
        
        # Reject if starts with common pronouns
        if tokens[0] in ['someone', 'what', 'you', 'he', 'she', 'it', 'they']:
            return False
        
        # Reject if all tokens are very short
        if all(len(token) < 3 for token in tokens):
            return False
        
        # Reject if contains obvious function words only
        function_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        if all(token in function_words for token in tokens):
            return False
        
        return True
    
    def _is_likely_function_word(self, token: str) -> bool:
        """Check if token is likely a function word"""
        function_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
            'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how'
        }
        return token.lower() in function_words
    
    def validate_existing_phrases(self, phrases: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Validate a list of existing phrases against linguistic rules
        
        Args:
            phrases: List of phrase strings to validate
            
        Returns:
            Dictionary mapping phrase to validation results
        """
        results = {}
        
        for phrase in phrases:
            # Simple tokenization for validation
            tokens = phrase.split()
            
            # Try to get POS tags using spaCy if available
            pos_tags = []
            if SPACY_AVAILABLE and self.nlp_en:
                try:
                    doc = self.nlp_en(phrase)
                    pos_tags = [token.pos_ for token in doc]
                except:
                    pos_tags = ['UNKNOWN'] * len(tokens)
            else:
                pos_tags = ['UNKNOWN'] * len(tokens)
            
            # Validate
            is_valid, reasons = self.pos_gate.validate_phrase(phrase, tokens, pos_tags)
            
            results[phrase] = {
                'is_valid': is_valid,
                'reasons': reasons,
                'tokens': tokens,
                'pos_tags': pos_tags
            }
        
        return results
    
    def get_validation_statistics(self, phrases: List[ValidatedPhrase]) -> Dict[str, Any]:
        """Get statistics about phrase validation"""
        total_phrases = len(phrases)
        valid_phrases = sum(1 for p in phrases if p.validation_passed)
        invalid_phrases = total_phrases - valid_phrases
        
        # Count rejection reasons
        rejection_reasons = {}
        for phrase in phrases:
            if not phrase.validation_passed:
                for reason in phrase.validation_reasons:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        # Count POS patterns
        pos_patterns = {}
        for phrase in phrases:
            if phrase.validation_passed:
                pattern = ' '.join(phrase.pos_tags)
                pos_patterns[pattern] = pos_patterns.get(pattern, 0) + 1
        
        return {
            'total_phrases': total_phrases,
            'valid_phrases': valid_phrases,
            'invalid_phrases': invalid_phrases,
            'validation_rate': valid_phrases / total_phrases if total_phrases > 0 else 0,
            'rejection_reasons': rejection_reasons,
            'valid_pos_patterns': pos_patterns
        }