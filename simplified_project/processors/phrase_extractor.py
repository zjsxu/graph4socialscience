"""
词组抽取器模块

实现词组/短语抽取功能，包括英文2-gram抽取、中文短语抽取和统计约束筛选。
根据需求3.1、3.2、3.3、3.6实现词组级节点单位升级。
"""

import re
import math
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass

# 导入NLTK
try:
    import nltk
    from nltk.util import ngrams
    from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, using basic n-gram extraction")

from ..core.data_models import ProcessedDocument, Phrase, Window
from ..core.config import Config
from ..core.error_handler import ErrorHandler


@dataclass
class StatisticalScores:
    """统计分数数据结构"""
    mutual_information: float = 0.0
    t_score: float = 0.0
    cohesion_score: float = 0.0
    frequency: int = 0
    left_entropy: float = 0.0
    right_entropy: float = 0.0


@dataclass
class PhraseCandidate:
    """词组候选数据结构"""
    text: str
    tokens: List[str]
    frequency: int
    statistical_scores: StatisticalScores
    language: str
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.tokens:
            self.tokens = self.text.split()


class EnglishBigramExtractor:
    """
    英文2-gram抽取器
    
    根据需求3.1，采用2-gram（bigram）作为主要节点候选。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_frequency = config.get('min_phrase_frequency', 3)
        self.use_nltk_collocations = config.get('use_nltk_collocations', True) and NLTK_AVAILABLE
        
        # 过滤模式
        self.filter_patterns = [
            r'^[^a-zA-Z]',  # 不以字母开头
            r'[^a-zA-Z]$',  # 不以字母结尾
            r'^\d+$',       # 纯数字
            r'^.{1,2}$'     # 太短的词组
        ]
        
        logging.debug(f"EnglishBigramExtractor initialized with min_frequency={self.min_frequency}")
    
    def extract_bigrams(self, tokens: List[str]) -> List[PhraseCandidate]:
        """
        抽取英文2-gram
        
        Args:
            tokens: 分词结果
            
        Returns:
            List[PhraseCandidate]: 2-gram候选列表
        """
        if len(tokens) < 2:
            return []
        
        try:
            if self.use_nltk_collocations:
                return self._extract_with_nltk(tokens)
            else:
                return self._extract_basic_bigrams(tokens)
        except Exception as e:
            logging.warning(f"Bigram extraction failed: {e}, falling back to basic method")
            return self._extract_basic_bigrams(tokens)
    
    def _extract_with_nltk(self, tokens: List[str]) -> List[PhraseCandidate]:
        """
        使用NLTK的搭配发现器抽取2-gram
        
        Args:
            tokens: 分词结果
            
        Returns:
            List[PhraseCandidate]: 2-gram候选列表
        """
        # 创建搭配发现器
        finder = BigramCollocationFinder.from_words(tokens)
        
        # 应用频率过滤
        finder.apply_freq_filter(self.min_frequency)
        
        # 获取2-gram及其频率
        bigram_freq = finder.ngram_fd
        
        candidates = []
        for (word1, word2), freq in bigram_freq.items():
            bigram_text = f"{word1} {word2}"
            
            # 应用过滤规则
            if self._should_filter_bigram(bigram_text):
                continue
            
            # 创建候选
            candidate = PhraseCandidate(
                text=bigram_text,
                tokens=[word1, word2],
                frequency=freq,
                statistical_scores=StatisticalScores(frequency=freq),
                language='english'
            )
            
            candidates.append(candidate)
        
        logging.debug(f"Extracted {len(candidates)} English bigrams using NLTK")
        return candidates
    
    def _extract_basic_bigrams(self, tokens: List[str]) -> List[PhraseCandidate]:
        """
        基础2-gram抽取方法
        
        Args:
            tokens: 分词结果
            
        Returns:
            List[PhraseCandidate]: 2-gram候选列表
        """
        # 生成所有2-gram
        if NLTK_AVAILABLE:
            bigrams = list(ngrams(tokens, 2))
        else:
            bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        
        # 统计频率
        bigram_counts = Counter(bigrams)
        
        candidates = []
        for (word1, word2), freq in bigram_counts.items():
            if freq < self.min_frequency:
                continue
            
            bigram_text = f"{word1} {word2}"
            
            # 应用过滤规则
            if self._should_filter_bigram(bigram_text):
                continue
            
            # 创建候选
            candidate = PhraseCandidate(
                text=bigram_text,
                tokens=[word1, word2],
                frequency=freq,
                statistical_scores=StatisticalScores(frequency=freq),
                language='english'
            )
            
            candidates.append(candidate)
        
        logging.debug(f"Extracted {len(candidates)} English bigrams using basic method")
        return candidates
    
    def _should_filter_bigram(self, bigram_text: str) -> bool:
        """
        检查是否应该过滤该2-gram
        
        Args:
            bigram_text: 2-gram文本
            
        Returns:
            bool: 是否应该过滤
        """
        for pattern in self.filter_patterns:
            if re.search(pattern, bigram_text):
                return True
        
        # 检查是否包含停词（基础停词）
        basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = bigram_text.lower().split()
        if len(words) == 2 and (words[0] in basic_stopwords or words[1] in basic_stopwords):
            return True
        
        return False


class ChinesePhraseExtractor:
    """
    中文短语抽取器
    
    根据需求3.3，采用分词后再进行短语抽取。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_frequency = config.get('min_phrase_frequency', 3)
        self.max_phrase_length = config.get('max_phrase_length', 4)
        self.min_phrase_length = config.get('min_phrase_length', 2)
        
        # 中文字符模式
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        
        # 过滤模式
        self.filter_patterns = [
            r'^\d+$',       # 纯数字
            r'^[a-zA-Z]+$', # 纯英文
            r'^.{1}$'       # 单字符
        ]
        
        logging.debug(f"ChinesePhraseExtractor initialized with min_frequency={self.min_frequency}")
    
    def extract_phrases(self, tokens: List[str]) -> List[PhraseCandidate]:
        """
        抽取中文短语
        
        Args:
            tokens: 分词结果
            
        Returns:
            List[PhraseCandidate]: 中文短语候选列表
        """
        if len(tokens) < self.min_phrase_length:
            return []
        
        try:
            # 生成不同长度的n-gram
            all_ngrams = []
            
            for n in range(self.min_phrase_length, min(self.max_phrase_length + 1, len(tokens) + 1)):
                if NLTK_AVAILABLE:
                    n_grams = list(ngrams(tokens, n))
                else:
                    n_grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                
                all_ngrams.extend(n_grams)
            
            # 统计频率
            ngram_counts = Counter(all_ngrams)
            
            candidates = []
            for ngram_tuple, freq in ngram_counts.items():
                if freq < self.min_frequency:
                    continue
                
                phrase_text = ''.join(ngram_tuple)  # 中文不需要空格连接
                
                # 应用过滤规则
                if self._should_filter_phrase(phrase_text):
                    continue
                
                # 创建候选
                candidate = PhraseCandidate(
                    text=phrase_text,
                    tokens=list(ngram_tuple),
                    frequency=freq,
                    statistical_scores=StatisticalScores(frequency=freq),
                    language='chinese'
                )
                
                candidates.append(candidate)
            
            logging.debug(f"Extracted {len(candidates)} Chinese phrases")
            return candidates
            
        except Exception as e:
            logging.error(f"Chinese phrase extraction failed: {e}")
            return []
    
    def _should_filter_phrase(self, phrase_text: str) -> bool:
        """
        检查是否应该过滤该中文短语
        
        Args:
            phrase_text: 短语文本
            
        Returns:
            bool: 是否应该过滤
        """
        # 应用基础过滤规则
        for pattern in self.filter_patterns:
            if re.search(pattern, phrase_text):
                return True
        
        # 检查是否包含足够的中文字符
        chinese_chars = len(self.chinese_pattern.findall(phrase_text))
        if chinese_chars < len(phrase_text) * 0.5:  # 至少50%是中文字符
            return True
        
        # 检查基础停词
        basic_stopwords = {'的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '也', '很', '到', '说', '要', '去', '会', '着'}
        if phrase_text in basic_stopwords:
            return True
        
        return False


class StatisticalFilter:
    """
    统计约束筛选器
    
    根据需求3.2，结合统计约束（互信息、t-score或凝固度）筛选高质量短语。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 阈值设置
        self.mi_threshold = config.get('mutual_information_threshold', 0.0)
        self.t_score_threshold = config.get('t_score_threshold', 2.0)
        self.cohesion_threshold = config.get('cohesion_threshold', 0.0)
        
        # 启用的过滤器
        self.use_mutual_information = config.get('use_mutual_information', True)
        self.use_t_score = config.get('use_t_score', True)
        self.use_cohesion = config.get('use_cohesion', False)
        
        logging.debug("StatisticalFilter initialized")
    
    def filter_phrases(self, candidates: List[PhraseCandidate], corpus_stats: Dict[str, Any]) -> List[PhraseCandidate]:
        """
        应用统计约束筛选短语
        
        Args:
            candidates: 候选短语列表
            corpus_stats: 语料库统计信息
            
        Returns:
            List[PhraseCandidate]: 筛选后的短语列表
        """
        if not candidates:
            return []
        
        try:
            # 计算统计分数
            for candidate in candidates:
                self._calculate_statistical_scores(candidate, corpus_stats)
            
            # 应用过滤条件
            filtered_candidates = []
            for candidate in candidates:
                if self._passes_statistical_filters(candidate):
                    filtered_candidates.append(candidate)
            
            logging.debug(f"Statistical filtering: {len(candidates)} -> {len(filtered_candidates)} candidates")
            return filtered_candidates
            
        except Exception as e:
            logging.error(f"Statistical filtering failed: {e}")
            return candidates  # 返回原始候选列表
    
    def _calculate_statistical_scores(self, candidate: PhraseCandidate, corpus_stats: Dict[str, Any]) -> None:
        """
        计算统计分数
        
        Args:
            candidate: 候选短语
            corpus_stats: 语料库统计信息
        """
        try:
            # 获取必要的统计信息
            total_tokens = corpus_stats.get('total_tokens', 1)
            token_counts = corpus_stats.get('token_counts', {})
            bigram_counts = corpus_stats.get('bigram_counts', {})
            
            if len(candidate.tokens) == 2:
                # 计算2-gram的统计分数
                word1, word2 = candidate.tokens
                
                # 频率信息
                freq_w1 = token_counts.get(word1, 1)
                freq_w2 = token_counts.get(word2, 1)
                freq_w1_w2 = candidate.frequency
                
                # 互信息 (Mutual Information)
                if self.use_mutual_information:
                    mi = self._calculate_mutual_information(freq_w1, freq_w2, freq_w1_w2, total_tokens)
                    candidate.statistical_scores.mutual_information = mi
                
                # t-score
                if self.use_t_score:
                    t_score = self._calculate_t_score(freq_w1, freq_w2, freq_w1_w2, total_tokens)
                    candidate.statistical_scores.t_score = t_score
                
                # 凝固度 (Cohesion)
                if self.use_cohesion:
                    cohesion = self._calculate_cohesion_score(candidate, corpus_stats)
                    candidate.statistical_scores.cohesion_score = cohesion
            
            else:
                # 对于更长的短语，使用简化的统计方法
                candidate.statistical_scores.mutual_information = math.log(candidate.frequency + 1)
                candidate.statistical_scores.t_score = math.sqrt(candidate.frequency)
                candidate.statistical_scores.cohesion_score = candidate.frequency / total_tokens
                
        except Exception as e:
            logging.warning(f"Failed to calculate statistical scores for '{candidate.text}': {e}")
            # 设置默认分数
            candidate.statistical_scores.mutual_information = 0.0
            candidate.statistical_scores.t_score = 0.0
            candidate.statistical_scores.cohesion_score = 0.0
    
    def _calculate_mutual_information(self, freq_w1: int, freq_w2: int, freq_w1_w2: int, total_tokens: int) -> float:
        """
        计算互信息
        
        MI(w1, w2) = log(P(w1, w2) / (P(w1) * P(w2)))
        """
        try:
            p_w1 = freq_w1 / total_tokens
            p_w2 = freq_w2 / total_tokens
            p_w1_w2 = freq_w1_w2 / (total_tokens - 1)  # 减1因为是bigram
            
            if p_w1 > 0 and p_w2 > 0 and p_w1_w2 > 0:
                mi = math.log(p_w1_w2 / (p_w1 * p_w2))
                return mi
            else:
                return 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _calculate_t_score(self, freq_w1: int, freq_w2: int, freq_w1_w2: int, total_tokens: int) -> float:
        """
        计算t-score
        
        t-score = (P(w1, w2) - P(w1) * P(w2)) / sqrt(P(w1, w2) / N)
        """
        try:
            p_w1 = freq_w1 / total_tokens
            p_w2 = freq_w2 / total_tokens
            p_w1_w2 = freq_w1_w2 / (total_tokens - 1)
            
            if p_w1_w2 > 0:
                numerator = p_w1_w2 - (p_w1 * p_w2)
                denominator = math.sqrt(p_w1_w2 / (total_tokens - 1))
                
                if denominator > 0:
                    t_score = numerator / denominator
                    return t_score
                else:
                    return 0.0
            else:
                return 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def _calculate_cohesion_score(self, candidate: PhraseCandidate, corpus_stats: Dict[str, Any]) -> float:
        """
        计算凝固度分数
        
        简化的凝固度计算，基于左右邻接词的熵
        """
        try:
            # 这里使用简化的凝固度计算
            # 实际实现中可以基于左右邻接词的分布计算熵
            return candidate.frequency / corpus_stats.get('total_tokens', 1)
        except Exception:
            return 0.0
    
    def _passes_statistical_filters(self, candidate: PhraseCandidate) -> bool:
        """
        检查候选短语是否通过统计过滤条件
        
        Args:
            candidate: 候选短语
            
        Returns:
            bool: 是否通过过滤
        """
        scores = candidate.statistical_scores
        
        # 互信息过滤
        if self.use_mutual_information and scores.mutual_information < self.mi_threshold:
            return False
        
        # t-score过滤
        if self.use_t_score and scores.t_score < self.t_score_threshold:
            return False
        
        # 凝固度过滤
        if self.use_cohesion and scores.cohesion_score < self.cohesion_threshold:
            return False
        
        return True


class PhraseExtractor:
    """
    词组抽取器主类
    
    根据需求3.1、3.2、3.3、3.6实现词组/短语抽取功能。
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化词组抽取器
        
        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.error_handler = ErrorHandler(self.config.get_section('error_handling'))
        
        # 获取词组抽取配置
        phrase_config = self.config.get_section('phrase_extraction')
        
        # 初始化子组件
        self.english_extractor = EnglishBigramExtractor(phrase_config)
        self.chinese_extractor = ChinesePhraseExtractor(phrase_config)
        self.statistical_filter = StatisticalFilter(phrase_config)
        
        # 扩展选项
        self.enable_multiword_extension = phrase_config.get('enable_multiword_extension', True)
        self.max_phrase_length = phrase_config.get('max_phrase_length', 4)
        
        logging.info("PhraseExtractor initialized successfully")
    
    def extract_phrases_from_document(self, processed_doc: ProcessedDocument) -> ProcessedDocument:
        """
        从处理后的文档中抽取短语
        
        Args:
            processed_doc: 处理后的文档
            
        Returns:
            ProcessedDocument: 包含短语信息的文档
        """
        try:
            language = processed_doc.original_doc.language or 'english'
            tokens = processed_doc.tokens
            
            if not tokens:
                logging.warning(f"No tokens found in document {processed_doc.original_doc.segment_id}")
                return processed_doc
            
            # 根据语言选择抽取方法
            if language == 'english':
                candidates = self.english_extractor.extract_bigrams(tokens)
            elif language == 'chinese':
                candidates = self.chinese_extractor.extract_phrases(tokens)
            else:
                logging.warning(f"Unknown language {language}, using English extractor")
                candidates = self.english_extractor.extract_bigrams(tokens)
            
            # 转换为短语列表
            phrases = [candidate.text for candidate in candidates]
            
            # 更新文档的短语信息
            processed_doc.phrases = phrases
            
            # 更新窗口中的短语信息
            if processed_doc.windows:
                processed_doc.windows[0].phrases = phrases
            
            # 更新元数据
            processed_doc.processing_metadata.update({
                'phrase_count': len(phrases),
                'phrase_extraction_method': language,
                'candidates_before_filtering': len(candidates)
            })
            
            logging.debug(f"Extracted {len(phrases)} phrases from document {processed_doc.original_doc.segment_id}")
            return processed_doc
            
        except Exception as e:
            error_msg = f"Failed to extract phrases from document {processed_doc.original_doc.segment_id}: {e}"
            logging.error(error_msg)
            return self.error_handler.handle_processing_error(e, f"extract_phrases:{processed_doc.original_doc.segment_id}")
    
    def extract_phrases_from_tokens(self, tokens: List[str], language: str = 'english') -> List[str]:
        """
        从分词结果中抽取短语
        
        Args:
            tokens: 分词结果
            language: 语言类型
            
        Returns:
            List[str]: 短语列表
        """
        try:
            if language == 'english':
                candidates = self.english_extractor.extract_bigrams(tokens)
            elif language == 'chinese':
                candidates = self.chinese_extractor.extract_phrases(tokens)
            else:
                candidates = self.english_extractor.extract_bigrams(tokens)
            
            return [candidate.text for candidate in candidates]
            
        except Exception as e:
            logging.error(f"Failed to extract phrases from tokens: {e}")
            return []
    
    def apply_statistical_filtering(self, candidates: List[PhraseCandidate], corpus_stats: Dict[str, Any]) -> List[PhraseCandidate]:
        """
        应用统计约束筛选
        
        Args:
            candidates: 候选短语列表
            corpus_stats: 语料库统计信息
            
        Returns:
            List[PhraseCandidate]: 筛选后的短语列表
        """
        try:
            return self.statistical_filter.filter_phrases(candidates, corpus_stats)
        except Exception as e:
            logging.error(f"Statistical filtering failed: {e}")
            return candidates
    
    def batch_extract_phrases(self, processed_docs: List[ProcessedDocument]) -> List[ProcessedDocument]:
        """
        批量抽取短语
        
        Args:
            processed_docs: 处理后的文档列表
            
        Returns:
            List[ProcessedDocument]: 包含短语信息的文档列表
        """
        updated_docs = []
        
        for i, doc in enumerate(processed_docs):
            try:
                updated_doc = self.extract_phrases_from_document(doc)
                updated_docs.append(updated_doc)
                
                if (i + 1) % 100 == 0:
                    logging.info(f"Extracted phrases from {i + 1}/{len(processed_docs)} documents")
                    
            except Exception as e:
                logging.error(f"Failed to extract phrases from document {doc.original_doc.segment_id}: {e}")
                # 继续处理其他文档
                updated_docs.append(doc)
                continue
        
        logging.info(f"Batch phrase extraction completed: {len(updated_docs)} documents processed")
        return updated_docs
    
    def calculate_corpus_statistics(self, processed_docs: List[ProcessedDocument]) -> Dict[str, Any]:
        """
        计算语料库统计信息
        
        Args:
            processed_docs: 处理后的文档列表
            
        Returns:
            Dict[str, Any]: 语料库统计信息
        """
        try:
            all_tokens = []
            all_bigrams = []
            
            for doc in processed_docs:
                all_tokens.extend(doc.tokens)
                
                # 生成bigrams用于统计
                tokens = doc.tokens
                if len(tokens) >= 2:
                    if NLTK_AVAILABLE:
                        bigrams = list(ngrams(tokens, 2))
                    else:
                        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
                    all_bigrams.extend(bigrams)
            
            # 统计频率
            token_counts = Counter(all_tokens)
            bigram_counts = Counter(all_bigrams)
            
            stats = {
                'total_tokens': len(all_tokens),
                'unique_tokens': len(token_counts),
                'total_bigrams': len(all_bigrams),
                'unique_bigrams': len(bigram_counts),
                'token_counts': dict(token_counts),
                'bigram_counts': dict(bigram_counts),
                'average_tokens_per_doc': len(all_tokens) / len(processed_docs) if processed_docs else 0
            }
            
            logging.debug(f"Calculated corpus statistics: {stats['total_tokens']} tokens, {stats['unique_tokens']} unique tokens")
            return stats
            
        except Exception as e:
            logging.error(f"Failed to calculate corpus statistics: {e}")
            return {}
    
    def get_extraction_statistics(self, processed_docs: List[ProcessedDocument]) -> Dict[str, Any]:
        """
        获取短语抽取统计信息
        
        Args:
            processed_docs: 处理后的文档列表
            
        Returns:
            Dict[str, Any]: 抽取统计信息
        """
        if not processed_docs:
            return {}
        
        # 语言分布
        language_counts = {}
        total_phrases = 0
        phrase_length_dist = Counter()
        
        for doc in processed_docs:
            lang = doc.original_doc.language or 'unknown'
            language_counts[lang] = language_counts.get(lang, 0) + 1
            
            phrases = doc.phrases
            total_phrases += len(phrases)
            
            # 统计短语长度分布
            for phrase in phrases:
                length = len(phrase.split()) if doc.original_doc.language == 'english' else len(phrase)
                phrase_length_dist[length] += 1
        
        return {
            'total_documents': len(processed_docs),
            'language_distribution': language_counts,
            'total_phrases': total_phrases,
            'average_phrases_per_doc': total_phrases / len(processed_docs),
            'phrase_length_distribution': dict(phrase_length_dist),
            'unique_phrases': len(set(phrase for doc in processed_docs for phrase in doc.phrases))
        }