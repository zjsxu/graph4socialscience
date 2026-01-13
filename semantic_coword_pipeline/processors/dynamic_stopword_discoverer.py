"""
动态停词发现器

实现基于TF-IDF的动态停词发现机制，用于识别高频低区分度的词组。
根据需求4.1-4.5，提供可解释的停词发现和合并功能。
"""

import logging
from tqdm import tqdm
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
import json

from ..core.data_models import ProcessedDocument, Phrase


@dataclass
class TFIDFScore:
    """TF-IDF分数数据结构"""
    phrase: str
    tf: float  # 词频
    df: int    # 文档频率
    idf: float # 逆文档频率
    tfidf: float # TF-IDF分数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'phrase': self.phrase,
            'tf': self.tf,
            'df': self.df,
            'idf': self.idf,
            'tfidf': self.tfidf
        }


@dataclass
class StopwordDiscoveryResult:
    """停词发现结果"""
    dynamic_stopwords: Set[str]
    merged_stopwords: Set[str]
    tfidf_scores: Dict[str, TFIDFScore]
    discovery_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'dynamic_stopwords': list(self.dynamic_stopwords),
            'merged_stopwords': list(self.merged_stopwords),
            'tfidf_scores': {phrase: score.to_dict() for phrase, score in self.tfidf_scores.items()},
            'discovery_metadata': self.discovery_metadata
        }


class DynamicStopwordDiscoverer:
    """
    动态停词发现器
    
    根据需求4.1-4.5实现：
    - 4.1: 计算全语料的TF-IDF指标
    - 4.2: 发现跨文档普遍高频但区分度极低的词组
    - 4.4: 将动态停词表与人工停词表合并
    - 4.5: 应用合并后的停词表作为强约束
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化动态停词发现器
        
        Args:
            config: 配置字典，包含阈值和路径设置
        """
        self.config = config
        self.tfidf_threshold = config.get('tfidf_threshold', 0.1)
        self.frequency_threshold = config.get('frequency_threshold', 0.8)
        self.min_document_frequency = config.get('min_document_frequency', 2)
        self.static_stopwords_path = config.get('static_stopwords_path', 'data/stopwords.txt')
        self.enable_dynamic_discovery = config.get('enable_dynamic_discovery', True)
        
        # 日志记录
        self.logger = logging.getLogger(__name__)
        
        # 加载静态停词表
        self.static_stopwords = self._load_static_stopwords()
        
    def _load_static_stopwords(self) -> Set[str]:
        """
        加载静态停词表
        
        Returns:
            静态停词集合
        """
        static_stopwords = set()
        
        if self.static_stopwords_path:
            stopwords_path = Path(self.static_stopwords_path)
            
            if stopwords_path.exists():
                try:
                    with open(stopwords_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            word = line.strip()
                            if word and not word.startswith('#'):
                                static_stopwords.add(word)
                    
                    self.logger.info(f"Loaded {len(static_stopwords)} static stopwords from {stopwords_path}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load static stopwords from {stopwords_path}: {e}")
            else:
                self.logger.info(f"Static stopwords file not found: {stopwords_path}")
        
        # 添加基础停词
        basic_stopwords = {
            # 英文基础停词
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            # 中文基础停词
            '的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '也', '很', '到', '说', '要', '去', '会', '着',
            '个', '上', '来', '下', '对', '从', '把', '被', '让', '使', '给', '为', '与', '及', '或', '但', '而', '且',
            '这', '那', '些', '此', '其', '他', '她', '它', '我', '你', '您', '们', '自己', '什么', '怎么', '为什么'
        }
        
        static_stopwords.update(basic_stopwords)
        
        return static_stopwords
    
    def discover_stopwords(self, processed_docs: List[ProcessedDocument]) -> StopwordDiscoveryResult:
        """
        发现动态停词
        
        Args:
            processed_docs: 处理后的文档列表
            
        Returns:
            停词发现结果
        """
        if not self.enable_dynamic_discovery:
            self.logger.info("Dynamic stopword discovery is disabled")
            return StopwordDiscoveryResult(
                dynamic_stopwords=set(),
                merged_stopwords=self.static_stopwords.copy(),
                tfidf_scores={},
                discovery_metadata={'enabled': False}
            )
        
        # 提取词组语料库
        phrase_corpus = []
        for doc in tqdm(processed_docs, desc="📄 Processing documents", unit="doc"):
            if doc.phrases:
                phrase_corpus.append(doc.phrases)
        
        if not phrase_corpus:
            self.logger.warning("No phrases found in processed documents")
            return StopwordDiscoveryResult(
                dynamic_stopwords=set(),
                merged_stopwords=self.static_stopwords.copy(),
                tfidf_scores={},
                discovery_metadata={'error': 'no_phrases'}
            )
        
        # 计算TF-IDF分数
        tfidf_scores = self.calculate_tfidf_matrix(phrase_corpus)
        
        # 识别低区分度词组
        dynamic_stopwords = self.identify_low_discrimination_phrases(tfidf_scores)
        
        # 合并停词表
        merged_stopwords = self.merge_stopword_lists(dynamic_stopwords)
        
        # 生成发现元数据
        discovery_metadata = {
            'total_documents': len(phrase_corpus),
            'total_unique_phrases': len(tfidf_scores),
            'dynamic_stopwords_count': len(dynamic_stopwords),
            'static_stopwords_count': len(self.static_stopwords),
            'merged_stopwords_count': len(merged_stopwords),
            'tfidf_threshold': self.tfidf_threshold,
            'frequency_threshold': self.frequency_threshold,
            'min_document_frequency': self.min_document_frequency
        }
        
        self.logger.info(f"Discovered {len(dynamic_stopwords)} dynamic stopwords")
        self.logger.info(f"Total merged stopwords: {len(merged_stopwords)}")
        
        return StopwordDiscoveryResult(
            dynamic_stopwords=dynamic_stopwords,
            merged_stopwords=merged_stopwords,
            tfidf_scores=tfidf_scores,
            discovery_metadata=discovery_metadata
        )
    
    def calculate_tfidf_matrix(self, phrase_corpus: List[List[str]]) -> Dict[str, TFIDFScore]:
        """
        计算TF-IDF矩阵
        
        根据需求4.1，计算全语料的TF-IDF指标。
        
        Args:
            phrase_corpus: 词组语料库，每个文档是一个词组列表
            
        Returns:
            词组到TF-IDF分数的映射
        """
        if not phrase_corpus:
            return {}
        
        # 统计文档频率 (DF)
        document_frequency = Counter()
        total_documents = len(phrase_corpus)
        
        # 统计每个词组在多少个文档中出现
        for doc_phrases in phrase_corpus:
            unique_phrases = set(doc_phrases)
            for phrase in unique_phrases:
                document_frequency[phrase] += 1
        
        # 过滤低频词组
        filtered_phrases = {
            phrase: df for phrase, df in document_frequency.items()
            if df >= self.min_document_frequency
        }
        
        # 计算TF-IDF分数
        tfidf_scores = {}
        
        for phrase, df in filtered_phrases.items():
            # 计算平均词频 (TF)
            total_tf = 0
            for doc_phrases in phrase_corpus:
                phrase_count = doc_phrases.count(phrase)
                if phrase_count > 0:
                    doc_length = len(doc_phrases)
                    if doc_length > 0:
                        total_tf += phrase_count / doc_length
            
            avg_tf = total_tf / total_documents if total_documents > 0 else 0
            
            # 计算逆文档频率 (IDF)
            idf = math.log(total_documents / df) if df > 0 else 0
            
            # 计算TF-IDF
            tfidf = avg_tf * idf
            
            tfidf_scores[phrase] = TFIDFScore(
                phrase=phrase,
                tf=avg_tf,
                df=df,
                idf=idf,
                tfidf=tfidf
            )
        
        self.logger.info(f"Calculated TF-IDF for {len(tfidf_scores)} phrases")
        
        return tfidf_scores
    
    def identify_low_discrimination_phrases(self, tfidf_scores: Dict[str, TFIDFScore]) -> Set[str]:
        """
        识别低区分度词组
        
        根据需求4.2，发现跨文档普遍高频但区分度极低的词组。
        
        Args:
            tfidf_scores: TF-IDF分数字典
            
        Returns:
            低区分度词组集合
        """
        if not tfidf_scores:
            return set()
        
        dynamic_stopwords = set()
        
        # 计算频率阈值
        all_dfs = [score.df for score in tfidf_scores.values()]
        max_df = max(all_dfs) if all_dfs else 0
        frequency_cutoff = max_df * self.frequency_threshold
        
        for phrase, score in tfidf_scores.items():
            # 条件1: TF-IDF分数低于阈值（低区分度）
            low_tfidf = score.tfidf < self.tfidf_threshold
            
            # 条件2: 文档频率高于阈值（高频）
            high_frequency = score.df >= frequency_cutoff
            
            # 条件3: 不是已知的静态停词
            not_static_stopword = phrase not in self.static_stopwords
            
            # 必须同时满足所有条件
            if low_tfidf and high_frequency and not_static_stopword:
                dynamic_stopwords.add(phrase)
                self.logger.debug(f"Identified dynamic stopword: {phrase} (TF-IDF: {score.tfidf:.4f}, DF: {score.df}, cutoff: {frequency_cutoff:.1f})")
        
        self.logger.info(f"Identified {len(dynamic_stopwords)} low discrimination phrases")
        
        return dynamic_stopwords
    
    def merge_stopword_lists(self, dynamic_stopwords: Set[str]) -> Set[str]:
        """
        合并静态和动态停词表
        
        根据需求4.4，将动态停词表与人工停词表合并。
        
        Args:
            dynamic_stopwords: 动态发现的停词集合
            
        Returns:
            合并后的停词集合
        """
        merged_stopwords = self.static_stopwords.copy()
        merged_stopwords.update(dynamic_stopwords)
        
        self.logger.info(f"Merged stopwords: {len(self.static_stopwords)} static + {len(dynamic_stopwords)} dynamic = {len(merged_stopwords)} total")
        
        return merged_stopwords
    
    def apply_stopword_filter(self, phrases: List[str], stopwords: Optional[Set[str]] = None) -> List[str]:
        """
        应用停词过滤
        
        根据需求4.5，应用合并后的停词表作为强约束。
        
        Args:
            phrases: 待过滤的词组列表
            stopwords: 停词集合，如果为None则使用静态停词
            
        Returns:
            过滤后的词组列表
        """
        if stopwords is None:
            stopwords = self.static_stopwords
        
        filtered_phrases = []
        for phrase in tqdm(phrases, desc="🔍 Processing phrases", unit="phrase"):
            if phrase not in stopwords:
                filtered_phrases.append(phrase)
        
        removed_count = len(phrases) - len(filtered_phrases)
        if removed_count > 0:
            self.logger.debug(f"Filtered out {removed_count} stopword phrases")
        
        return filtered_phrases
    
    def get_stopword_explanation(self, phrase: str, tfidf_scores: Dict[str, TFIDFScore]) -> Dict[str, Any]:
        """
        获取停词的可解释信息
        
        根据需求4.3，形成可解释的动态stopword列表。
        
        Args:
            phrase: 词组
            tfidf_scores: TF-IDF分数字典
            
        Returns:
            停词解释信息
        """
        explanation = {
            'phrase': phrase,
            'is_static_stopword': phrase in self.static_stopwords,
            'is_dynamic_stopword': False,
            'reason': []
        }
        
        if phrase in tfidf_scores:
            score = tfidf_scores[phrase]
            explanation.update({
                'tfidf_score': score.tfidf,
                'document_frequency': score.df,
                'term_frequency': score.tf,
                'inverse_document_frequency': score.idf
            })
            
            # 判断是否为动态停词
            max_df = max([s.df for s in tfidf_scores.values()]) if tfidf_scores else 0
            frequency_cutoff = max_df * self.frequency_threshold
            
            if score.tfidf < self.tfidf_threshold:
                explanation['reason'].append(f"Low TF-IDF score ({score.tfidf:.4f} < {self.tfidf_threshold})")
                
            if score.df >= frequency_cutoff:
                explanation['reason'].append(f"High document frequency ({score.df} >= {frequency_cutoff:.1f})")
                
            explanation['is_dynamic_stopword'] = (
                score.tfidf < self.tfidf_threshold and 
                score.df >= frequency_cutoff and 
                phrase not in self.static_stopwords
            )
        
        if phrase in self.static_stopwords:
            explanation['reason'].append("Present in static stopword list")
        
        return explanation
    
    def save_stopword_analysis(self, result: StopwordDiscoveryResult, output_path: str) -> None:
        """
        保存停词分析结果
        
        Args:
            result: 停词发现结果
            output_path: 输出文件路径
        """
        try:
            output_data = result.to_dict()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved stopword analysis to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save stopword analysis: {e}")
            raise