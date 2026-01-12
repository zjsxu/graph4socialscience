"""
核心数据模型

定义了系统中使用的所有核心数据结构，包括文档、词组、图结构等。
这些数据类遵循需求5.1中定义的TOC文档格式和需求10.5中的可追溯性要求。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import json
from datetime import datetime
import numpy as np
import scipy.sparse


@dataclass
class TOCDocument:
    """
    TOC文档数据结构
    
    根据需求5.1，接受TOC分段输出的JSON作为输入，
    包含segment_id/title/level/order/text字段。
    """
    segment_id: str
    title: str
    level: int
    order: int
    text: str
    state: Optional[str] = None
    language: Optional[str] = None
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> 'TOCDocument':
        """从JSON数据创建TOCDocument实例"""
        required_fields = ['segment_id', 'title', 'level', 'order', 'text']
        
        # 验证必需字段
        for field in required_fields:
            if field not in json_data:
                raise ValueError(f"Missing required field: {field}")
        
        return cls(
            segment_id=json_data['segment_id'],
            title=json_data['title'],
            level=int(json_data['level']),
            order=int(json_data['order']),
            text=json_data['text'],
            state=json_data.get('state'),
            language=json_data.get('language')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'segment_id': self.segment_id,
            'title': self.title,
            'level': self.level,
            'order': self.order,
            'text': self.text,
            'state': self.state,
            'language': self.language
        }


@dataclass
class Window:
    """
    共现窗口数据结构
    
    根据需求5.2，每个segment的text视为一个共现窗口。
    """
    window_id: str
    phrases: List[str]
    source_doc: str
    state: str
    segment_id: str
    
    def __len__(self) -> int:
        """返回窗口中的短语数量"""
        return len(self.phrases)
    
    def is_empty(self) -> bool:
        """检查窗口是否为空"""
        return len(self.phrases) == 0


@dataclass
class Phrase:
    """
    词组数据结构
    
    包含词组文本、频率、TF-IDF分数和统计分数等信息。
    """
    text: str
    frequency: int = 0
    tfidf_score: float = 0.0
    statistical_scores: Dict[str, float] = field(default_factory=dict)
    is_stopword: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.statistical_scores:
            self.statistical_scores = {}
    
    def add_statistical_score(self, metric: str, score: float) -> None:
        """添加统计分数"""
        self.statistical_scores[metric] = score
    
    def get_statistical_score(self, metric: str) -> Optional[float]:
        """获取统计分数"""
        return self.statistical_scores.get(metric)


@dataclass
class ProcessedDocument:
    """
    处理后的文档数据结构
    
    包含原始文档、清洗后的文本、分词结果、词组和窗口信息。
    """
    original_doc: TOCDocument
    cleaned_text: str
    tokens: List[str]
    phrases: List[str]
    windows: List[Window]
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.processing_metadata:
            self.processing_metadata = {
                'processed_at': datetime.now().isoformat(),
                'token_count': len(self.tokens),
                'phrase_count': len(self.phrases),
                'window_count': len(self.windows)
            }
    
    def get_phrase_count(self) -> int:
        """获取词组数量"""
        return len(self.phrases)
    
    def get_window_count(self) -> int:
        """获取窗口数量"""
        return len(self.windows)


@dataclass
class GlobalGraph:
    """
    全局图数据结构
    
    根据需求2.1和2.3，包含从所有文档构建的全局共现网络。
    """
    vocabulary: Dict[str, int]  # phrase -> node_id
    reverse_vocabulary: Dict[int, str]  # node_id -> phrase
    cooccurrence_matrix: Optional[scipy.sparse.csr_matrix] = None
    easygraph_instance: Optional[Any] = None  # easygraph.Graph
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.metadata:
            self.metadata = {
                'created_at': datetime.now().isoformat(),
                'node_count': len(self.vocabulary),
                'vocabulary_size': len(self.vocabulary)
            }
        
        # 验证词表映射的一致性
        if len(self.vocabulary) != len(self.reverse_vocabulary):
            raise ValueError("Vocabulary and reverse vocabulary size mismatch")
    
    def get_node_id(self, phrase: str) -> Optional[int]:
        """获取词组对应的节点ID"""
        return self.vocabulary.get(phrase)
    
    def get_phrase(self, node_id: int) -> Optional[str]:
        """获取节点ID对应的词组"""
        return self.reverse_vocabulary.get(node_id)
    
    def get_node_count(self) -> int:
        """获取节点数量"""
        return len(self.vocabulary)
    
    def has_phrase(self, phrase: str) -> bool:
        """检查是否包含指定词组"""
        return phrase in self.vocabulary
    
    def add_phrase(self, phrase: str) -> int:
        """添加新词组并返回节点ID"""
        if phrase in self.vocabulary:
            return self.vocabulary[phrase]
        
        node_id = len(self.vocabulary)
        self.vocabulary[phrase] = node_id
        self.reverse_vocabulary[node_id] = phrase
        
        # 更新元数据
        self.metadata['node_count'] = len(self.vocabulary)
        self.metadata['vocabulary_size'] = len(self.vocabulary)
        
        return node_id


@dataclass
class StateSubgraph:
    """
    州级子图数据结构
    
    根据需求6.1和6.2，通过激活掩码从总图中提取的特定州/地区子图。
    """
    state_name: str
    parent_global_graph: GlobalGraph
    activation_mask: Optional[np.ndarray] = None
    easygraph_instance: Optional[Any] = None  # easygraph.Graph
    node_positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    statistics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.metadata:
            self.metadata = {
                'created_at': datetime.now().isoformat(),
                'state_name': self.state_name,
                'parent_graph_nodes': self.parent_global_graph.get_node_count()
            }
    
    def get_active_nodes(self) -> Set[int]:
        """获取激活的节点集合"""
        if self.activation_mask is None:
            return set()
        
        return set(np.where(self.activation_mask)[0])
    
    def is_node_active(self, node_id: int) -> bool:
        """检查节点是否激活"""
        if self.activation_mask is None:
            return False
        
        if node_id >= len(self.activation_mask):
            return False
        
        return bool(self.activation_mask[node_id])
    
    def get_node_position(self, node_id: int) -> Optional[Tuple[float, float]]:
        """获取节点位置"""
        return self.node_positions.get(node_id)
    
    def set_node_position(self, node_id: int, position: Tuple[float, float]) -> None:
        """设置节点位置"""
        self.node_positions[node_id] = position
    
    def get_statistic(self, metric: str) -> Optional[float]:
        """获取统计指标"""
        return self.statistics.get(metric)
    
    def set_statistic(self, metric: str, value: float) -> None:
        """设置统计指标"""
        self.statistics[metric] = value


# 辅助函数
def validate_toc_json(json_data: Dict[str, Any]) -> bool:
    """
    验证TOC JSON数据格式
    
    根据需求5.1验证JSON格式的正确性。
    """
    required_fields = ['segment_id', 'title', 'level', 'order', 'text']
    
    for field in required_fields:
        if field not in json_data:
            return False
    
    # 验证数据类型
    try:
        int(json_data['level'])
        int(json_data['order'])
        str(json_data['segment_id'])
        str(json_data['title'])
        str(json_data['text'])
    except (ValueError, TypeError):
        return False
    
    return True


def create_phrase_mapping(phrases: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    创建词组到节点ID的映射
    
    根据需求5.4确保节点映射的唯一性。
    """
    vocabulary = {}
    reverse_vocabulary = {}
    
    for i, phrase in enumerate(sorted(set(phrases))):
        vocabulary[phrase] = i
        reverse_vocabulary[i] = phrase
    
    return vocabulary, reverse_vocabulary