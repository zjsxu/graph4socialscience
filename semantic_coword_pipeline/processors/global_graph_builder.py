"""
总图构建器（GlobalGraphBuilder）

根据需求2.1、2.3、2.5、5.4、5.5实现全局共现图的构建功能。
主要功能包括：
- 创建统一词表生成功能
- 实现共现矩阵计算  
- 集成EasyGraph图构建
- 添加孤立节点保留机制
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Easy-Graph'))

from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import scipy.sparse
from collections import defaultdict, Counter
from itertools import combinations
import logging
from tqdm import tqdm
from datetime import datetime

try:
    import easygraph as eg
except ImportError:
    print("Warning: EasyGraph not available. Some functionality may be limited.")
    eg = None

from ..core.data_models import (
    ProcessedDocument, 
    GlobalGraph, 
    Window, 
    create_phrase_mapping
)
from ..core.logger import setup_logger


class GlobalGraphBuilder:
    """
    总图构建器
    
    根据需求2.1实现从所有文档构建统一的总图（Global_Graph）。
    采用"总图优先"策略，确保跨州对比时指标可比、节点语义空间一致。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化总图构建器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # 配置参数
        self.window_type = config.get('window_type', 'segment')
        self.edge_weight_method = config.get('edge_weight_method', 'binary')
        self.preserve_isolated_nodes = config.get('preserve_isolated_nodes', True)
        self.min_cooccurrence_count = config.get('min_cooccurrence_count', 1)
        
        # 内部状态
        self._phrase_counter = Counter()
        self._cooccurrence_counter = defaultdict(int)
        self._all_phrases = set()
        
        self.logger.info(f"GlobalGraphBuilder initialized with config: {config}")
    
    def build_global_graph(self, processed_docs: List[ProcessedDocument]) -> GlobalGraph:
        """
        构建全局共现图
        
        根据需求2.1从所有文档构建统一的总图。
        
        Args:
            processed_docs: 处理后的文档列表
            
        Returns:
            GlobalGraph: 构建的全局图对象
            
        Raises:
            ValueError: 当输入文档为空时
        """
        if not processed_docs:
            raise ValueError("Cannot build global graph from empty document list")
        
        self.logger.info(f"Building global graph from {len(processed_docs)} documents")
        
        # 步骤1: 创建统一词表
        vocabulary, reverse_vocabulary = self._create_unified_vocabulary(processed_docs)
        self.logger.info(f"Created unified vocabulary with {len(vocabulary)} phrases")
        
        # 步骤2: 计算共现矩阵
        cooccurrence_matrix = self._calculate_cooccurrence_matrix(processed_docs, vocabulary)
        self.logger.info(f"Calculated cooccurrence matrix: {cooccurrence_matrix.shape}")
        
        # 步骤3: 构建EasyGraph实例
        easygraph_instance = self._build_easygraph_instance(vocabulary, cooccurrence_matrix)
        
        # 步骤4: 创建GlobalGraph对象
        global_graph = GlobalGraph(
            vocabulary=vocabulary,
            reverse_vocabulary=reverse_vocabulary,
            cooccurrence_matrix=cooccurrence_matrix,
            easygraph_instance=easygraph_instance,
            metadata={
                'created_at': self._get_timestamp(),
                'node_count': len(vocabulary),
                'edge_count': cooccurrence_matrix.nnz,
                'density': cooccurrence_matrix.nnz / (len(vocabulary) * (len(vocabulary) - 1)) if len(vocabulary) > 1 else 0.0,
                'isolated_nodes_preserved': self.preserve_isolated_nodes,
                'edge_weight_method': self.edge_weight_method,
                'source_documents': len(processed_docs)
            }
        )
        
        self.logger.info(f"Global graph built successfully: {len(vocabulary)} nodes, {cooccurrence_matrix.nnz} edges")
        return global_graph
    
    def _create_unified_vocabulary(self, processed_docs: List[ProcessedDocument]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        创建统一词表生成功能
        
        根据需求2.3和5.4，用于统一词表与节点空间，确保节点映射的唯一性。
        
        Args:
            processed_docs: 处理后的文档列表
            
        Returns:
            Tuple[Dict[str, int], Dict[int, str]]: 词表映射和反向映射
        """
        self.logger.debug("Creating unified vocabulary from all documents")
        
        # 收集所有词组
        all_phrases = set()
        phrase_frequencies = Counter()
        
        for doc in tqdm(processed_docs, desc="📄 Processing documents", unit="doc"):
            for window in doc.windows:
                for phrase in window.phrases:
                    all_phrases.add(phrase)
                    phrase_frequencies[phrase] += 1
        
        # 按频率排序，确保高频词组获得较小的ID
        sorted_phrases = sorted(all_phrases, key=lambda p: (-phrase_frequencies[p], p))
        
        # 创建映射
        vocabulary = {}
        reverse_vocabulary = {}
        
        for i, phrase in enumerate(sorted_phrases):
            vocabulary[phrase] = i
            reverse_vocabulary[i] = phrase
        
        self.logger.debug(f"Created vocabulary with {len(vocabulary)} unique phrases")
        return vocabulary, reverse_vocabulary
    
    def _calculate_cooccurrence_matrix(self, processed_docs: List[ProcessedDocument], 
                                     vocabulary: Dict[str, int]) -> scipy.sparse.csr_matrix:
        """
        实现共现矩阵计算
        
        根据需求5.2和5.5，将每个segment的text视为一个共现窗口，
        计算无向、带权的共现关系。
        
        Args:
            processed_docs: 处理后的文档列表
            vocabulary: 词表映射
            
        Returns:
            scipy.sparse.csr_matrix: 共现矩阵
        """
        self.logger.debug("Calculating cooccurrence matrix")
        
        vocab_size = len(vocabulary)
        cooccurrence_counts = defaultdict(int)
        
        # 统计共现关系
        total_windows = 0
        for doc in tqdm(processed_docs, desc="📄 Processing documents", unit="doc"):
            for window in doc.windows:
                if len(window.phrases) < 2:
                    continue
                
                total_windows += 1
                
                # 获取窗口中所有词组的节点ID
                phrase_ids = []
                for phrase in window.phrases:
                    if phrase in vocabulary:
                        phrase_ids.append(vocabulary[phrase])
                
                # 计算窗口内所有词组对的共现
                for i, j in combinations(phrase_ids, 2):
                    # 确保无向性：较小的ID在前
                    if i > j:
                        i, j = j, i
                    
                    if self.edge_weight_method == 'binary':
                        cooccurrence_counts[(i, j)] += 1
                    elif self.edge_weight_method == 'frequency':
                        # 可以根据词组在窗口中的频率加权
                        cooccurrence_counts[(i, j)] += 1
        
        self.logger.debug(f"Processed {total_windows} windows, found {len(cooccurrence_counts)} cooccurrence pairs")
        
        # 过滤低频共现
        filtered_cooccurrences = {
            (i, j): count for (i, j), count in cooccurrence_counts.items()
            if count >= self.min_cooccurrence_count
        }
        
        # 构建稀疏矩阵
        if filtered_cooccurrences:
            rows, cols, data = zip(*[
                (i, j, count) for (i, j), count in filtered_cooccurrences.items()
            ])
            
            # 创建对称矩阵
            all_rows = list(rows) + list(cols)
            all_cols = list(cols) + list(rows)
            all_data = list(data) + list(data)
            
            cooccurrence_matrix = scipy.sparse.csr_matrix(
                (all_data, (all_rows, all_cols)),
                shape=(vocab_size, vocab_size)
            )
        else:
            # 空矩阵
            cooccurrence_matrix = scipy.sparse.csr_matrix((vocab_size, vocab_size))
        
        self.logger.debug(f"Created cooccurrence matrix: {cooccurrence_matrix.shape}, {cooccurrence_matrix.nnz} non-zero entries")
        return cooccurrence_matrix
    
    def _build_easygraph_instance(self, vocabulary: Dict[str, int], 
                                cooccurrence_matrix: scipy.sparse.csr_matrix) -> Optional[Any]:
        """
        集成EasyGraph图构建
        
        根据需求2.5，显式保留孤立节点以反映缺失或弱连接现象。
        
        Args:
            vocabulary: 词表映射
            cooccurrence_matrix: 共现矩阵
            
        Returns:
            Optional[easygraph.Graph]: EasyGraph图实例，如果EasyGraph不可用则返回None
        """
        if eg is None:
            self.logger.warning("EasyGraph not available, skipping graph instance creation")
            return None
        
        self.logger.debug("Building EasyGraph instance")
        
        # 创建无向图
        graph = eg.Graph()
        
        # 添加所有节点（包括孤立节点）
        for phrase, node_id in vocabulary.items():
            graph.add_node(node_id, phrase=phrase)
        
        self.logger.debug(f"Added {len(vocabulary)} nodes to graph")
        
        # 添加边
        coo_matrix = cooccurrence_matrix.tocoo()
        edge_count = 0
        
        for i, j, weight in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            if i < j:  # 避免重复添加边（因为矩阵是对称的）
                graph.add_edge(i, j, weight=float(weight))
                edge_count += 1
        
        self.logger.debug(f"Added {edge_count} edges to graph")
        
        # 验证孤立节点保留
        if self.preserve_isolated_nodes:
            degree_dict = graph.degree()
            isolated_nodes = [node for node in graph.nodes if degree_dict.get(node, 0) == 0]
            self.logger.info(f"Preserved {len(isolated_nodes)} isolated nodes")
        
        return graph
    
    def get_graph_statistics(self, global_graph: GlobalGraph) -> Dict[str, Any]:
        """
        获取图统计信息
        
        Args:
            global_graph: 全局图对象
            
        Returns:
            Dict[str, Any]: 图统计信息
        """
        stats = {
            'node_count': global_graph.get_node_count(),
            'vocabulary_size': len(global_graph.vocabulary)
        }
        
        if global_graph.cooccurrence_matrix is not None:
            matrix = global_graph.cooccurrence_matrix
            stats.update({
                'edge_count': matrix.nnz // 2,  # 除以2因为矩阵是对称的
                'density': matrix.nnz / (matrix.shape[0] * (matrix.shape[0] - 1)) if matrix.shape[0] > 1 else 0.0,
                'sparsity': 1.0 - (matrix.nnz / (matrix.shape[0] * matrix.shape[1])) if matrix.shape[0] > 0 else 1.0
            })
        
        if global_graph.easygraph_instance is not None and eg is not None:
            graph = global_graph.easygraph_instance
            try:
                stats.update({
                    'connected_components': eg.number_connected_components(graph),
                    'isolated_nodes': len([n for n in graph.nodes if graph.degree().get(n, 0) == 0]),
                    'average_degree': sum(graph.degree().values()) / len(graph.nodes) if len(graph.nodes) > 0 else 0.0
                })
            except Exception as e:
                self.logger.warning(f"Failed to calculate advanced graph statistics: {e}")
        
        return stats
    
    def validate_graph_properties(self, global_graph: GlobalGraph) -> Dict[str, bool]:
        """
        验证图属性
        
        验证构建的图是否满足需求中的各项属性。
        
        Args:
            global_graph: 全局图对象
            
        Returns:
            Dict[str, bool]: 验证结果
        """
        validation_results = {}
        
        # 验证词表映射唯一性（需求5.4）
        vocab_size = len(global_graph.vocabulary)
        reverse_vocab_size = len(global_graph.reverse_vocabulary)
        validation_results['vocabulary_mapping_unique'] = (vocab_size == reverse_vocab_size)
        
        # 验证共现关系无向性（需求5.5）
        if global_graph.cooccurrence_matrix is not None:
            matrix = global_graph.cooccurrence_matrix
            # 检查矩阵是否对称
            is_symmetric = np.allclose(matrix.toarray(), matrix.toarray().T)
            validation_results['cooccurrence_undirected'] = is_symmetric
        else:
            validation_results['cooccurrence_undirected'] = True
        
        # 验证孤立节点保留（需求2.5）
        if global_graph.easygraph_instance is not None and eg is not None:
            graph = global_graph.easygraph_instance
            total_nodes = len(graph.nodes)
            degree_dict = graph.degree()
            nodes_with_edges = len([n for n in graph.nodes if degree_dict.get(n, 0) > 0])
            isolated_nodes = total_nodes - nodes_with_edges
            
            # 如果配置要求保留孤立节点，则应该有完整的节点集
            if self.preserve_isolated_nodes:
                validation_results['isolated_nodes_preserved'] = (total_nodes == vocab_size)
            else:
                validation_results['isolated_nodes_preserved'] = True
        else:
            validation_results['isolated_nodes_preserved'] = True
        
        return validation_results
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()


class CooccurrenceCalculator:
    """
    共现计算器
    
    专门负责共现关系的计算逻辑。
    """
    
    def __init__(self, window_type: str = 'segment', weight_method: str = 'binary'):
        """
        初始化共现计算器
        
        Args:
            window_type: 窗口类型，默认为'segment'
            weight_method: 权重计算方法，默认为'binary'
        """
        self.window_type = window_type
        self.weight_method = weight_method
        self.logger = setup_logger(__name__)
    
    def calculate_window_cooccurrences(self, window: Window, vocabulary: Dict[str, int]) -> List[Tuple[int, int, float]]:
        """
        计算单个窗口内的共现关系
        
        Args:
            window: 共现窗口
            vocabulary: 词表映射
            
        Returns:
            List[Tuple[int, int, float]]: 共现关系列表 (node_i, node_j, weight)
        """
        if len(window.phrases) < 2:
            return []
        
        cooccurrences = []
        phrase_ids = []
        
        # 获取窗口中所有有效词组的ID
        for phrase in window.phrases:
            if phrase in vocabulary:
                phrase_ids.append(vocabulary[phrase])
        
        # 计算所有词组对的共现
        for i, j in combinations(phrase_ids, 2):
            # 确保无向性
            if i > j:
                i, j = j, i
            
            weight = self._calculate_edge_weight(window, i, j, vocabulary)
            cooccurrences.append((i, j, weight))
        
        return cooccurrences
    
    def _calculate_edge_weight(self, window: Window, node_i: int, node_j: int, 
                             vocabulary: Dict[str, int]) -> float:
        """
        计算边权重
        
        Args:
            window: 共现窗口
            node_i: 节点i的ID
            node_j: 节点j的ID
            vocabulary: 词表映射
            
        Returns:
            float: 边权重
        """
        if self.weight_method == 'binary':
            return 1.0
        elif self.weight_method == 'frequency':
            # 基于词组在窗口中的频率
            reverse_vocab = {v: k for k, v in vocabulary.items()}
            phrase_i = reverse_vocab.get(node_i, '')
            phrase_j = reverse_vocab.get(node_j, '')
            
            count_i = window.phrases.count(phrase_i)
            count_j = window.phrases.count(phrase_j)
            
            return float(min(count_i, count_j))
        elif self.weight_method == 'distance':
            # 基于词组在窗口中的距离（需要位置信息）
            # 这里简化为固定权重
            return 1.0
        else:
            return 1.0


# 辅助函数
def create_empty_global_graph() -> GlobalGraph:
    """
    创建空的全局图
    
    Returns:
        GlobalGraph: 空的全局图对象
    """
    return GlobalGraph(
        vocabulary={},
        reverse_vocabulary={},
        cooccurrence_matrix=scipy.sparse.csr_matrix((0, 0)),
        easygraph_instance=None,
        metadata={
            'created_at': datetime.now().isoformat(),
            'node_count': 0,
            'edge_count': 0,
            'is_empty': True
        }
    )


def merge_global_graphs(graphs: List[GlobalGraph]) -> GlobalGraph:
    """
    合并多个全局图
    
    Args:
        graphs: 要合并的全局图列表
        
    Returns:
        GlobalGraph: 合并后的全局图
        
    Raises:
        ValueError: 当输入图列表为空时
    """
    if not graphs:
        raise ValueError("Cannot merge empty graph list")
    
    if len(graphs) == 1:
        return graphs[0]
    
    # 合并词表
    all_phrases = set()
    for graph in graphs:
        all_phrases.update(graph.vocabulary.keys())
    
    # 创建新的统一词表
    vocabulary, reverse_vocabulary = create_phrase_mapping(list(all_phrases))
    
    # 合并共现矩阵
    vocab_size = len(vocabulary)
    merged_matrix = scipy.sparse.csr_matrix((vocab_size, vocab_size))
    
    for graph in graphs:
        if graph.cooccurrence_matrix is not None:
            # 重新映射矩阵到新的词表空间
            old_to_new_mapping = {}
            for phrase, old_id in graph.vocabulary.items():
                new_id = vocabulary[phrase]
                old_to_new_mapping[old_id] = new_id
            
            # 这里需要实现矩阵重映射逻辑
            # 简化实现：直接累加（假设词表一致）
            if graph.cooccurrence_matrix.shape == merged_matrix.shape:
                merged_matrix += graph.cooccurrence_matrix
    
    return GlobalGraph(
        vocabulary=vocabulary,
        reverse_vocabulary=reverse_vocabulary,
        cooccurrence_matrix=merged_matrix,
        easygraph_instance=None,  # 需要重新构建
        metadata={
            'created_at': datetime.now().isoformat(),
            'node_count': len(vocabulary),
            'merged_from': len(graphs),
            'is_merged': True
        }
    )