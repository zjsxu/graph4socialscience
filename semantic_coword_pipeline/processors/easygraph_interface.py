"""
EasyGraph/OpenRank兼容接口

根据需求9.3、9.4、9.5、9.6实现与EasyGraph/OpenRank框架的兼容接口。
主要功能包括：
- 创建标准化图数据格式输出
- 实现多视图图支持接口
- 添加图融合数据接口
- 确保EasyGraph格式兼容性
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Easy-Graph'))

from typing import Dict, List, Set, Tuple, Any, Optional, Union
import numpy as np
import scipy.sparse
import json
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
from collections import defaultdict

try:
    import easygraph as eg
except ImportError:
    print("Warning: EasyGraph not available. Some functionality may be limited.")
    eg = None

from ..core.data_models import (
    GlobalGraph, 
    StateSubgraph, 
    ProcessedDocument
)
from ..core.logger import setup_logger


class GraphFormat(Enum):
    """支持的图数据格式"""
    EASYGRAPH = "easygraph"
    NETWORKX = "networkx"
    ADJACENCY_MATRIX = "adjacency_matrix"
    EDGE_LIST = "edge_list"
    GRAPHML = "graphml"
    GML = "gml"
    JSON = "json"


class FusionStrategy(Enum):
    """图融合策略"""
    UNION = "union"
    INTERSECTION = "intersection"
    CONSENSUS = "consensus"
    WEIGHTED = "weighted"


@dataclass
class MultiViewGraph:
    """
    多视图图数据结构
    
    根据需求9.3，支持多视图图（embedding图、共现图、影响力图）在同一节点空间下融合。
    """
    node_space: Dict[str, int]  # 统一节点空间：phrase -> node_id
    reverse_node_space: Dict[int, str]  # 反向映射：node_id -> phrase
    views: Dict[str, Any] = field(default_factory=dict)  # 视图名称 -> 图对象
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.metadata:
            self.metadata = {
                'created_at': datetime.now().isoformat(),
                'node_count': len(self.node_space),
                'view_count': len(self.views)
            }
    
    def add_view(self, view_name: str, graph: Any, view_type: str = "unknown") -> None:
        """添加视图"""
        self.views[view_name] = {
            'graph': graph,
            'type': view_type,
            'added_at': datetime.now().isoformat()
        }
        self.metadata['view_count'] = len(self.views)
    
    def get_view(self, view_name: str) -> Optional[Any]:
        """获取视图"""
        view_data = self.views.get(view_name)
        return view_data['graph'] if view_data else None
    
    def list_views(self) -> List[str]:
        """列出所有视图名称"""
        return list(self.views.keys())
    
    def validate_node_space_consistency(self) -> bool:
        """验证节点空间一致性"""
        return len(self.node_space) == len(self.reverse_node_space)


@dataclass
class FusionResult:
    """
    图融合结果
    
    根据需求9.4，提供union/consensus/加权融合的数据接口。
    """
    fused_graph: Any  # 融合后的图
    fusion_strategy: FusionStrategy
    source_views: List[str]
    fusion_weights: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.metadata:
            self.metadata = {
                'created_at': datetime.now().isoformat(),
                'fusion_strategy': self.fusion_strategy.value,
                'source_view_count': len(self.source_views)
            }


class EasyGraphInterface:
    """
    EasyGraph/OpenRank兼容接口
    
    根据需求9.5和9.6，输出与EasyGraph/OpenRank框架兼容的图数据格式，
    为图融合策略验证提供可复用的实验设计。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化EasyGraph接口
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # 配置参数
        self.default_format = GraphFormat(config.get('default_format', 'easygraph'))
        self.preserve_node_attributes = config.get('preserve_node_attributes', True)
        self.preserve_edge_attributes = config.get('preserve_edge_attributes', True)
        self.validate_compatibility = config.get('validate_compatibility', True)
        
        self.logger.info(f"EasyGraphInterface initialized with format: {self.default_format.value}")
    
    def export_global_graph(self, global_graph: GlobalGraph, 
                          output_format: GraphFormat = None,
                          output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        导出全局图为标准化格式
        
        根据需求9.5，输出与EasyGraph框架兼容的图数据格式。
        
        Args:
            global_graph: 全局图对象
            output_format: 输出格式
            output_path: 输出路径
            
        Returns:
            Dict[str, Any]: 导出结果信息
        """
        if output_format is None:
            output_format = self.default_format
        
        self.logger.info(f"Exporting global graph to format: {output_format.value}")
        
        export_result = {
            'format': output_format.value,
            'node_count': global_graph.get_node_count(),
            'exported_at': datetime.now().isoformat()
        }
        
        try:
            if output_format == GraphFormat.EASYGRAPH:
                result = self._export_to_easygraph(global_graph, output_path)
            elif output_format == GraphFormat.ADJACENCY_MATRIX:
                result = self._export_to_adjacency_matrix(global_graph, output_path)
            elif output_format == GraphFormat.EDGE_LIST:
                result = self._export_to_edge_list(global_graph, output_path)
            elif output_format == GraphFormat.GRAPHML:
                result = self._export_to_graphml(global_graph, output_path)
            elif output_format == GraphFormat.JSON:
                result = self._export_to_json(global_graph, output_path)
            else:
                raise ValueError(f"Unsupported export format: {output_format.value}")
            
            export_result.update(result)
            export_result['success'] = True
            
        except Exception as e:
            self.logger.error(f"Failed to export global graph: {e}")
            export_result.update({
                'success': False,
                'error': str(e)
            })
        
        return export_result
    
    def create_multi_view_graph(self, base_graph: GlobalGraph, 
                              additional_views: Dict[str, Any] = None) -> MultiViewGraph:
        """
        创建多视图图
        
        根据需求9.3，支持多视图图在同一节点空间下融合。
        
        Args:
            base_graph: 基础图（通常是共现图）
            additional_views: 额外的视图字典
            
        Returns:
            MultiViewGraph: 多视图图对象
        """
        self.logger.info("Creating multi-view graph")
        
        # 使用基础图的节点空间作为统一节点空间
        multi_view = MultiViewGraph(
            node_space=base_graph.vocabulary.copy(),
            reverse_node_space=base_graph.reverse_vocabulary.copy()
        )
        
        # 添加基础视图（共现图）
        if base_graph.easygraph_instance is not None:
            multi_view.add_view("cooccurrence", base_graph.easygraph_instance, "cooccurrence")
        
        # 添加额外视图
        if additional_views:
            for view_name, view_data in additional_views.items():
                if isinstance(view_data, dict):
                    graph = view_data.get('graph')
                    view_type = view_data.get('type', 'unknown')
                else:
                    graph = view_data
                    view_type = 'unknown'
                
                # 验证节点空间一致性
                if self._validate_view_compatibility(graph, multi_view.node_space):
                    multi_view.add_view(view_name, graph, view_type)
                else:
                    self.logger.warning(f"View {view_name} has incompatible node space, skipping")
        
        self.logger.info(f"Created multi-view graph with {len(multi_view.views)} views")
        return multi_view
    
    def fuse_graphs(self, multi_view: MultiViewGraph, 
                   strategy: FusionStrategy,
                   view_weights: Optional[Dict[str, float]] = None,
                   target_views: Optional[List[str]] = None) -> FusionResult:
        """
        图融合功能
        
        根据需求9.4，提供union/consensus/加权融合的数据接口。
        
        Args:
            multi_view: 多视图图对象
            strategy: 融合策略
            view_weights: 视图权重（用于加权融合）
            target_views: 目标视图列表（如果为None则融合所有视图）
            
        Returns:
            FusionResult: 融合结果
        """
        if eg is None:
            raise ImportError("EasyGraph is required for graph fusion")
        
        self.logger.info(f"Fusing graphs with strategy: {strategy.value}")
        
        # 确定要融合的视图
        if target_views is None:
            target_views = multi_view.list_views()
        
        # 验证视图存在性
        available_views = []
        for view_name in target_views:
            if view_name in multi_view.views:
                available_views.append(view_name)
            else:
                self.logger.warning(f"View {view_name} not found, skipping")
        
        if len(available_views) < 2:
            raise ValueError("At least 2 views are required for fusion")
        
        # 执行融合
        try:
            if strategy == FusionStrategy.UNION:
                fused_graph = self._fuse_union(multi_view, available_views)
            elif strategy == FusionStrategy.INTERSECTION:
                fused_graph = self._fuse_intersection(multi_view, available_views)
            elif strategy == FusionStrategy.CONSENSUS:
                fused_graph = self._fuse_consensus(multi_view, available_views)
            elif strategy == FusionStrategy.WEIGHTED:
                if view_weights is None:
                    # 默认等权重
                    view_weights = {view: 1.0 / len(available_views) for view in available_views}
                fused_graph = self._fuse_weighted(multi_view, available_views, view_weights)
            else:
                raise ValueError(f"Unsupported fusion strategy: {strategy.value}")
            
            result = FusionResult(
                fused_graph=fused_graph,
                fusion_strategy=strategy,
                source_views=available_views,
                fusion_weights=view_weights
            )
            
            self.logger.info(f"Successfully fused {len(available_views)} views")
            return result
            
        except Exception as e:
            self.logger.error(f"Graph fusion failed: {e}")
            raise
    
    def validate_easygraph_compatibility(self, graph: Any) -> Dict[str, bool]:
        """
        验证EasyGraph兼容性
        
        根据需求9.5，确保输出的图数据能够被EasyGraph框架正确加载和处理。
        
        Args:
            graph: 要验证的图对象
            
        Returns:
            Dict[str, bool]: 验证结果
        """
        validation_results = {}
        
        if eg is None:
            validation_results['easygraph_available'] = False
            return validation_results
        
        validation_results['easygraph_available'] = True
        
        try:
            # 检查是否为EasyGraph对象
            # 使用更安全的方式检查类型，避免递归问题
            is_easygraph_instance = (
                hasattr(graph, '__class__') and 
                hasattr(graph.__class__, '__name__') and
                graph.__class__.__name__ == 'Graph' and
                hasattr(graph.__class__, '__module__') and
                'easygraph' in str(graph.__class__.__module__)
            )
            validation_results['is_easygraph_instance'] = is_easygraph_instance
            
            if validation_results['is_easygraph_instance']:
                # 检查基本图属性
                validation_results['has_nodes'] = len(graph.nodes) > 0
                validation_results['has_edges'] = len(graph.edges) >= 0
                
                # 检查节点属性
                if self.preserve_node_attributes and len(graph.nodes) > 0:
                    sample_node = list(graph.nodes)[0]
                    node_data = graph.nodes.get(sample_node, {})
                    validation_results['has_node_attributes'] = len(node_data) > 0
                else:
                    validation_results['has_node_attributes'] = True
                
                # 检查边属性
                if self.preserve_edge_attributes and len(graph.edges) > 0:
                    sample_edge = list(graph.edges)[0]
                    edge_data = graph.edges.get(sample_edge, {})
                    validation_results['has_edge_attributes'] = len(edge_data) > 0
                else:
                    validation_results['has_edge_attributes'] = True
                
                # 检查图的连通性
                try:
                    validation_results['is_connected'] = eg.is_connected(graph)
                except:
                    validation_results['is_connected'] = False
                
                # 检查图的基本统计信息
                try:
                    validation_results['valid_statistics'] = True
                    _ = len(graph.nodes)
                    _ = len(graph.edges)
                    _ = graph.degree()
                except:
                    validation_results['valid_statistics'] = False
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            validation_results['validation_error'] = str(e)
        
        return validation_results
    
    def create_fusion_experiment_design(self, multi_view: MultiViewGraph) -> Dict[str, Any]:
        """
        创建图融合实验设计
        
        根据需求9.6，为图融合策略验证提供可复用的实验设计。
        
        Args:
            multi_view: 多视图图对象
            
        Returns:
            Dict[str, Any]: 实验设计配置
        """
        self.logger.info("Creating fusion experiment design")
        
        available_views = multi_view.list_views()
        
        experiment_design = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'node_count': len(multi_view.node_space),
                'available_views': available_views,
                'view_count': len(available_views)
            },
            'fusion_strategies': [],
            'evaluation_metrics': [],
            'baseline_comparisons': [],
            'parameter_grids': {}
        }
        
        # 定义融合策略实验
        for strategy in FusionStrategy:
            strategy_config = {
                'strategy': strategy.value,
                'description': self._get_strategy_description(strategy),
                'parameters': self._get_strategy_parameters(strategy, available_views)
            }
            experiment_design['fusion_strategies'].append(strategy_config)
        
        # 定义评估指标
        experiment_design['evaluation_metrics'] = [
            {
                'name': 'modularity',
                'description': '模块度，衡量社群结构质量',
                'type': 'graph_structure'
            },
            {
                'name': 'clustering_coefficient',
                'description': '聚类系数，衡量局部连接密度',
                'type': 'graph_structure'
            },
            {
                'name': 'average_path_length',
                'description': '平均路径长度，衡量图的紧密性',
                'type': 'graph_structure'
            },
            {
                'name': 'node_coverage',
                'description': '节点覆盖率，衡量融合后保留的节点比例',
                'type': 'fusion_quality'
            },
            {
                'name': 'edge_preservation',
                'description': '边保留率，衡量融合后保留的边比例',
                'type': 'fusion_quality'
            }
        ]
        
        # 定义基线对比
        experiment_design['baseline_comparisons'] = [
            {
                'name': 'single_view_baseline',
                'description': '单视图基线，使用最大的单个视图作为基线',
                'method': 'largest_view'
            },
            {
                'name': 'random_fusion',
                'description': '随机融合基线，随机选择边进行融合',
                'method': 'random'
            }
        ]
        
        # 定义参数网格
        if FusionStrategy.WEIGHTED in [FusionStrategy(s['strategy']) for s in experiment_design['fusion_strategies']]:
            weight_combinations = self._generate_weight_combinations(available_views)
            experiment_design['parameter_grids']['weighted_fusion'] = {
                'view_weights': weight_combinations
            }
        
        self.logger.info(f"Created experiment design with {len(experiment_design['fusion_strategies'])} strategies")
        return experiment_design
    
    # 私有方法
    
    def _export_to_easygraph(self, global_graph: GlobalGraph, output_path: Optional[str]) -> Dict[str, Any]:
        """导出为EasyGraph格式"""
        if global_graph.easygraph_instance is None:
            raise ValueError("No EasyGraph instance available in global graph")
        
        result = {
            'graph_object': global_graph.easygraph_instance,
            'node_count': len(global_graph.easygraph_instance.nodes),
            'edge_count': len(global_graph.easygraph_instance.edges)
        }
        
        if output_path:
            # 保存为pickle文件
            with open(output_path, 'wb') as f:
                pickle.dump(global_graph.easygraph_instance, f)
            result['file_path'] = output_path
        
        return result
    
    def _export_to_adjacency_matrix(self, global_graph: GlobalGraph, output_path: Optional[str]) -> Dict[str, Any]:
        """导出为邻接矩阵格式"""
        if global_graph.cooccurrence_matrix is None:
            raise ValueError("No cooccurrence matrix available in global graph")
        
        matrix = global_graph.cooccurrence_matrix
        result = {
            'matrix': matrix,
            'shape': matrix.shape,
            'nnz': matrix.nnz,
            'format': 'sparse_csr'
        }
        
        if output_path:
            scipy.sparse.save_npz(output_path, matrix)
            result['file_path'] = output_path
        
        return result
    
    def _export_to_edge_list(self, global_graph: GlobalGraph, output_path: Optional[str]) -> Dict[str, Any]:
        """导出为边列表格式"""
        edge_list = []
        
        if global_graph.easygraph_instance is not None:
            # 从EasyGraph实例提取边列表
            for edge in global_graph.easygraph_instance.edges:
                u, v = edge
                weight = global_graph.easygraph_instance.edges.get(edge, {}).get('weight', 1.0)
                edge_list.append({
                    'source': u,
                    'target': v,
                    'weight': weight,
                    'source_phrase': global_graph.reverse_vocabulary.get(u, f'node_{u}'),
                    'target_phrase': global_graph.reverse_vocabulary.get(v, f'node_{v}')
                })
        elif global_graph.cooccurrence_matrix is not None:
            # 从共现矩阵提取边列表
            coo_matrix = global_graph.cooccurrence_matrix.tocoo()
            for i, j, weight in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
                if i < j:  # 避免重复（无向图）
                    edge_list.append({
                        'source': int(i),
                        'target': int(j),
                        'weight': float(weight),
                        'source_phrase': global_graph.reverse_vocabulary.get(i, f'node_{i}'),
                        'target_phrase': global_graph.reverse_vocabulary.get(j, f'node_{j}')
                    })
        
        result = {
            'edge_list': edge_list,
            'edge_count': len(edge_list)
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(edge_list, f, ensure_ascii=False, indent=2)
            result['file_path'] = output_path
        
        return result
    
    def _export_to_graphml(self, global_graph: GlobalGraph, output_path: Optional[str]) -> Dict[str, Any]:
        """导出为GraphML格式"""
        if global_graph.easygraph_instance is None:
            raise ValueError("EasyGraph instance required for GraphML export")
        
        if eg is None:
            raise ImportError("EasyGraph is required for GraphML export")
        
        result = {
            'format': 'graphml',
            'node_count': len(global_graph.easygraph_instance.nodes),
            'edge_count': len(global_graph.easygraph_instance.edges)
        }
        
        if output_path:
            try:
                eg.write_graphml(global_graph.easygraph_instance, output_path)
                result['file_path'] = output_path
                result['success'] = True
            except Exception as e:
                result['success'] = False
                result['error'] = str(e)
        
        return result
    
    def _export_to_json(self, global_graph: GlobalGraph, output_path: Optional[str]) -> Dict[str, Any]:
        """导出为JSON格式"""
        # 创建节点列表
        nodes = []
        for phrase, node_id in global_graph.vocabulary.items():
            nodes.append({
                'id': node_id,
                'phrase': phrase,
                'type': 'phrase_node'
            })
        
        # 创建边列表
        edges = []
        if global_graph.cooccurrence_matrix is not None:
            coo_matrix = global_graph.cooccurrence_matrix.tocoo()
            for i, j, weight in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
                if i < j:  # 避免重复
                    edges.append({
                        'source': int(i),
                        'target': int(j),
                        'weight': float(weight),
                        'type': 'cooccurrence'
                    })
        
        graph_data = {
            'metadata': {
                'format': 'json_graph',
                'created_at': datetime.now().isoformat(),
                'node_count': len(nodes),
                'edge_count': len(edges)
            },
            'nodes': nodes,
            'edges': edges,
            'vocabulary': global_graph.vocabulary
        }
        
        result = {
            'graph_data': graph_data,
            'node_count': len(nodes),
            'edge_count': len(edges)
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            result['file_path'] = output_path
        
        return result
    
    def _validate_view_compatibility(self, graph: Any, node_space: Dict[str, int]) -> bool:
        """验证视图与节点空间的兼容性"""
        if eg is None or not isinstance(graph, eg.Graph):
            return False
        
        # 检查节点ID是否在节点空间范围内
        graph_nodes = set(graph.nodes)
        valid_node_ids = set(node_space.values())
        
        return graph_nodes.issubset(valid_node_ids)
    
    def _fuse_union(self, multi_view: MultiViewGraph, view_names: List[str]) -> Any:
        """联合融合策略"""
        if eg is None:
            raise ImportError("EasyGraph is required for fusion")
        
        fused_graph = eg.Graph()
        
        # 添加所有节点
        for node_id in multi_view.reverse_node_space.keys():
            phrase = multi_view.reverse_node_space[node_id]
            fused_graph.add_node(node_id, phrase=phrase)
        
        # 联合所有边
        all_edges = set()
        for view_name in view_names:
            view_graph = multi_view.get_view(view_name)
            if view_graph and hasattr(view_graph, 'edges'):
                for edge in view_graph.edges:
                    all_edges.add(edge)
        
        # 添加边到融合图
        for edge in all_edges:
            u, v = edge
            fused_graph.add_edge(u, v, fusion_type='union')
        
        return fused_graph
    
    def _fuse_intersection(self, multi_view: MultiViewGraph, view_names: List[str]) -> Any:
        """交集融合策略"""
        if eg is None:
            raise ImportError("EasyGraph is required for fusion")
        
        if len(view_names) < 2:
            raise ValueError("At least 2 views required for intersection")
        
        # 找到所有视图的边交集
        edge_sets = []
        for view_name in view_names:
            view_graph = multi_view.get_view(view_name)
            if view_graph and hasattr(view_graph, 'edges'):
                edge_sets.append(set(view_graph.edges))
        
        if not edge_sets:
            raise ValueError("No valid views found for intersection")
        
        # 计算交集
        common_edges = edge_sets[0]
        for edge_set in edge_sets[1:]:
            common_edges = common_edges.intersection(edge_set)
        
        # 创建融合图
        fused_graph = eg.Graph()
        
        # 添加所有节点
        for node_id in multi_view.reverse_node_space.keys():
            phrase = multi_view.reverse_node_space[node_id]
            fused_graph.add_node(node_id, phrase=phrase)
        
        # 添加交集边
        for edge in common_edges:
            u, v = edge
            fused_graph.add_edge(u, v, fusion_type='intersection')
        
        return fused_graph
    
    def _fuse_consensus(self, multi_view: MultiViewGraph, view_names: List[str]) -> Any:
        """共识融合策略"""
        if eg is None:
            raise ImportError("EasyGraph is required for fusion")
        
        # 统计每条边在多少个视图中出现
        edge_counts = defaultdict(int)
        for view_name in view_names:
            view_graph = multi_view.get_view(view_name)
            if view_graph and hasattr(view_graph, 'edges'):
                for edge in view_graph.edges:
                    edge_counts[edge] += 1
        
        # 设置共识阈值（超过一半的视图包含该边）
        consensus_threshold = len(view_names) / 2
        
        # 创建融合图
        fused_graph = eg.Graph()
        
        # 添加所有节点
        for node_id in multi_view.reverse_node_space.keys():
            phrase = multi_view.reverse_node_space[node_id]
            fused_graph.add_node(node_id, phrase=phrase)
        
        # 添加共识边
        for edge, count in edge_counts.items():
            if count > consensus_threshold:
                u, v = edge
                fused_graph.add_edge(u, v, fusion_type='consensus', consensus_score=count/len(view_names))
        
        return fused_graph
    
    def _fuse_weighted(self, multi_view: MultiViewGraph, view_names: List[str], 
                      weights: Dict[str, float]) -> Any:
        """加权融合策略"""
        if eg is None:
            raise ImportError("EasyGraph is required for fusion")
        
        # 收集所有边及其权重
        edge_weights = defaultdict(float)
        
        for view_name in view_names:
            view_weight = weights.get(view_name, 0.0)
            view_graph = multi_view.get_view(view_name)
            
            if view_graph and hasattr(view_graph, 'edges') and view_weight > 0:
                for edge in view_graph.edges:
                    # 获取边的原始权重
                    original_weight = view_graph.edges.get(edge, {}).get('weight', 1.0)
                    # 应用视图权重
                    edge_weights[edge] += original_weight * view_weight
        
        # 创建融合图
        fused_graph = eg.Graph()
        
        # 添加所有节点
        for node_id in multi_view.reverse_node_space.keys():
            phrase = multi_view.reverse_node_space[node_id]
            fused_graph.add_node(node_id, phrase=phrase)
        
        # 添加加权边
        for edge, weight in edge_weights.items():
            u, v = edge
            fused_graph.add_edge(u, v, weight=weight, fusion_type='weighted')
        
        return fused_graph
    
    def _get_strategy_description(self, strategy: FusionStrategy) -> str:
        """获取策略描述"""
        descriptions = {
            FusionStrategy.UNION: "联合所有视图的边，保留所有连接",
            FusionStrategy.INTERSECTION: "只保留在所有视图中都存在的边",
            FusionStrategy.CONSENSUS: "保留在大多数视图中存在的边",
            FusionStrategy.WEIGHTED: "根据视图权重加权融合边"
        }
        return descriptions.get(strategy, "未知策略")
    
    def _get_strategy_parameters(self, strategy: FusionStrategy, available_views: List[str]) -> Dict[str, Any]:
        """获取策略参数"""
        if strategy == FusionStrategy.WEIGHTED:
            return {
                'view_weights': {view: 1.0 / len(available_views) for view in available_views},
                'weight_normalization': True
            }
        elif strategy == FusionStrategy.CONSENSUS:
            return {
                'consensus_threshold': 0.5,
                'min_view_count': 2
            }
        else:
            return {}
    
    def _generate_weight_combinations(self, view_names: List[str]) -> List[Dict[str, float]]:
        """生成权重组合"""
        combinations = []
        
        # 等权重
        equal_weight = 1.0 / len(view_names)
        combinations.append({view: equal_weight for view in view_names})
        
        # 单视图主导
        for dominant_view in view_names:
            weights = {view: 0.1 for view in view_names}
            weights[dominant_view] = 0.7
            # 重新归一化
            total = sum(weights.values())
            weights = {view: w/total for view, w in weights.items()}
            combinations.append(weights)
        
        return combinations


# 辅助函数
def create_easygraph_from_matrix(adjacency_matrix: scipy.sparse.csr_matrix, 
                                vocabulary: Dict[str, int]) -> Optional[Any]:
    """
    从邻接矩阵创建EasyGraph实例
    
    Args:
        adjacency_matrix: 邻接矩阵
        vocabulary: 词表映射
        
    Returns:
        Optional[easygraph.Graph]: EasyGraph实例，如果EasyGraph不可用则返回None
    """
    if eg is None:
        return None
    
    graph = eg.Graph()
    
    # 添加节点
    reverse_vocab = {v: k for k, v in vocabulary.items()}
    for node_id in range(adjacency_matrix.shape[0]):
        phrase = reverse_vocab.get(node_id, f'node_{node_id}')
        graph.add_node(node_id, phrase=phrase)
    
    # 添加边
    coo_matrix = adjacency_matrix.tocoo()
    for i, j, weight in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
        if i < j:  # 避免重复添加边
            graph.add_edge(i, j, weight=float(weight))
    
    return graph


def validate_multi_view_consistency(multi_view: MultiViewGraph) -> Dict[str, bool]:
    """
    验证多视图图的一致性
    
    Args:
        multi_view: 多视图图对象
        
    Returns:
        Dict[str, bool]: 验证结果
    """
    results = {}
    
    # 验证节点空间一致性
    results['node_space_consistent'] = multi_view.validate_node_space_consistency()
    
    # 验证所有视图的节点空间兼容性
    all_compatible = True
    for view_name, view_data in multi_view.views.items():
        view_graph = view_data['graph']
        if hasattr(view_graph, 'nodes'):
            view_nodes = set(view_graph.nodes)
            valid_nodes = set(multi_view.node_space.values())
            if not view_nodes.issubset(valid_nodes):
                all_compatible = False
                break
    
    results['all_views_compatible'] = all_compatible
    
    # 验证视图数量
    results['has_multiple_views'] = len(multi_view.views) >= 2
    
    return results