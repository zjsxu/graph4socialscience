"""
EasyGraph接口测试

测试EasyGraph/OpenRank兼容接口的功能，包括：
- 标准化图数据格式输出
- 多视图图支持接口
- 图融合数据接口
- EasyGraph格式兼容性
"""

import pytest
import numpy as np
import scipy.sparse
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from semantic_coword_pipeline.processors.easygraph_interface import (
    EasyGraphInterface,
    MultiViewGraph,
    FusionResult,
    GraphFormat,
    FusionStrategy,
    create_easygraph_from_matrix,
    validate_multi_view_consistency
)
from semantic_coword_pipeline.core.data_models import GlobalGraph


class TestEasyGraphInterface:
    """EasyGraph接口测试类"""
    
    @pytest.fixture
    def sample_config(self):
        """测试配置"""
        return {
            'default_format': 'easygraph',
            'preserve_node_attributes': True,
            'preserve_edge_attributes': True,
            'validate_compatibility': True
        }
    
    @pytest.fixture
    def sample_vocabulary(self):
        """测试词表"""
        return {
            'phrase_1': 0,
            'phrase_2': 1,
            'phrase_3': 2,
            'phrase_4': 3
        }
    
    @pytest.fixture
    def sample_global_graph(self, sample_vocabulary):
        """测试全局图"""
        reverse_vocab = {v: k for k, v in sample_vocabulary.items()}
        
        # 创建稀疏矩阵
        matrix = scipy.sparse.csr_matrix([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ])
        
        return GlobalGraph(
            vocabulary=sample_vocabulary,
            reverse_vocabulary=reverse_vocab,
            cooccurrence_matrix=matrix,
            easygraph_instance=None,  # 将在需要时模拟
            metadata={'test': True}
        )
    
    @pytest.fixture
    def easygraph_interface(self, sample_config):
        """EasyGraph接口实例"""
        return EasyGraphInterface(sample_config)
    
    def test_interface_initialization(self, sample_config):
        """测试接口初始化"""
        interface = EasyGraphInterface(sample_config)
        
        assert interface.default_format == GraphFormat.EASYGRAPH
        assert interface.preserve_node_attributes is True
        assert interface.preserve_edge_attributes is True
        assert interface.validate_compatibility is True
    
    def test_export_to_adjacency_matrix(self, easygraph_interface, sample_global_graph):
        """测试导出为邻接矩阵格式"""
        result = easygraph_interface._export_to_adjacency_matrix(sample_global_graph, None)
        
        assert 'matrix' in result
        assert 'shape' in result
        assert 'nnz' in result
        assert result['format'] == 'sparse_csr'
        assert result['shape'] == (4, 4)
        assert result['nnz'] == 8  # 4个边，每个边在对称矩阵中出现2次
    
    def test_export_to_edge_list(self, easygraph_interface, sample_global_graph):
        """测试导出为边列表格式"""
        result = easygraph_interface._export_to_edge_list(sample_global_graph, None)
        
        assert 'edge_list' in result
        assert 'edge_count' in result
        assert isinstance(result['edge_list'], list)
        assert result['edge_count'] > 0
        
        # 检查边列表格式
        if result['edge_list']:
            edge = result['edge_list'][0]
            assert 'source' in edge
            assert 'target' in edge
            assert 'weight' in edge
            assert 'source_phrase' in edge
            assert 'target_phrase' in edge
    
    def test_export_to_json(self, easygraph_interface, sample_global_graph):
        """测试导出为JSON格式"""
        result = easygraph_interface._export_to_json(sample_global_graph, None)
        
        assert 'graph_data' in result
        assert 'node_count' in result
        assert 'edge_count' in result
        
        graph_data = result['graph_data']
        assert 'metadata' in graph_data
        assert 'nodes' in graph_data
        assert 'edges' in graph_data
        assert 'vocabulary' in graph_data
        
        # 检查节点格式
        if graph_data['nodes']:
            node = graph_data['nodes'][0]
            assert 'id' in node
            assert 'phrase' in node
            assert 'type' in node
    
    def test_export_with_file_output(self, easygraph_interface, sample_global_graph):
        """测试文件输出功能"""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
            result = easygraph_interface._export_to_json(sample_global_graph, tmp_file.name)
            
            assert 'file_path' in result
            assert result['file_path'] == tmp_file.name
            
            # 验证文件内容
            with open(tmp_file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                assert 'nodes' in data
                assert 'edges' in data
    
    def test_create_multi_view_graph(self, easygraph_interface, sample_global_graph):
        """测试创建多视图图"""
        # 模拟额外视图
        additional_views = {
            'embedding': {
                'graph': Mock(),
                'type': 'embedding'
            },
            'influence': {
                'graph': Mock(),
                'type': 'influence'
            }
        }
        
        # 模拟视图兼容性验证
        with patch.object(easygraph_interface, '_validate_view_compatibility', return_value=True):
            multi_view = easygraph_interface.create_multi_view_graph(
                sample_global_graph, 
                additional_views
            )
        
        assert isinstance(multi_view, MultiViewGraph)
        assert len(multi_view.node_space) == len(sample_global_graph.vocabulary)
        assert len(multi_view.views) >= 2  # 至少包含基础视图和额外视图
        assert multi_view.validate_node_space_consistency()
    
    def test_multi_view_graph_operations(self, sample_vocabulary):
        """测试多视图图操作"""
        reverse_vocab = {v: k for k, v in sample_vocabulary.items()}
        
        multi_view = MultiViewGraph(
            node_space=sample_vocabulary,
            reverse_node_space=reverse_vocab
        )
        
        # 测试添加视图
        mock_graph = Mock()
        multi_view.add_view('test_view', mock_graph, 'test')
        
        assert 'test_view' in multi_view.views
        assert multi_view.get_view('test_view') == mock_graph
        assert 'test_view' in multi_view.list_views()
        
        # 测试节点空间一致性
        assert multi_view.validate_node_space_consistency()
    
    @patch('semantic_coword_pipeline.processors.easygraph_interface.eg')
    def test_graph_fusion_union(self, mock_eg, easygraph_interface, sample_vocabulary):
        """测试联合融合策略"""
        # 设置mock
        mock_graph_class = Mock()
        mock_eg.Graph = mock_graph_class
        
        # 创建模拟图实例
        mock_fused_graph = Mock()
        mock_graph_class.return_value = mock_fused_graph
        
        # 创建多视图图
        reverse_vocab = {v: k for k, v in sample_vocabulary.items()}
        multi_view = MultiViewGraph(
            node_space=sample_vocabulary,
            reverse_node_space=reverse_vocab
        )
        
        # 添加模拟视图
        mock_view1 = Mock()
        mock_view1.edges = [(0, 1), (1, 2)]
        mock_view2 = Mock()
        mock_view2.edges = [(1, 2), (2, 3)]
        
        multi_view.add_view('view1', mock_view1)
        multi_view.add_view('view2', mock_view2)
        
        # 执行融合
        result = easygraph_interface.fuse_graphs(
            multi_view, 
            FusionStrategy.UNION,
            target_views=['view1', 'view2']
        )
        
        assert isinstance(result, FusionResult)
        assert result.fusion_strategy == FusionStrategy.UNION
        assert len(result.source_views) == 2
        assert 'view1' in result.source_views
        assert 'view2' in result.source_views
    
    @patch('semantic_coword_pipeline.processors.easygraph_interface.eg')
    def test_graph_fusion_weighted(self, mock_eg, easygraph_interface, sample_vocabulary):
        """测试加权融合策略"""
        # 设置mock
        mock_graph_class = Mock()
        mock_eg.Graph = mock_graph_class
        
        mock_fused_graph = Mock()
        mock_graph_class.return_value = mock_fused_graph
        
        # 创建多视图图
        reverse_vocab = {v: k for k, v in sample_vocabulary.items()}
        multi_view = MultiViewGraph(
            node_space=sample_vocabulary,
            reverse_node_space=reverse_vocab
        )
        
        # 添加模拟视图
        mock_view1 = Mock()
        mock_view1.edges = [(0, 1), (1, 2)]
        mock_edges1 = Mock()
        mock_edges1.__iter__ = Mock(return_value=iter([(0, 1), (1, 2)]))
        mock_edges1.get = Mock(return_value={'weight': 1.0})
        mock_view1.edges = mock_edges1
        
        mock_view2 = Mock()
        mock_view2.edges = [(1, 2), (2, 3)]
        mock_edges2 = Mock()
        mock_edges2.__iter__ = Mock(return_value=iter([(1, 2), (2, 3)]))
        mock_edges2.get = Mock(return_value={'weight': 2.0})
        mock_view2.edges = mock_edges2
        
        multi_view.add_view('view1', mock_view1)
        multi_view.add_view('view2', mock_view2)
        
        # 定义权重
        view_weights = {'view1': 0.3, 'view2': 0.7}
        
        # 执行加权融合
        result = easygraph_interface.fuse_graphs(
            multi_view,
            FusionStrategy.WEIGHTED,
            view_weights=view_weights,
            target_views=['view1', 'view2']
        )
        
        assert isinstance(result, FusionResult)
        assert result.fusion_strategy == FusionStrategy.WEIGHTED
        assert result.fusion_weights == view_weights
    
    def test_fusion_experiment_design(self, easygraph_interface, sample_vocabulary):
        """测试融合实验设计"""
        reverse_vocab = {v: k for k, v in sample_vocabulary.items()}
        multi_view = MultiViewGraph(
            node_space=sample_vocabulary,
            reverse_node_space=reverse_vocab
        )
        
        # 添加多个视图
        multi_view.add_view('cooccurrence', Mock())
        multi_view.add_view('embedding', Mock())
        multi_view.add_view('influence', Mock())
        
        experiment_design = easygraph_interface.create_fusion_experiment_design(multi_view)
        
        assert 'metadata' in experiment_design
        assert 'fusion_strategies' in experiment_design
        assert 'evaluation_metrics' in experiment_design
        assert 'baseline_comparisons' in experiment_design
        assert 'parameter_grids' in experiment_design
        
        # 检查融合策略
        strategies = [s['strategy'] for s in experiment_design['fusion_strategies']]
        assert 'union' in strategies
        assert 'intersection' in strategies
        assert 'consensus' in strategies
        assert 'weighted' in strategies
        
        # 检查评估指标
        metrics = [m['name'] for m in experiment_design['evaluation_metrics']]
        assert 'modularity' in metrics
        assert 'clustering_coefficient' in metrics
        assert 'node_coverage' in metrics
    
    @patch('semantic_coword_pipeline.processors.easygraph_interface.eg')
    def test_validate_easygraph_compatibility(self, mock_eg, easygraph_interface):
        """测试EasyGraph兼容性验证"""
        # 模拟EasyGraph可用
        mock_graph = Mock()
        mock_graph.nodes = [0, 1, 2]
        mock_graph.edges = [(0, 1), (1, 2)]
        mock_nodes = Mock()
        mock_nodes.__len__ = Mock(return_value=3)
        mock_nodes.get = Mock(return_value={'phrase': 'test'})
        mock_graph.nodes = mock_nodes
        mock_edges = Mock()
        mock_edges.__len__ = Mock(return_value=2)
        mock_edges.get = Mock(return_value={'weight': 1.0})
        mock_graph.edges = mock_edges
        
        mock_eg.Graph = Mock
        mock_eg.is_connected = Mock(return_value=True)
        
        # 正确设置mock对象的类信息
        mock_graph.__class__ = Mock()
        mock_graph.__class__.__name__ = 'Graph'
        mock_graph.__class__.__module__ = 'easygraph.classes.graph'
        
        result = easygraph_interface.validate_easygraph_compatibility(mock_graph)
        
        assert result['easygraph_available'] is True
        assert result['is_easygraph_instance'] is True
        assert result['has_nodes'] is True
        assert result['has_edges'] is True
    
    def test_validate_easygraph_compatibility_no_easygraph(self, easygraph_interface):
        """测试EasyGraph不可用时的兼容性验证"""
        with patch('semantic_coword_pipeline.processors.easygraph_interface.eg', None):
            result = easygraph_interface.validate_easygraph_compatibility(Mock())
        
        assert result['easygraph_available'] is False
    
    def test_create_easygraph_from_matrix(self, sample_vocabulary):
        """测试从矩阵创建EasyGraph实例"""
        matrix = scipy.sparse.csr_matrix([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        
        vocab = {'phrase_1': 0, 'phrase_2': 1, 'phrase_3': 2}
        
        # 测试EasyGraph不可用的情况
        with patch('semantic_coword_pipeline.processors.easygraph_interface.eg', None):
            result = create_easygraph_from_matrix(matrix, vocab)
            assert result is None
    
    def test_validate_multi_view_consistency(self, sample_vocabulary):
        """测试多视图一致性验证"""
        reverse_vocab = {v: k for k, v in sample_vocabulary.items()}
        
        multi_view = MultiViewGraph(
            node_space=sample_vocabulary,
            reverse_node_space=reverse_vocab
        )
        
        # 添加兼容的视图
        mock_view = Mock()
        mock_view.nodes = [0, 1, 2]  # 节点ID在词表范围内
        multi_view.views['test_view'] = {'graph': mock_view}
        
        results = validate_multi_view_consistency(multi_view)
        
        assert 'node_space_consistent' in results
        assert 'all_views_compatible' in results
        assert 'has_multiple_views' in results
        
        assert results['node_space_consistent'] is True
        assert results['has_multiple_views'] is False  # 只有一个视图


class TestGraphFormats:
    """图格式测试类"""
    
    def test_graph_format_enum(self):
        """测试图格式枚举"""
        assert GraphFormat.EASYGRAPH.value == "easygraph"
        assert GraphFormat.ADJACENCY_MATRIX.value == "adjacency_matrix"
        assert GraphFormat.EDGE_LIST.value == "edge_list"
        assert GraphFormat.JSON.value == "json"
    
    def test_fusion_strategy_enum(self):
        """测试融合策略枚举"""
        assert FusionStrategy.UNION.value == "union"
        assert FusionStrategy.INTERSECTION.value == "intersection"
        assert FusionStrategy.CONSENSUS.value == "consensus"
        assert FusionStrategy.WEIGHTED.value == "weighted"


class TestErrorHandling:
    """错误处理测试类"""
    
    def test_export_invalid_format(self):
        """测试无效格式导出"""
        interface = EasyGraphInterface({})
        
        # 创建空的全局图
        global_graph = GlobalGraph(
            vocabulary={},
            reverse_vocabulary={}
        )
        
        # 测试不支持的格式
        result = interface.export_global_graph(global_graph, GraphFormat.NETWORKX)
        assert result['success'] is False
        assert 'Unsupported export format' in result['error']
    
    def test_fusion_insufficient_views(self):
        """测试视图数量不足的融合"""
        interface = EasyGraphInterface({})
        
        multi_view = MultiViewGraph(
            node_space={'phrase': 0},
            reverse_node_space={0: 'phrase'}
        )
        
        # 只添加一个视图
        multi_view.add_view('single_view', Mock())
        
        with pytest.raises(ValueError, match="At least 2 views are required"):
            interface.fuse_graphs(multi_view, FusionStrategy.UNION)
    
    def test_export_missing_data(self):
        """测试缺失数据的导出"""
        interface = EasyGraphInterface({})
        
        # 创建没有共现矩阵的全局图
        global_graph = GlobalGraph(
            vocabulary={'phrase': 0},
            reverse_vocabulary={0: 'phrase'},
            cooccurrence_matrix=None
        )
        
        with pytest.raises(ValueError, match="No cooccurrence matrix available"):
            interface._export_to_adjacency_matrix(global_graph, None)


if __name__ == '__main__':
    pytest.main([__file__])