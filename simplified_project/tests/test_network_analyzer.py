"""
网络分析器测试

测试NetworkAnalyzer的各种功能，包括基础统计、高级指标、社群分析和对比分析。
"""

import pytest
import numpy as np
import sys
import os
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Easy-Graph'))

from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import Mock, patch

try:
    import easygraph as eg
except ImportError:
    eg = None

from semantic_coword_pipeline.analyzers.network_analyzer import NetworkAnalyzer
from semantic_coword_pipeline.core.data_models import GlobalGraph, StateSubgraph


class TestNetworkAnalyzer:
    """NetworkAnalyzer单元测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.config = {
            'enable_community_detection': True,
            'centrality_metrics': ['degree', 'betweenness', 'closeness', 'pagerank'],
            'community_algorithm': 'louvain',
            'comparison_metrics': [
                'node_count', 'edge_count', 'density', 'isolated_nodes_ratio',
                'connected_components', 'clustering_coefficient', 'average_path_length'
            ]
        }
        self.analyzer = NetworkAnalyzer(self.config)
    
    def create_mock_graph(self, node_count: int = 5, edge_count: int = 4) -> GlobalGraph:
        """创建模拟图用于测试"""
        # 创建词表
        vocabulary = {f"phrase_{i}": i for i in range(node_count)}
        reverse_vocabulary = {i: f"phrase_{i}" for i in range(node_count)}
        
        # 创建模拟的EasyGraph实例
        mock_graph = Mock()
        mock_graph.number_of_nodes.return_value = node_count
        mock_graph.number_of_edges.return_value = edge_count
        mock_graph.nodes.return_value = list(range(node_count))
        
        # 模拟度分布
        degrees = [2, 2, 1, 1, 0] if node_count == 5 else [1] * node_count
        mock_graph.degree.side_effect = lambda node: degrees[node] if node < len(degrees) else 0
        
        return GlobalGraph(
            vocabulary=vocabulary,
            reverse_vocabulary=reverse_vocabulary,
            easygraph_instance=mock_graph
        )
    
    def test_initialization(self):
        """测试NetworkAnalyzer初始化"""
        assert self.analyzer.config == self.config
        assert self.analyzer.enable_community_detection is True
        assert 'degree' in self.analyzer.centrality_metrics
        assert self.analyzer.community_algorithm == 'louvain'
    
    def test_calculate_basic_statistics_empty_graph(self):
        """测试空图的基础统计计算"""
        empty_graph = self.create_mock_graph(0, 0)
        
        stats = self.analyzer.calculate_basic_statistics(empty_graph)
        
        assert stats['node_count'] == 0.0
        assert stats['edge_count'] == 0.0
        assert stats['density'] == 0.0
        assert stats['isolated_nodes_ratio'] == 0.0
        assert stats['average_degree'] == 0.0
    
    def test_calculate_basic_statistics_normal_graph(self):
        """测试正常图的基础统计计算"""
        graph = self.create_mock_graph(5, 4)
        
        stats = self.analyzer.calculate_basic_statistics(graph)
        
        assert stats['node_count'] == 5.0
        assert stats['edge_count'] == 4.0
        assert stats['density'] == 4.0 / 10.0  # 4 edges / (5*4/2) max edges
        assert stats['isolated_nodes_count'] == 1.0  # 一个度为0的节点
        assert stats['isolated_nodes_ratio'] == 0.2  # 1/5
        assert stats['average_degree'] == 1.2  # (2+2+1+1+0)/5
    
    def test_calculate_basic_statistics_no_easygraph(self):
        """测试没有EasyGraph实例时的错误处理"""
        graph = GlobalGraph(
            vocabulary={"phrase_1": 0},
            reverse_vocabulary={0: "phrase_1"},
            easygraph_instance=None
        )
        
        with pytest.raises(ValueError, match="EasyGraph instance is required"):
            self.analyzer.calculate_basic_statistics(graph)
    
    @patch('semantic_coword_pipeline.analyzers.network_analyzer.eg')
    def test_calculate_advanced_metrics_connected_graph(self, mock_eg):
        """测试连通图的高级指标计算"""
        graph = self.create_mock_graph(5, 6)
        
        # 模拟EasyGraph函数
        mock_eg.is_connected.return_value = True
        mock_eg.clustering.return_value = {0: 0.5, 1: 0.3, 2: 0.7, 3: 0.2, 4: 0.0}
        mock_eg.transitivity.return_value = 0.4
        mock_eg.average_shortest_path_length.return_value = 2.5
        mock_eg.diameter.return_value = 4
        mock_eg.degree_centrality.return_value = {0: 0.5, 1: 0.3, 2: 0.7, 3: 0.2, 4: 0.0}
        mock_eg.betweenness_centrality.return_value = {0: 0.4, 1: 0.2, 2: 0.6, 3: 0.1, 4: 0.0}
        mock_eg.closeness_centrality.return_value = {0: 0.6, 1: 0.4, 2: 0.8, 3: 0.3, 4: 0.1}
        mock_eg.pagerank.return_value = {0: 0.25, 1: 0.15, 2: 0.35, 3: 0.15, 4: 0.1}
        
        metrics = self.analyzer.calculate_advanced_metrics(graph)
        
        assert metrics['connected_components_count'] == 1
        assert metrics['largest_component_size'] == 5.0
        assert metrics['largest_component_ratio'] == 1.0
        assert abs(metrics['average_clustering_coefficient'] - 0.34) < 1e-10  # (0.5+0.3+0.7+0.2+0.0)/5
        assert metrics['global_clustering_coefficient'] == 0.4
        assert metrics['average_path_length'] == 2.5
        assert metrics['diameter'] == 4.0
        
        # 检查中心性指标
        centrality_metrics = metrics['centrality_metrics']
        assert 'degree_centrality' in centrality_metrics
        assert 'betweenness_centrality' in centrality_metrics
        assert 'closeness_centrality' in centrality_metrics
        assert 'pagerank_centrality' in centrality_metrics
    
    @patch('semantic_coword_pipeline.analyzers.network_analyzer.eg')
    def test_calculate_advanced_metrics_disconnected_graph(self, mock_eg):
        """测试非连通图的高级指标计算"""
        graph = self.create_mock_graph(5, 2)
        
        # 模拟非连通图
        mock_eg.is_connected.return_value = False
        mock_eg.connected_components.return_value = [{0, 1, 2}, {3}, {4}]
        mock_eg.clustering.return_value = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        mock_eg.transitivity.return_value = 0.0
        mock_eg.degree_centrality.return_value = {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.0, 4: 0.0}
        mock_eg.betweenness_centrality.return_value = {0: 0.2, 1: 0.4, 2: 0.2, 3: 0.0, 4: 0.0}
        mock_eg.pagerank.return_value = {0: 0.2, 1: 0.3, 2: 0.2, 3: 0.15, 4: 0.15}
        
        metrics = self.analyzer.calculate_advanced_metrics(graph)
        
        assert metrics['connected_components_count'] == 3
        assert metrics['largest_component_size'] == 3.0
        assert metrics['largest_component_ratio'] == 0.6  # 3/5
        assert metrics['average_path_length'] == float('inf')  # 非连通图
        assert metrics['diameter'] == float('inf')
        
        # 非连通图不计算接近中心性
        centrality_metrics = metrics['centrality_metrics']
        assert centrality_metrics['closeness_centrality'] == {}
    
    @patch('semantic_coword_pipeline.analyzers.network_analyzer.eg')
    def test_detect_communities(self, mock_eg):
        """测试社群检测功能"""
        graph = self.create_mock_graph(6, 8)
        
        # 模拟社群检测结果
        mock_communities = [{0, 1, 2}, {3, 4, 5}]
        mock_eg.louvain_communities.return_value = mock_communities
        mock_eg.modularity.return_value = 0.45
        
        result = self.analyzer.detect_communities(graph)
        
        assert result['community_count'] == 2
        assert result['modularity'] == 0.45
        assert len(result['communities']) == 2
        assert result['community_sizes'] == [3, 3]
        assert result['average_community_size'] == 3.0
        assert result['largest_community_size'] == 3
        assert result['smallest_community_size'] == 3
    
    def test_detect_communities_disabled(self):
        """测试禁用社群检测时的行为"""
        config = self.config.copy()
        config['enable_community_detection'] = False
        analyzer = NetworkAnalyzer(config)
        
        graph = self.create_mock_graph(5, 4)
        result = analyzer.detect_communities(graph)
        
        assert result['communities'] == []
        assert result['modularity'] == 0.0
        assert result['community_count'] == 0
    
    def test_detect_communities_small_graph(self):
        """测试小图（节点数<2）的社群检测"""
        graph = self.create_mock_graph(1, 0)
        
        result = self.analyzer.detect_communities(graph)
        
        assert result['communities'] == []
        assert result['modularity'] == 0.0
        assert result['community_count'] == 0
    
    def test_compare_network_structures(self):
        """测试网络结构对比分析"""
        graph1 = self.create_mock_graph(5, 4)
        graph2 = self.create_mock_graph(8, 10)
        
        graphs = {'graph1': graph1, 'graph2': graph2}
        
        with patch.object(self.analyzer, 'calculate_basic_statistics') as mock_basic, \
             patch.object(self.analyzer, 'calculate_advanced_metrics') as mock_advanced, \
             patch.object(self.analyzer, 'detect_communities') as mock_communities:
            
            mock_basic.side_effect = [
                {'node_count': 5.0, 'edge_count': 4.0, 'density': 0.4, 'isolated_nodes_ratio': 0.2},
                {'node_count': 8.0, 'edge_count': 10.0, 'density': 0.36, 'isolated_nodes_ratio': 0.0}
            ]
            mock_advanced.return_value = {'connected_components_count': 1}
            mock_communities.return_value = {'modularity': 0.3, 'community_count': 2}
            
            result = self.analyzer.compare_network_structures(graphs, "test_comparison")
        
        assert result['comparison_name'] == "test_comparison"
        assert result['graph_names'] == ['graph1', 'graph2']
        assert 'basic_statistics' in result
        assert 'advanced_metrics' in result
        assert 'community_analysis' in result
        assert 'summary' in result
        
        # 检查摘要
        summary = result['summary']
        assert 'node_count_comparison' in summary
        assert summary['node_count_comparison']['max_graph'] == 'graph2'
        assert summary['node_count_comparison']['min_graph'] == 'graph1'
        assert summary['node_count_comparison']['ratio_max_to_min'] == 1.6  # 8/5
    
    def test_compare_network_structures_empty_input(self):
        """测试空输入的网络结构对比"""
        with pytest.raises(ValueError, match="At least one graph is required"):
            self.analyzer.compare_network_structures({})
    
    def test_analyze_cross_state_differences(self):
        """测试跨州差异分析"""
        # 创建模拟的州级子图
        global_graph = self.create_mock_graph(10, 15)
        
        subgraph1 = StateSubgraph(
            state_name="state1",
            parent_global_graph=global_graph,
            activation_mask=np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
        )
        subgraph1.easygraph_instance = Mock()
        subgraph1.easygraph_instance.number_of_nodes.return_value = 3
        
        subgraph2 = StateSubgraph(
            state_name="state2", 
            parent_global_graph=global_graph,
            activation_mask=np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        )
        subgraph2.easygraph_instance = Mock()
        subgraph2.easygraph_instance.number_of_nodes.return_value = 4
        
        state_subgraphs = {'state1': subgraph1, 'state2': subgraph2}
        
        with patch.object(self.analyzer, 'calculate_basic_statistics') as mock_basic, \
             patch('semantic_coword_pipeline.analyzers.network_analyzer.eg') as mock_eg:
            
            mock_basic.side_effect = [
                {'node_count': 3.0, 'edge_count': 2.0},
                {'node_count': 4.0, 'edge_count': 5.0}
            ]
            mock_eg.degree_centrality.side_effect = [
                {0: 0.5, 1: 0.3, 2: 0.2},
                {2: 0.4, 3: 0.3, 4: 0.2, 5: 0.1}
            ]
            
            result = self.analyzer.analyze_cross_state_differences(state_subgraphs)
        
        assert result['states_analyzed'] == ['state1', 'state2']
        assert 'state_statistics' in result
        assert 'unique_phrases_by_state' in result
        assert 'centrality_ranking_stability' in result
        
        # 检查州特有短语分析
        assert result['unique_phrases_by_state']['state1'] == 3
        assert result['unique_phrases_by_state']['state2'] == 4
    
    def test_analyze_cross_state_differences_insufficient_states(self):
        """测试州数不足时的跨州差异分析"""
        subgraph = StateSubgraph(
            state_name="state1",
            parent_global_graph=self.create_mock_graph(5, 4)
        )
        
        result = self.analyzer.analyze_cross_state_differences({'state1': subgraph})
        
        assert 'error' in result
        assert 'At least 2 state subgraphs required' in result['error']
    
    def test_generate_analysis_report(self):
        """测试分析报告生成"""
        analysis_results = {
            'basic_statistics': {
                'graph1': {
                    'node_count': 5.0,
                    'edge_count': 4.0,
                    'density': 0.4,
                    'isolated_nodes_ratio': 0.2,
                    'average_degree': 1.6
                }
            },
            'advanced_metrics': {
                'graph1': {
                    'connected_components_count': 1,
                    'largest_component_ratio': 1.0,
                    'average_clustering_coefficient': 0.3,
                    'global_clustering_coefficient': 0.25,
                    'average_path_length': 2.5
                }
            },
            'community_analysis': {
                'graph1': {
                    'community_count': 2,
                    'modularity': 0.45,
                    'average_community_size': 2.5
                }
            },
            'summary': {
                'node_count_comparison': {
                    'max_graph': 'graph1',
                    'min_graph': 'graph1',
                    'values': {'graph1': 5.0},
                    'ratio_max_to_min': 1.0
                }
            }
        }
        
        report = self.analyzer.generate_analysis_report(analysis_results)
        
        assert "# 网络分析报告" in report
        assert "## 基础网络统计" in report
        assert "## 高级网络指标" in report
        assert "## 社群分析" in report
        assert "## 对比分析摘要" in report
        assert "节点数: 5" in report
        assert "边数: 4" in report
        assert "密度: 0.4000" in report


# 属性测试
class TestNetworkAnalyzerProperties:
    """NetworkAnalyzer属性测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.config = {
            'enable_community_detection': True,
            'centrality_metrics': ['degree'],
            'community_algorithm': 'louvain'
        }
        self.analyzer = NetworkAnalyzer(self.config)
    
    @given(
        node_count=st.integers(min_value=0, max_value=20),
        edge_count=st.integers(min_value=0, max_value=50)
    )
    @settings(max_examples=100, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
    def test_property_21_network_statistics_correctness(self, node_count: int, edge_count: int):
        """
        属性 21: 网络统计指标正确性
        对于任何网络图，计算的基础统计指标（节点数、边数、密度、孤立点比例）应该与图的实际结构一致
        **验证：需求 8.4**
        **Feature: semantic-coword-enhancement, Property 21: 网络统计指标正确性**
        """
        # 创建模拟图
        vocabulary = {f"phrase_{i}": i for i in range(node_count)}
        reverse_vocabulary = {i: f"phrase_{i}" for i in range(node_count)}
        
        mock_graph = Mock()
        mock_graph.number_of_nodes.return_value = node_count
        mock_graph.number_of_edges.return_value = min(edge_count, node_count * (node_count - 1) // 2)
        mock_graph.nodes.return_value = list(range(node_count))
        
        # 模拟度分布（简化：假设度均匀分布）
        if node_count > 0:
            avg_degree = min(2 * edge_count / node_count, node_count - 1) if node_count > 0 else 0
            degrees = [int(avg_degree)] * node_count
            # 确保有一些孤立节点用于测试
            if node_count > 2:
                degrees[-1] = 0  # 最后一个节点设为孤立节点
        else:
            degrees = []
        
        mock_graph.degree.side_effect = lambda node: degrees[node] if node < len(degrees) else 0
        
        graph = GlobalGraph(
            vocabulary=vocabulary,
            reverse_vocabulary=reverse_vocabulary,
            easygraph_instance=mock_graph
        )
        
        # 计算统计指标
        stats = self.analyzer.calculate_basic_statistics(graph)
        
        # 验证基础指标正确性
        assert stats['node_count'] == float(node_count)
        assert stats['edge_count'] == float(min(edge_count, node_count * (node_count - 1) // 2))
        
        # 验证密度计算
        if node_count > 1:
            max_edges = node_count * (node_count - 1) / 2
            expected_density = stats['edge_count'] / max_edges
            assert abs(stats['density'] - expected_density) < 1e-10
        else:
            assert stats['density'] == 0.0
        
        # 验证孤立节点比例
        if node_count > 0:
            isolated_count = sum(1 for d in degrees if d == 0)
            expected_ratio = isolated_count / node_count
            assert abs(stats['isolated_nodes_ratio'] - expected_ratio) < 1e-10
        else:
            assert stats['isolated_nodes_ratio'] == 0.0
    
    @given(
        graph_count=st.integers(min_value=2, max_value=5),
        node_counts=st.lists(st.integers(min_value=1, max_value=10), min_size=2, max_size=5)
    )
    @settings(max_examples=50, deadline=15000, suppress_health_check=[HealthCheck.too_slow])
    def test_property_23_multidimensional_comparison_completeness(self, graph_count: int, node_counts: List[int]):
        """
        属性 23: 多维度对比分析完整性
        对于任何对比分析请求，系统应该生成"单词vs词组节点"、"静态vs动态停词"、"总图vs子图"三个维度的完整对比结果
        **验证：需求 8.1, 8.2, 8.3**
        **Feature: semantic-coword-enhancement, Property 23: 多维度对比分析完整性**
        """
        # 限制输入大小以避免测试超时
        if len(node_counts) > graph_count:
            node_counts = node_counts[:graph_count]
        elif len(node_counts) < graph_count:
            node_counts.extend([5] * (graph_count - len(node_counts)))
        
        # 创建多个模拟图进行对比
        graphs = {}
        
        for i, node_count in enumerate(node_counts):
            vocabulary = {f"phrase_{j}": j for j in range(node_count)}
            reverse_vocabulary = {j: f"phrase_{j}" for j in range(node_count)}
            
            mock_graph = Mock()
            mock_graph.number_of_nodes.return_value = node_count
            mock_graph.number_of_edges.return_value = max(0, node_count - 1)
            mock_graph.nodes.return_value = list(range(node_count))
            mock_graph.degree.side_effect = lambda node: 1 if node < node_count - 1 else 0
            
            graph = GlobalGraph(
                vocabulary=vocabulary,
                reverse_vocabulary=reverse_vocabulary,
                easygraph_instance=mock_graph
            )
            
            graphs[f"graph_{i}"] = graph
        
        # 执行对比分析
        result = self.analyzer.compare_network_structures(graphs, "test_comparison")
        
        # 验证对比分析的完整性
        assert 'comparison_name' in result
        assert 'graph_names' in result
        assert 'basic_statistics' in result
        assert 'advanced_metrics' in result
        assert 'community_analysis' in result
        assert 'summary' in result
        
        # 验证所有图都被分析
        assert len(result['graph_names']) == len(graphs)
        assert len(result['basic_statistics']) == len(graphs)
        assert len(result['advanced_metrics']) == len(graphs)
        assert len(result['community_analysis']) == len(graphs)
        
        # 验证摘要包含关键对比指标
        summary = result['summary']
        expected_comparisons = [
            'node_count_comparison',
            'edge_count_comparison', 
            'density_comparison',
            'isolated_nodes_ratio_comparison',
            'modularity_comparison'
        ]
        
        for comparison in expected_comparisons:
            assert comparison in summary
            assert 'values' in summary[comparison]
            assert 'max_graph' in summary[comparison]
            assert 'min_graph' in summary[comparison]