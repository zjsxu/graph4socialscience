"""
确定性布局引擎测试

测试DeterministicLayoutEngine的各项功能，包括：
- 布局确定性（属性1）
- 节点位置缓存一致性（属性3）
- 可视化过滤一致性（属性2）
"""

import pytest
import tempfile
import shutil
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Tuple, List, Any

from semantic_coword_pipeline.processors.deterministic_layout_engine import (
    DeterministicLayoutEngine,
    LayoutParameters,
    VisualizationFilter,
    LayoutResult,
    PositionCache,
    ForceDirectedLayout,
    HierarchicalLayout
)
from semantic_coword_pipeline.core.config import Config
from semantic_coword_pipeline.core.data_models import StateSubgraph, GlobalGraph


class MockGraph:
    """模拟EasyGraph图对象"""
    
    def __init__(self, nodes: List[int], edges: List[Tuple[int, int]]):
        self._nodes = nodes
        self._edges = edges
        self._node_degrees = {}
        
        # 计算度
        for node in nodes:
            self._node_degrees[node] = 0
        
        for edge in edges:
            if len(edge) >= 2:
                node1, node2 = edge[0], edge[1]
                if node1 in self._node_degrees:
                    self._node_degrees[node1] += 1
                if node2 in self._node_degrees:
                    self._node_degrees[node2] += 1
    
    def nodes(self):
        return self._nodes
    
    def edges(self):
        return self._edges
    
    def degree(self, node_id: int) -> int:
        return self._node_degrees.get(node_id, 0)
    
    def neighbors(self, node_id: int) -> List[int]:
        neighbors = []
        for edge in self._edges:
            if len(edge) >= 2:
                if edge[0] == node_id:
                    neighbors.append(edge[1])
                elif edge[1] == node_id:
                    neighbors.append(edge[0])
        return neighbors
    
    def number_of_nodes(self) -> int:
        return len(self._nodes)


class TestPositionCache:
    """测试位置缓存管理器"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = PositionCache(self.temp_dir)
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_and_load_positions(self):
        """测试保存和加载位置"""
        positions = {1: (0.1, 0.2), 2: (0.3, 0.4), 3: (0.5, 0.6)}
        graph_id = "test_graph"
        algorithm = "force_directed"
        seed = 42
        
        # 保存位置
        self.cache.save_positions(graph_id, algorithm, seed, positions)
        
        # 加载位置
        loaded_positions = self.cache.load_positions(graph_id, algorithm, seed)
        
        assert loaded_positions == positions
    
    def test_load_nonexistent_positions(self):
        """测试加载不存在的位置"""
        result = self.cache.load_positions("nonexistent", "algorithm", 42)
        assert result is None
    
    def test_cache_key_uniqueness(self):
        """测试缓存键的唯一性"""
        positions1 = {1: (0.1, 0.2)}
        positions2 = {1: (0.3, 0.4)}
        
        # 不同的参数应该产生不同的缓存
        self.cache.save_positions("graph1", "algo1", 42, positions1)
        self.cache.save_positions("graph1", "algo2", 42, positions2)
        
        loaded1 = self.cache.load_positions("graph1", "algo1", 42)
        loaded2 = self.cache.load_positions("graph1", "algo2", 42)
        
        assert loaded1 == positions1
        assert loaded2 == positions2
        assert loaded1 != loaded2
    
    def test_clear_cache(self):
        """测试清理缓存"""
        positions = {1: (0.1, 0.2)}
        
        self.cache.save_positions("graph1", "algo", 42, positions)
        self.cache.save_positions("graph2", "algo", 42, positions)
        
        # 清理特定图的缓存
        self.cache.clear_cache("graph1")
        
        assert self.cache.load_positions("graph1", "algo", 42) is None
        assert self.cache.load_positions("graph2", "algo", 42) == positions
        
        # 清理所有缓存
        self.cache.clear_cache()
        assert self.cache.load_positions("graph2", "algo", 42) is None


class TestForceDirectedLayout:
    """测试力导向布局算法"""
    
    def setup_method(self):
        """设置测试环境"""
        self.params = LayoutParameters(
            random_seed=42,
            max_iterations=10,  # 减少迭代次数以加快测试
            convergence_threshold=1e-3
        )
        self.layout = ForceDirectedLayout(self.params)
    
    def test_empty_graph_layout(self):
        """测试空图布局"""
        graph = MockGraph([], [])
        result = self.layout.compute(graph, [])
        
        assert result.positions == {}
        assert result.converged is True
        assert result.iterations_completed == 0
    
    def test_single_node_layout(self):
        """测试单节点布局"""
        graph = MockGraph([1], [])
        result = self.layout.compute(graph, [1])
        
        assert len(result.positions) == 1
        assert 1 in result.positions
        assert result.converged is True
    
    def test_two_node_layout(self):
        """测试两节点布局"""
        graph = MockGraph([1, 2], [(1, 2)])
        result = self.layout.compute(graph, [1, 2])
        
        assert len(result.positions) == 2
        assert 1 in result.positions
        assert 2 in result.positions
        
        # 检查位置不同
        pos1 = result.positions[1]
        pos2 = result.positions[2]
        assert pos1 != pos2
    
    def test_deterministic_layout(self):
        """测试布局的确定性"""
        graph = MockGraph([1, 2, 3], [(1, 2), (2, 3), (1, 3)])
        
        # 多次计算应该得到相同结果
        result1 = self.layout.compute(graph, [1, 2, 3])
        result2 = self.layout.compute(graph, [1, 2, 3])
        
        assert result1.positions == result2.positions


class TestHierarchicalLayout:
    """测试层级布局算法"""
    
    def setup_method(self):
        """设置测试环境"""
        self.params = LayoutParameters(random_seed=42)
        self.layout = HierarchicalLayout(self.params)
    
    def test_empty_graph_layout(self):
        """测试空图布局"""
        graph = MockGraph([], [])
        result = self.layout.compute(graph, [])
        
        assert result.positions == {}
        assert result.converged is True
    
    def test_single_node_layout(self):
        """测试单节点布局"""
        graph = MockGraph([1], [])
        result = self.layout.compute(graph, [1])
        
        assert len(result.positions) == 1
        assert 1 in result.positions
        assert result.positions[1] == (0.0, 0.0)
    
    def test_tree_layout(self):
        """测试树形结构布局"""
        # 创建简单的树：1-2, 1-3
        graph = MockGraph([1, 2, 3], [(1, 2), (1, 3)])
        result = self.layout.compute(graph, [1, 2, 3])
        
        assert len(result.positions) == 3
        
        # 根节点应该在最高层
        pos1 = result.positions[1]
        pos2 = result.positions[2]
        pos3 = result.positions[3]
        
        # 检查y坐标（层级）
        assert pos2[1] == pos3[1]  # 叶子节点在同一层
        assert pos1[1] > pos2[1]   # 根节点在更高层


class TestDeterministicLayoutEngine:
    """测试确定性布局引擎"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试配置
        config_data = {
            'layout_engine': {
                'algorithm': 'force_directed',
                'random_seed': 42,
                'cache_enabled': True,
                'max_iterations': 10
            },
            'output': {
                'base_path': self.temp_dir
            }
        }
        
        self.config = Config()
        for key, value in config_data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    self.config.set(f"{key}.{subkey}", subvalue)
            else:
                self.config.set(key, value)
        
        self.engine = DeterministicLayoutEngine(self.config)
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """测试初始化"""
        assert self.engine.params.algorithm == 'force_directed'
        assert self.engine.params.random_seed == 42
        assert self.engine.params.max_iterations == 10
    
    def test_compute_layout_empty_graph(self):
        """测试空图布局计算"""
        graph = MockGraph([], [])
        result = self.engine.compute_layout(graph, "empty_graph")
        
        assert result.positions == {}
        assert result.converged is True
    
    def test_compute_layout_with_cache(self):
        """测试带缓存的布局计算"""
        graph = MockGraph([1, 2, 3], [(1, 2), (2, 3)])
        graph_id = "test_graph"
        
        # 第一次计算
        result1 = self.engine.compute_layout(graph, graph_id)
        
        # 第二次计算应该使用缓存
        result2 = self.engine.compute_layout(graph, graph_id)
        
        assert result1.positions == result2.positions
        assert result2.metadata.get('from_cache') is True
    
    def test_force_recompute(self):
        """测试强制重新计算"""
        graph = MockGraph([1, 2], [(1, 2)])
        graph_id = "test_graph"
        
        # 第一次计算
        result1 = self.engine.compute_layout(graph, graph_id)
        
        # 强制重新计算
        result2 = self.engine.compute_layout(graph, graph_id, force_recompute=True)
        
        # 结果应该相同（因为是确定性的），但不是来自缓存
        assert result1.positions == result2.positions
        assert result2.metadata.get('from_cache') is not True
    
    def test_different_algorithms(self):
        """测试不同的布局算法"""
        graph = MockGraph([1, 2, 3], [(1, 2), (2, 3)])
        
        # 测试力导向布局
        self.engine.params.algorithm = 'force_directed'
        result_fd = self.engine.compute_layout(graph, "test_fd")
        
        # 测试层级布局
        self.engine.params.algorithm = 'hierarchical'
        result_hier = self.engine.compute_layout(graph, "test_hier")
        
        assert result_fd.algorithm_used == 'force_directed'
        assert result_hier.algorithm_used == 'hierarchical'
        
        # 不同算法应该产生不同的结果
        assert result_fd.positions != result_hier.positions
    
    def test_visualization_filter(self):
        """测试可视化过滤"""
        # 创建有不同度的图
        graph = MockGraph([1, 2, 3, 4], [(1, 2), (1, 3), (1, 4)])  # 节点1度为3，其他度为1
        positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.0, 1.0), 4: (1.0, 1.0)}
        
        # 过滤掉度小于2的节点
        filter_config = VisualizationFilter(min_degree=2)
        filtered_graph, filtered_positions = self.engine.apply_visualization_filter(
            graph, positions, filter_config
        )
        
        # 只有节点1应该保留
        assert len(filtered_positions) == 1
        assert 1 in filtered_positions
    
    def test_max_nodes_filter(self):
        """测试最大节点数过滤"""
        graph = MockGraph([1, 2, 3, 4, 5], [(1, 2), (2, 3), (3, 4), (4, 5)])
        positions = {i: (float(i), 0.0) for i in range(1, 6)}
        
        # 限制最多3个节点
        filter_config = VisualizationFilter(max_nodes=3)
        filtered_graph, filtered_positions = self.engine.apply_visualization_filter(
            graph, positions, filter_config
        )
        
        assert len(filtered_positions) <= 3
    
    def test_update_subgraph_positions(self):
        """测试更新子图位置"""
        # 创建模拟的全局图和子图
        global_graph = GlobalGraph(
            vocabulary={'word1': 1, 'word2': 2, 'word3': 3},
            reverse_vocabulary={1: 'word1', 2: 'word2', 3: 'word3'}
        )
        
        # 创建激活掩码，激活节点1和2（索引1和2对应节点1和2）
        activation_mask = np.array([False, True, True, False])  # 索引0不用，1和2激活，3不激活
        
        subgraph = StateSubgraph(
            state_name="test_state",
            parent_global_graph=global_graph,
            activation_mask=activation_mask
        )
        
        global_positions = {1: (0.1, 0.2), 2: (0.3, 0.4), 3: (0.5, 0.6)}
        
        # 更新子图位置
        self.engine.update_subgraph_positions(subgraph, global_positions)
        
        # 检查激活节点的位置被更新
        assert subgraph.get_node_position(1) == (0.1, 0.2)
        assert subgraph.get_node_position(2) == (0.3, 0.4)
        assert subgraph.get_node_position(3) is None  # 未激活的节点
    
    def test_clear_cache(self):
        """测试清理缓存"""
        graph = MockGraph([1, 2], [(1, 2)])
        
        # 计算布局以创建缓存
        self.engine.compute_layout(graph, "test_graph")
        
        # 清理缓存
        self.engine.clear_cache("test_graph")
        
        # 再次计算应该重新计算而不是使用缓存
        result = self.engine.compute_layout(graph, "test_graph")
        assert result.metadata.get('from_cache') is not True
    
    def test_get_layout_info(self):
        """测试获取布局信息"""
        info = self.engine.get_layout_info()
        
        assert 'algorithm' in info
        assert 'random_seed' in info
        assert 'max_iterations' in info
        assert 'cache_enabled' in info
        assert info['algorithm'] == 'force_directed'
        assert info['random_seed'] == 42
    
    @patch('semantic_coword_pipeline.processors.deterministic_layout_engine.eg', None)
    def test_missing_easygraph(self):
        """测试缺少EasyGraph的情况"""
        graph = MockGraph([1, 2], [(1, 2)])
        
        # 由于错误处理器会捕获异常并提供回退，我们需要检查结果而不是异常
        result = self.engine.compute_layout(graph, "test_graph")
        
        # 应该使用回退策略
        assert result.metadata.get('fallback_used') is True
        assert result.metadata.get('error') is not None


# 属性测试
try:
    from hypothesis import given, strategies as st, settings
    from hypothesis import HealthCheck
    
    class TestLayoutEngineProperties:
        """布局引擎属性测试"""
        
        def setup_method(self):
            """设置测试环境"""
            self.temp_dir = tempfile.mkdtemp()
            config = Config()
            config.set('layout_engine.random_seed', 42)
            config.set('layout_engine.max_iterations', 10)
            config.set('output.base_path', self.temp_dir)
            self.engine = DeterministicLayoutEngine(config)
        
        def teardown_method(self):
            """清理测试环境"""
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        @given(
            nodes=st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=20, unique=True),
            seed=st.integers(min_value=1, max_value=1000)
        )
        @settings(max_examples=50, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
        def test_layout_determinism_property(self, nodes, seed):
            """
            属性1: 布局确定性
            对于任何给定的图结构和相同的随机种子，多次运行布局算法应该产生完全相同的节点位置坐标
            **验证：需求 1.1**
            """
            # 创建简单的连通图
            edges = []
            for i in range(len(nodes) - 1):
                edges.append((nodes[i], nodes[i + 1]))
            
            graph = MockGraph(nodes, edges)
            
            # 设置相同的种子
            self.engine.params.random_seed = seed
            
            # 多次计算布局
            result1 = self.engine.compute_layout(graph, f"test_graph_{seed}_1", force_recompute=True)
            result2 = self.engine.compute_layout(graph, f"test_graph_{seed}_2", force_recompute=True)
            
            # 位置应该完全相同
            assert result1.positions == result2.positions, f"Layout not deterministic for seed {seed}"
        
        @given(
            nodes=st.lists(st.integers(min_value=1, max_value=50), min_size=2, max_size=10, unique=True),
            min_degree=st.integers(min_value=0, max_value=3),
            max_nodes=st.integers(min_value=1, max_value=20)
        )
        @settings(max_examples=30, deadline=10000)
        def test_visualization_filter_consistency_property(self, nodes, min_degree, max_nodes):
            """
            属性2: 可视化过滤一致性
            对于任何网络图，当应用相同的边权阈值和节点重要性过滤条件时，
            过滤后的节点和边数量应该符合预设的阈值要求
            **验证：需求 1.3**
            """
            # 创建图
            edges = [(nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes))]
            graph = MockGraph(nodes, edges)
            
            # 创建位置
            positions = {node: (float(node), 0.0) for node in nodes}
            
            # 应用过滤
            filter_config = VisualizationFilter(
                min_degree=min_degree,
                max_nodes=max_nodes
            )
            
            filtered_graph, filtered_positions = self.engine.apply_visualization_filter(
                graph, positions, filter_config
            )
            
            # 验证过滤结果
            assert len(filtered_positions) <= max_nodes, "Filtered nodes exceed max_nodes limit"
            
            # 验证度约束
            for node_id in filtered_positions:
                degree = graph.degree(node_id)
                assert degree >= min_degree, f"Node {node_id} degree {degree} < min_degree {min_degree}"
        
        @given(
            nodes=st.lists(st.integers(min_value=1, max_value=30), min_size=1, max_size=15, unique=True)
        )
        @settings(max_examples=30, deadline=10000)
        def test_position_cache_consistency_property(self, nodes):
            """
            属性3: 节点位置缓存一致性
            对于任何图结构，当启用位置缓存时，相同节点在总图和所有子图中的位置坐标应该完全一致
            **验证：需求 1.4, 2.4, 6.4**
            """
            # 创建图
            edges = []
            if len(nodes) > 1:
                for i in range(len(nodes) - 1):
                    edges.append((nodes[i], nodes[i + 1]))
            
            graph = MockGraph(nodes, edges)
            graph_id = f"cache_test_{'_'.join(map(str, sorted(nodes)))}"
            
            # 第一次计算
            result1 = self.engine.compute_layout(graph, graph_id, force_recompute=True)
            
            # 第二次计算（应该使用缓存）
            result2 = self.engine.compute_layout(graph, graph_id)
            
            # 位置应该完全一致
            assert result1.positions == result2.positions, "Cached positions not consistent"
            
            # 模拟子图位置更新
            global_positions = result1.positions
            
            # 创建模拟的全局图和子图
            vocab = {f'word{node}': node for node in nodes}
            reverse_vocab = {node: f'word{node}' for node in nodes}
            
            global_graph = GlobalGraph(
                vocabulary=vocab,
                reverse_vocabulary=reverse_vocab
            )
            
            # 激活部分节点
            activation_mask = np.zeros(max(nodes) + 1, dtype=bool)  # 确保数组足够大
            active_indices = nodes[:len(nodes)//2] if len(nodes) > 1 else nodes
            for node in active_indices:
                if node < len(activation_mask):  # 确保索引有效
                    activation_mask[node] = True
            
            subgraph = StateSubgraph(
                state_name="test_state",
                parent_global_graph=global_graph,
                activation_mask=activation_mask
            )
            
            # 更新子图位置
            self.engine.update_subgraph_positions(subgraph, global_positions)
            
            # 验证激活节点的位置一致性
            for node in active_indices:
                if node in global_positions and subgraph.is_node_active(node):
                    cached_pos = subgraph.get_node_position(node)
                    global_pos = global_positions[node]
                    assert cached_pos == global_pos, f"Position inconsistent for node {node}"

except ImportError:
    # 如果没有hypothesis，跳过属性测试
    pass


if __name__ == '__main__':
    pytest.main([__file__])