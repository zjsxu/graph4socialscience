#!/usr/bin/env python3
"""
确定性布局引擎演示

展示DeterministicLayoutEngine的基本功能，包括：
- 固定种子的力导向布局
- 节点位置缓存机制
- 层级布局
- 可视化过滤功能
"""

import numpy as np
from semantic_coword_pipeline.processors.deterministic_layout_engine import (
    DeterministicLayoutEngine,
    LayoutParameters,
    VisualizationFilter
)
from semantic_coword_pipeline.core.config import Config
from semantic_coword_pipeline.core.data_models import GlobalGraph, StateSubgraph


class MockGraph:
    """模拟EasyGraph图对象用于演示"""
    
    def __init__(self, nodes, edges):
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
    
    def degree(self, node_id):
        return self._node_degrees.get(node_id, 0)


def main():
    print("=== 确定性布局引擎演示 ===\n")
    
    # 1. 初始化布局引擎
    print("1. 初始化布局引擎")
    config = Config()
    config.set('layout_engine.random_seed', 42)
    config.set('layout_engine.algorithm', 'force_directed')
    config.set('layout_engine.max_iterations', 50)
    
    engine = DeterministicLayoutEngine(config)
    print(f"   算法: {engine.params.algorithm}")
    print(f"   随机种子: {engine.params.random_seed}")
    print(f"   最大迭代次数: {engine.params.max_iterations}")
    
    # 2. 创建示例图
    print("\n2. 创建示例图")
    nodes = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5)]
    graph = MockGraph(nodes, edges)
    print(f"   节点: {nodes}")
    print(f"   边: {edges}")
    
    # 3. 计算力导向布局
    print("\n3. 计算力导向布局")
    result = engine.compute_layout(graph, "demo_graph")
    print(f"   算法: {result.algorithm_used}")
    print(f"   迭代次数: {result.iterations_completed}")
    print(f"   是否收敛: {result.converged}")
    print("   节点位置:")
    for node_id, pos in result.positions.items():
        print(f"     节点{node_id}: ({pos[0]:.3f}, {pos[1]:.3f})")
    
    # 4. 测试缓存功能
    print("\n4. 测试缓存功能")
    result2 = engine.compute_layout(graph, "demo_graph")
    print(f"   使用缓存: {result2.metadata.get('from_cache', False)}")
    print(f"   位置一致: {result.positions == result2.positions}")
    
    # 5. 测试层级布局
    print("\n5. 测试层级布局")
    engine.params.algorithm = 'hierarchical'
    result_hier = engine.compute_layout(graph, "demo_graph_hier", force_recompute=True)
    print(f"   算法: {result_hier.algorithm_used}")
    print("   节点位置:")
    for node_id, pos in result_hier.positions.items():
        print(f"     节点{node_id}: ({pos[0]:.3f}, {pos[1]:.3f})")
    
    # 6. 测试可视化过滤
    print("\n6. 测试可视化过滤")
    filter_config = VisualizationFilter(min_degree=2, max_nodes=3)
    filtered_graph, filtered_positions = engine.apply_visualization_filter(
        graph, result.positions, filter_config
    )
    print(f"   原始节点数: {len(result.positions)}")
    print(f"   过滤后节点数: {len(filtered_positions)}")
    print("   过滤后节点:")
    for node_id, pos in filtered_positions.items():
        print(f"     节点{node_id}: ({pos[0]:.3f}, {pos[1]:.3f}), 度: {graph.degree(node_id)}")
    
    print("\n=== 演示完成 ===")


if __name__ == '__main__':
    main()