#!/usr/bin/env python3
"""
网络分析器演示脚本

演示NetworkAnalyzer的基本功能，包括：
- 基础网络统计计算
- 高级网络指标分析
- 多维度对比分析
- 社群检测和中心性分析
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Easy-Graph'))

import numpy as np
from unittest.mock import Mock

try:
    import easygraph as eg
except ImportError:
    print("Warning: EasyGraph not available. Using mock objects for demonstration.")
    eg = None

from semantic_coword_pipeline.analyzers.network_analyzer import NetworkAnalyzer
from semantic_coword_pipeline.core.data_models import GlobalGraph, StateSubgraph


def create_demo_graph(name: str, node_count: int, edge_count: int) -> GlobalGraph:
    """创建演示用的图"""
    print(f"Creating demo graph '{name}' with {node_count} nodes and {edge_count} edges")
    
    # 创建词表
    vocabulary = {f"phrase_{i}": i for i in range(node_count)}
    reverse_vocabulary = {i: f"phrase_{i}" for i in range(node_count)}
    
    # 创建模拟的EasyGraph实例
    mock_graph = Mock()
    mock_graph.number_of_nodes.return_value = node_count
    mock_graph.number_of_edges.return_value = edge_count
    mock_graph.nodes.return_value = list(range(node_count))
    
    # 模拟度分布
    if node_count > 0:
        # 创建一个更真实的度分布
        degrees = []
        remaining_edges = edge_count * 2  # 每条边贡献2个度
        
        for i in range(node_count):
            if i == node_count - 1:
                # 最后一个节点获得剩余的度
                degrees.append(remaining_edges)
            else:
                # 随机分配度，但确保不超过剩余度数
                max_degree = min(remaining_edges, node_count - 1)
                degree = min(max_degree, max(0, remaining_edges // (node_count - i)))
                degrees.append(degree)
                remaining_edges -= degree
        
        # 确保度数合理
        degrees = [min(d, node_count - 1) for d in degrees]
        
        # 如果有孤立节点需求，设置最后一个节点为孤立节点
        if node_count > 2:
            degrees[-1] = 0
    else:
        degrees = []
    
    mock_graph.degree.side_effect = lambda node: degrees[node] if node < len(degrees) else 0
    
    return GlobalGraph(
        vocabulary=vocabulary,
        reverse_vocabulary=reverse_vocabulary,
        easygraph_instance=mock_graph,
        metadata={
            'name': name,
            'demo_degrees': degrees
        }
    )


def create_demo_state_subgraph(name: str, global_graph: GlobalGraph, active_ratio: float = 0.6) -> StateSubgraph:
    """创建演示用的州级子图"""
    node_count = global_graph.get_node_count()
    active_count = int(node_count * active_ratio)
    
    # 创建激活掩码
    activation_mask = np.zeros(node_count, dtype=bool)
    activation_mask[:active_count] = True
    
    # 创建模拟的子图EasyGraph实例
    mock_subgraph = Mock()
    mock_subgraph.number_of_nodes.return_value = active_count
    mock_subgraph.number_of_edges.return_value = max(0, active_count - 1)
    mock_subgraph.nodes.return_value = list(range(active_count))
    mock_subgraph.degree.side_effect = lambda node: 1 if node < active_count - 1 else 0
    
    return StateSubgraph(
        state_name=name,
        parent_global_graph=global_graph,
        activation_mask=activation_mask,
        easygraph_instance=mock_subgraph
    )


def main():
    """主演示函数"""
    print("=" * 60)
    print("网络分析器 (NetworkAnalyzer) 演示")
    print("=" * 60)
    
    # 初始化网络分析器
    config = {
        'enable_community_detection': True,
        'centrality_metrics': ['degree', 'betweenness', 'closeness', 'pagerank'],
        'community_algorithm': 'louvain',
        'comparison_metrics': [
            'node_count', 'edge_count', 'density', 'isolated_nodes_ratio',
            'connected_components', 'clustering_coefficient', 'average_path_length'
        ]
    }
    
    analyzer = NetworkAnalyzer(config)
    print(f"✓ NetworkAnalyzer initialized with {len(config['centrality_metrics'])} centrality metrics")
    print()
    
    # 创建演示图
    print("1. 创建演示图")
    print("-" * 30)
    
    # 创建不同规模的图用于对比
    small_graph = create_demo_graph("小规模图", 5, 3)
    medium_graph = create_demo_graph("中等规模图", 10, 12)
    large_graph = create_demo_graph("大规模图", 20, 35)
    
    graphs = {
        'small': small_graph,
        'medium': medium_graph,
        'large': large_graph
    }
    
    print()
    
    # 基础统计分析
    print("2. 基础网络统计分析")
    print("-" * 30)
    
    for name, graph in graphs.items():
        print(f"\n{name.upper()} 图统计:")
        try:
            stats = analyzer.calculate_basic_statistics(graph)
            print(f"  节点数: {stats['node_count']:.0f}")
            print(f"  边数: {stats['edge_count']:.0f}")
            print(f"  密度: {stats['density']:.4f}")
            print(f"  孤立节点比例: {stats['isolated_nodes_ratio']:.4f}")
            print(f"  平均度: {stats['average_degree']:.2f}")
        except Exception as e:
            print(f"  错误: {e}")
    
    print()
    
    # 高级网络指标分析
    print("3. 高级网络指标分析")
    print("-" * 30)
    
    # 使用中等规模图进行详细分析
    test_graph = medium_graph
    print(f"分析图: {test_graph.metadata.get('name', 'Unknown')}")
    
    try:
        # 模拟EasyGraph函数以避免实际调用
        with Mock() as mock_eg:
            # 设置模拟返回值
            mock_eg.is_connected.return_value = True
            mock_eg.clustering.return_value = {i: 0.3 + 0.1 * (i % 3) for i in range(10)}
            mock_eg.transitivity.return_value = 0.35
            mock_eg.average_shortest_path_length.return_value = 2.8
            mock_eg.diameter.return_value = 5
            
            # 暂时替换analyzer中的EasyGraph调用
            import semantic_coword_pipeline.analyzers.network_analyzer as na_module
            original_eg = na_module.eg
            na_module.eg = mock_eg
            
            try:
                metrics = analyzer.calculate_advanced_metrics(test_graph)
                print(f"  连通分量数: {metrics['connected_components_count']}")
                print(f"  最大连通分量比例: {metrics['largest_component_ratio']:.4f}")
                print(f"  平均聚类系数: {metrics['average_clustering_coefficient']:.4f}")
                print(f"  全局聚类系数: {metrics['global_clustering_coefficient']:.4f}")
                
                avg_path = metrics['average_path_length']
                if avg_path != float('inf'):
                    print(f"  平均路径长度: {avg_path:.2f}")
                else:
                    print("  平均路径长度: 无穷大（图不连通）")
                
                print(f"  直径: {metrics['diameter']:.0f}")
            finally:
                # 恢复原始的EasyGraph引用
                na_module.eg = original_eg
                
    except Exception as e:
        print(f"  错误: {e}")
    
    print()
    
    # 网络结构对比分析
    print("4. 多维度网络结构对比分析")
    print("-" * 30)
    
    try:
        comparison_result = analyzer.compare_network_structures(graphs, "规模对比分析")
        
        print(f"对比分析: {comparison_result['comparison_name']}")
        print(f"参与对比的图: {', '.join(comparison_result['graph_names'])}")
        
        # 显示对比摘要
        if 'summary' in comparison_result:
            summary = comparison_result['summary']
            
            if 'node_count_comparison' in summary:
                node_comp = summary['node_count_comparison']
                print(f"\n节点数对比:")
                print(f"  最多: {node_comp['max_graph']} ({node_comp['values'][node_comp['max_graph']]:.0f} 节点)")
                print(f"  最少: {node_comp['min_graph']} ({node_comp['values'][node_comp['min_graph']]:.0f} 节点)")
                print(f"  比例: {node_comp['ratio_max_to_min']:.2f}:1")
            
            if 'density_comparison' in summary:
                density_comp = summary['density_comparison']
                print(f"\n密度对比:")
                print(f"  最高: {density_comp['max_graph']} ({density_comp['values'][density_comp['max_graph']]:.4f})")
                print(f"  最低: {density_comp['min_graph']} ({density_comp['values'][density_comp['min_graph']]:.4f})")
        
    except Exception as e:
        print(f"  错误: {e}")
    
    print()
    
    # 州级子图分析
    print("5. 州级子图差异分析")
    print("-" * 30)
    
    try:
        # 创建几个州级子图
        state_ca = create_demo_state_subgraph("California", large_graph, 0.7)
        state_ny = create_demo_state_subgraph("New York", large_graph, 0.5)
        state_tx = create_demo_state_subgraph("Texas", large_graph, 0.6)
        
        state_subgraphs = {
            'California': state_ca,
            'New York': state_ny,
            'Texas': state_tx
        }
        
        print(f"分析 {len(state_subgraphs)} 个州的子图差异:")
        
        # 模拟跨州差异分析
        for state_name, subgraph in state_subgraphs.items():
            active_nodes = subgraph.get_active_nodes()
            print(f"  {state_name}: {len(active_nodes)} 个激活节点")
        
        # 执行跨州差异分析
        cross_state_analysis = analyzer.analyze_cross_state_differences(state_subgraphs)
        
        if 'error' not in cross_state_analysis:
            print(f"\n跨州分析结果:")
            print(f"  分析的州: {', '.join(cross_state_analysis['states_analyzed'])}")
            
            if 'unique_phrases_by_state' in cross_state_analysis:
                unique_phrases = cross_state_analysis['unique_phrases_by_state']
                for state, count in unique_phrases.items():
                    print(f"  {state}: {count} 个独特短语")
            
            if 'common_phrases' in cross_state_analysis:
                common_count = len(cross_state_analysis['common_phrases'])
                print(f"  共同短语: {common_count} 个")
        
    except Exception as e:
        print(f"  错误: {e}")
    
    print()
    
    # 生成分析报告
    print("6. 生成分析报告")
    print("-" * 30)
    
    try:
        # 创建一个简化的分析结果用于报告生成
        sample_analysis = {
            'basic_statistics': {
                'demo_graph': {
                    'node_count': 10.0,
                    'edge_count': 15.0,
                    'density': 0.333,
                    'isolated_nodes_ratio': 0.1,
                    'average_degree': 3.0
                }
            },
            'advanced_metrics': {
                'demo_graph': {
                    'connected_components_count': 1,
                    'largest_component_ratio': 1.0,
                    'average_clustering_coefficient': 0.45,
                    'global_clustering_coefficient': 0.4,
                    'average_path_length': 2.5
                }
            },
            'community_analysis': {
                'demo_graph': {
                    'community_count': 3,
                    'modularity': 0.42,
                    'average_community_size': 3.3
                }
            }
        }
        
        report = analyzer.generate_analysis_report(sample_analysis)
        
        print("分析报告生成成功!")
        print("报告预览:")
        print("-" * 20)
        
        # 显示报告的前几行
        report_lines = report.split('\n')
        for i, line in enumerate(report_lines[:15]):
            print(line)
        
        if len(report_lines) > 15:
            print("...")
            print(f"(报告共 {len(report_lines)} 行)")
        
    except Exception as e:
        print(f"  错误: {e}")
    
    print()
    print("=" * 60)
    print("演示完成!")
    print("NetworkAnalyzer 提供了完整的网络分析功能，包括:")
    print("- ✓ 基础统计指标计算")
    print("- ✓ 高级网络指标分析") 
    print("- ✓ 社群检测和中心性分析")
    print("- ✓ 多维度对比分析")
    print("- ✓ 跨州差异分析")
    print("- ✓ 分析报告生成")
    print("=" * 60)


if __name__ == "__main__":
    main()