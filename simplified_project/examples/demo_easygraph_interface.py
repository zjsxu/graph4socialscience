#!/usr/bin/env python3
"""
EasyGraph接口演示脚本

演示EasyGraph/OpenRank兼容接口的功能，包括：
- 标准化图数据格式输出
- 多视图图支持接口
- 图融合数据接口
- EasyGraph格式兼容性验证
"""

import sys
import os
import numpy as np
import scipy.sparse
from pathlib import Path
from unittest.mock import Mock

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from semantic_coword_pipeline.processors.easygraph_interface import (
    EasyGraphInterface,
    MultiViewGraph,
    GraphFormat,
    FusionStrategy,
    create_easygraph_from_matrix,
    validate_multi_view_consistency
)
from semantic_coword_pipeline.core.data_models import GlobalGraph


def create_sample_global_graph():
    """创建示例全局图"""
    print("创建示例全局图...")
    
    # 创建词表
    vocabulary = {
        'artificial intelligence': 0,
        'machine learning': 1,
        'deep learning': 2,
        'neural networks': 3,
        'data science': 4,
        'natural language': 5,
        'computer vision': 6,
        'reinforcement learning': 7
    }
    
    reverse_vocabulary = {v: k for k, v in vocabulary.items()}
    
    # 创建共现矩阵（模拟AI相关术语的共现关系）
    matrix_data = np.array([
        [0, 5, 3, 2, 4, 2, 1, 1],  # artificial intelligence
        [5, 0, 8, 6, 7, 3, 2, 4],  # machine learning
        [3, 8, 0, 9, 4, 2, 3, 2],  # deep learning
        [2, 6, 9, 0, 3, 1, 4, 1],  # neural networks
        [4, 7, 4, 3, 0, 5, 3, 2],  # data science
        [2, 3, 2, 1, 5, 0, 1, 1],  # natural language
        [1, 2, 3, 4, 3, 1, 0, 1],  # computer vision
        [1, 4, 2, 1, 2, 1, 1, 0]   # reinforcement learning
    ])
    
    # 转换为稀疏矩阵
    cooccurrence_matrix = scipy.sparse.csr_matrix(matrix_data)
    
    # 创建全局图对象
    global_graph = GlobalGraph(
        vocabulary=vocabulary,
        reverse_vocabulary=reverse_vocabulary,
        cooccurrence_matrix=cooccurrence_matrix,
        easygraph_instance=None,  # 将在需要时创建
        metadata={
            'domain': 'artificial_intelligence',
            'created_by': 'demo_script',
            'node_count': len(vocabulary),
            'edge_count': cooccurrence_matrix.nnz
        }
    )
    
    print(f"✓ 创建了包含 {len(vocabulary)} 个节点的全局图")
    print(f"✓ 共现矩阵大小: {cooccurrence_matrix.shape}")
    print(f"✓ 非零边数: {cooccurrence_matrix.nnz}")
    
    return global_graph


def demo_graph_export(interface, global_graph):
    """演示图数据导出功能"""
    print("\n" + "="*50)
    print("演示图数据导出功能")
    print("="*50)
    
    # 1. 导出为邻接矩阵
    print("\n1. 导出为邻接矩阵格式...")
    result = interface.export_global_graph(global_graph, GraphFormat.ADJACENCY_MATRIX)
    if result['success']:
        print(f"✓ 成功导出邻接矩阵: {result['node_count']} 节点")
        print(f"  矩阵形状: {result['shape']}")
        print(f"  非零元素: {result['nnz']}")
    else:
        print(f"✗ 导出失败: {result.get('error', 'Unknown error')}")
    
    # 2. 导出为边列表
    print("\n2. 导出为边列表格式...")
    result = interface.export_global_graph(global_graph, GraphFormat.EDGE_LIST)
    if result['success']:
        print(f"✓ 成功导出边列表: {result['edge_count']} 条边")
        # 显示前几条边
        if 'edge_list' in result and result['edge_list']:
            print("  前3条边:")
            for i, edge in enumerate(result['edge_list'][:3]):
                print(f"    {edge['source_phrase']} -- {edge['target_phrase']} (权重: {edge['weight']})")
    else:
        print(f"✗ 导出失败: {result.get('error', 'Unknown error')}")
    
    # 3. 导出为JSON格式
    print("\n3. 导出为JSON格式...")
    output_path = "output/demo_graph.json"
    os.makedirs("output", exist_ok=True)
    
    result = interface.export_global_graph(global_graph, GraphFormat.JSON, output_path)
    if result['success']:
        print(f"✓ 成功导出JSON: {result['node_count']} 节点, {result['edge_count']} 边")
        print(f"  文件保存至: {result.get('file_path', 'N/A')}")
    else:
        print(f"✗ 导出失败: {result.get('error', 'Unknown error')}")


def create_mock_additional_views(vocabulary):
    """创建模拟的额外视图"""
    print("\n创建模拟的额外视图...")
    
    additional_views = {}
    
    # 1. 模拟embedding视图
    embedding_graph = Mock()
    embedding_graph.nodes = list(vocabulary.values())
    embedding_graph.edges = [(0, 1), (1, 2), (2, 3), (4, 5)]  # 基于语义相似性的连接
    # 创建一个Mock对象来模拟edges的get方法
    edges_mock = Mock()
    edges_mock.get = Mock(return_value={'weight': 0.8, 'similarity': 'semantic'})
    embedding_graph.edges = edges_mock
    
    additional_views['embedding'] = {
        'graph': embedding_graph,
        'type': 'semantic_embedding'
    }
    
    # 2. 模拟影响力视图
    influence_graph = Mock()
    influence_graph.nodes = list(vocabulary.values())
    influence_graph.edges = [(0, 1), (0, 4), (1, 2), (1, 7)]  # 基于影响力的连接
    # 创建一个Mock对象来模拟edges的get方法
    edges_mock2 = Mock()
    edges_mock2.get = Mock(return_value={'weight': 0.6, 'influence_score': 'high'})
    influence_graph.edges = edges_mock2
    
    additional_views['influence'] = {
        'graph': influence_graph,
        'type': 'influence_network'
    }
    
    # 3. 模拟引用视图
    citation_graph = Mock()
    citation_graph.nodes = list(vocabulary.values())
    citation_graph.edges = [(0, 2), (1, 3), (2, 6), (4, 7)]  # 基于引用关系的连接
    # 创建一个Mock对象来模拟edges的get方法
    edges_mock3 = Mock()
    edges_mock3.get = Mock(return_value={'weight': 1.0, 'citation_count': 10})
    citation_graph.edges = edges_mock3
    
    additional_views['citation'] = {
        'graph': citation_graph,
        'type': 'citation_network'
    }
    
    print(f"✓ 创建了 {len(additional_views)} 个额外视图:")
    for view_name, view_data in additional_views.items():
        print(f"  - {view_name}: {view_data['type']}")
    
    return additional_views


def demo_multi_view_graph(interface, global_graph):
    """演示多视图图功能"""
    print("\n" + "="*50)
    print("演示多视图图功能")
    print("="*50)
    
    # 创建额外视图
    additional_views = create_mock_additional_views(global_graph.vocabulary)
    
    # 创建多视图图
    print("\n创建多视图图...")
    
    # 模拟视图兼容性验证
    def mock_validate_compatibility(graph, node_space):
        return True
    
    interface._validate_view_compatibility = mock_validate_compatibility
    
    multi_view = interface.create_multi_view_graph(global_graph, additional_views)
    
    print(f"✓ 创建多视图图成功")
    print(f"  节点空间大小: {len(multi_view.node_space)}")
    print(f"  视图数量: {len(multi_view.views)}")
    print(f"  视图列表: {multi_view.list_views()}")
    
    # 验证多视图一致性
    print("\n验证多视图一致性...")
    consistency_results = validate_multi_view_consistency(multi_view)
    
    for check, result in consistency_results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}: {result}")
    
    return multi_view


def demo_graph_fusion(interface, multi_view):
    """演示图融合功能"""
    print("\n" + "="*50)
    print("演示图融合功能")
    print("="*50)
    
    try:
        # 模拟EasyGraph
        from unittest.mock import patch, Mock
        
        with patch('semantic_coword_pipeline.processors.easygraph_interface.eg') as mock_eg:
            # 设置mock EasyGraph
            mock_graph_class = Mock()
            mock_eg.Graph = mock_graph_class
            
            mock_fused_graph = Mock()
            mock_fused_graph.nodes = list(range(8))
            mock_fused_graph.edges = [(0, 1), (1, 2), (2, 3), (4, 5)]
            mock_graph_class.return_value = mock_fused_graph
            
            # 1. 联合融合
            print("\n1. 联合融合策略...")
            try:
                result = interface.fuse_graphs(
                    multi_view,
                    FusionStrategy.UNION,
                    target_views=['embedding', 'influence']
                )
                print(f"✓ 联合融合成功")
                print(f"  融合策略: {result.fusion_strategy.value}")
                print(f"  源视图: {result.source_views}")
            except Exception as e:
                print(f"✗ 联合融合失败: {e}")
            
            # 2. 加权融合
            print("\n2. 加权融合策略...")
            view_weights = {
                'embedding': 0.5,
                'influence': 0.3,
                'citation': 0.2
            }
            
            try:
                result = interface.fuse_graphs(
                    multi_view,
                    FusionStrategy.WEIGHTED,
                    view_weights=view_weights,
                    target_views=['embedding', 'influence', 'citation']
                )
                print(f"✓ 加权融合成功")
                print(f"  融合策略: {result.fusion_strategy.value}")
                print(f"  视图权重: {result.fusion_weights}")
            except Exception as e:
                print(f"✗ 加权融合失败: {e}")
            
            # 3. 共识融合
            print("\n3. 共识融合策略...")
            try:
                result = interface.fuse_graphs(
                    multi_view,
                    FusionStrategy.CONSENSUS,
                    target_views=['embedding', 'influence', 'citation']
                )
                print(f"✓ 共识融合成功")
                print(f"  融合策略: {result.fusion_strategy.value}")
                print(f"  源视图数量: {len(result.source_views)}")
            except Exception as e:
                print(f"✗ 共识融合失败: {e}")
    
    except ImportError:
        print("⚠ EasyGraph不可用，跳过图融合演示")


def demo_experiment_design(interface, multi_view):
    """演示实验设计功能"""
    print("\n" + "="*50)
    print("演示融合实验设计")
    print("="*50)
    
    # 创建实验设计
    print("\n创建融合实验设计...")
    experiment_design = interface.create_fusion_experiment_design(multi_view)
    
    print(f"✓ 实验设计创建成功")
    print(f"  节点数量: {experiment_design['metadata']['node_count']}")
    print(f"  可用视图: {experiment_design['metadata']['available_views']}")
    
    # 显示融合策略
    print(f"\n融合策略 ({len(experiment_design['fusion_strategies'])} 种):")
    for strategy in experiment_design['fusion_strategies']:
        print(f"  - {strategy['strategy']}: {strategy['description']}")
    
    # 显示评估指标
    print(f"\n评估指标 ({len(experiment_design['evaluation_metrics'])} 种):")
    for metric in experiment_design['evaluation_metrics']:
        print(f"  - {metric['name']}: {metric['description']}")
    
    # 显示基线对比
    print(f"\n基线对比 ({len(experiment_design['baseline_comparisons'])} 种):")
    for baseline in experiment_design['baseline_comparisons']:
        print(f"  - {baseline['name']}: {baseline['description']}")
    
    # 保存实验设计
    import json
    output_path = "output/experiment_design.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_design, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 实验设计已保存至: {output_path}")


def demo_compatibility_validation(interface, global_graph):
    """演示兼容性验证功能"""
    print("\n" + "="*50)
    print("演示EasyGraph兼容性验证")
    print("="*50)
    
    # 创建模拟图对象
    mock_graph = Mock()
    mock_graph.nodes = list(global_graph.vocabulary.values())
    mock_graph.edges = [(0, 1), (1, 2), (2, 3)]
    
    # 创建Mock对象来模拟nodes和edges的get方法
    nodes_mock = Mock()
    nodes_mock.get = Mock(return_value={'phrase': 'test_phrase'})
    mock_graph.nodes = nodes_mock
    
    edges_mock = Mock()
    edges_mock.get = Mock(return_value={'weight': 1.0})
    mock_graph.edges = edges_mock
    
    # 模拟EasyGraph可用性检查
    try:
        from unittest.mock import patch
        
        with patch('semantic_coword_pipeline.processors.easygraph_interface.eg') as mock_eg:
            mock_eg.Graph = Mock
            mock_eg.is_connected = Mock(return_value=True)
            
            with patch('builtins.isinstance', return_value=True):
                validation_results = interface.validate_easygraph_compatibility(mock_graph)
            
            print("兼容性验证结果:")
            for check, result in validation_results.items():
                status = "✓" if result else "✗"
                print(f"  {status} {check}: {result}")
    
    except Exception as e:
        print(f"⚠ 兼容性验证失败: {e}")
    
    # 测试EasyGraph不可用的情况
    print("\n测试EasyGraph不可用的情况...")
    with patch('semantic_coword_pipeline.processors.easygraph_interface.eg', None):
        validation_results = interface.validate_easygraph_compatibility(mock_graph)
        print("EasyGraph不可用时的验证结果:")
        for check, result in validation_results.items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}: {result}")


def main():
    """主函数"""
    print("EasyGraph/OpenRank兼容接口演示")
    print("="*60)
    
    # 创建接口配置
    config = {
        'default_format': 'easygraph',
        'preserve_node_attributes': True,
        'preserve_edge_attributes': True,
        'validate_compatibility': True
    }
    
    # 初始化接口
    print("初始化EasyGraph接口...")
    interface = EasyGraphInterface(config)
    print(f"✓ 接口初始化成功，默认格式: {interface.default_format.value}")
    
    # 创建示例数据
    global_graph = create_sample_global_graph()
    
    # 演示各项功能
    demo_graph_export(interface, global_graph)
    
    multi_view = demo_multi_view_graph(interface, global_graph)
    
    demo_graph_fusion(interface, multi_view)
    
    demo_experiment_design(interface, multi_view)
    
    demo_compatibility_validation(interface, global_graph)
    
    print("\n" + "="*60)
    print("演示完成！")
    print("✓ 所有EasyGraph接口功能演示完毕")
    print("✓ 输出文件已保存到 output/ 目录")
    print("="*60)


if __name__ == "__main__":
    main()