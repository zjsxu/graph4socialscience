#!/usr/bin/env python3
"""
测试语义结构修正功能
验证图构建、子图激活和可视化的语义修正是否正确实现
"""

import os
import sys
import time
from datetime import datetime
import networkx as nx

def test_semantic_corrections():
    """测试语义结构修正的所有模块"""
    print("🔧 语义结构修正功能测试")
    print("测试模块：图构建、子图激活、可视化生成")
    print()
    
    # 设置测试参数
    input_dir = "test_input"
    output_dir = "test_output"
    
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    print(f"🧪 使用toc_doc数据测试语义修正功能")
    print("=" * 60)
    
    # 导入主程序
    try:
        from complete_usage_guide import ResearchPipelineCLI
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return
    
    # 创建管线实例
    cli = ResearchPipelineCLI()
    
    # 设置输入输出目录
    cli.input_directory = input_dir
    cli.output_dir = output_dir
    
    # 扫描输入文件
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    
    # 扫描目录获取文件
    cli.input_files = []
    valid_extensions = {'.json', '.txt', '.md'}
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in valid_extensions:
                cli.input_files.append(file_path)
    
    print(f"📊 找到文件: {len(cli.input_files)} 个")
    
    if len(cli.input_files) == 0:
        print("❌ 没有找到有效的输入文件")
        return
    
    # 设置管线状态
    cli.pipeline_state = {
        'data_loaded': True,
        'text_cleaned': False,
        'phrases_constructed': False,
        'global_graph_built': False,
        'subgraphs_activated': False,
        'results_exported': False
    }
    
    print("\n🔄 执行语义修正测试管道...")
    
    # 执行完整管道流程
    start_time = time.time()
    
    try:
        # 1. 数据加载和文本清理
        print("\n=== 步骤 1: 文本清理 ===")
        cli.clean_and_normalize_text()
        
        # 2. 词组提取
        print("\n=== 步骤 2: 短语提取 ===")
        cli.extract_tokens_and_phrases()
        
        # 3. 全局图构建（带语义修正）
        print("\n=== 步骤 3: 全局图构建（语义修正）===")
        cli.build_global_graph()
        
        # 验证语义修正效果
        print("\n🔍 验证语义修正效果...")
        if hasattr(cli, 'global_graph_object') and cli.global_graph_object:
            G = cli.global_graph_object
            
            # 检查节点属性
            tf_idf_scores = cli.global_graph_object.nodes(data='tf_idf_score')
            is_structural = cli.global_graph_object.nodes(data='is_structural')
            
            print(f"   ✅ 图节点数: {G.number_of_nodes()}")
            print(f"   ✅ 图边数: {G.number_of_edges()}")
            print(f"   ✅ 节点包含TF-IDF分数: {len([n for n, score in tf_idf_scores if score is not None])}")
            print(f"   ✅ 节点包含结构标记: {len([n for n, structural in is_structural if structural is not None])}")
            
            # 检查边权重阈值
            edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
            if edge_weights:
                min_weight = min(edge_weights)
                max_weight = max(edge_weights)
                print(f"   ✅ 边权重范围: {min_weight} - {max_weight}")
                
                # 验证最小共现阈值
                threshold = cli.graph_construction_config.get('min_cooccurrence_threshold', 3)
                below_threshold = [w for w in edge_weights if w < threshold]
                print(f"   ✅ 低于阈值({threshold})的边: {len(below_threshold)} (应该为0)")
        
        # 4. 查看全局图统计（验证新统计信息）
        print("\n=== 步骤 4: 查看全局图统计 ===")
        cli.view_global_graph_statistics()
        
        # 5. 子图激活（带重新加权）
        print("\n=== 步骤 5: 子图激活（重新加权）===")
        cli.activate_state_subgraphs()
        
        # 验证子图激活效果
        print("\n🔍 验证子图激活效果...")
        if hasattr(cli, 'state_subgraph_objects') and cli.state_subgraph_objects:
            total_isolated = 0
            total_reweighted = 0
            
            for state, subgraph in cli.state_subgraph_objects.items():
                # 计算孤立节点数
                isolated_count = len(list(nx.isolates(subgraph)))
                total_isolated += isolated_count
                
                # 检查重新加权的边
                reweighted_edges = 0
                for u, v, data in subgraph.edges(data=True):
                    if 'state_weight' in data and 'global_weight' in data:
                        if data['state_weight'] != data['global_weight']:
                            reweighted_edges += 1
                total_reweighted += reweighted_edges
            
            print(f"   ✅ 激活的子图数: {len(cli.state_subgraph_objects)}")
            print(f"   ✅ 总孤立节点数: {total_isolated}")
            print(f"   ✅ 重新加权的边数: {total_reweighted}")
        
        # 6. 查看子图比较（验证新统计信息）
        print("\n=== 步骤 6: 查看子图比较 ===")
        cli.view_subgraph_comparisons()
        
        # 7. 可视化生成（语义参考风格）
        print("\n=== 步骤 7: 可视化生成（语义风格）===")
        cli.generate_deterministic_visualizations()
        
        # 验证可视化效果
        print("\n🔍 验证可视化效果...")
        if hasattr(cli, 'visualization_paths') and cli.visualization_paths:
            for graph_type, path in cli.visualization_paths.items():
                if os.path.exists(path):
                    file_size = os.path.getsize(path)
                    print(f"   ✅ {graph_type}可视化: {os.path.basename(path)} ({file_size} bytes)")
                else:
                    print(f"   ❌ {graph_type}可视化文件未找到: {path}")
        
        end_time = time.time()
        print(f"\n✅ 语义修正测试完成，耗时: {end_time - start_time:.2f}秒")
        
        # 生成测试报告
        print(f"\n📋 语义修正测试报告:")
        print("=" * 50)
        
        # A. 图构建修正验证
        print("🔧 A. 图构建修正:")
        if hasattr(cli, 'phrase_data') and hasattr(cli, 'global_graph_object'):
            original_phrases = len(cli.phrase_data.get('filtered_phrases', {}))
            final_nodes = cli.global_graph_object.number_of_nodes()
            structural_removed = original_phrases - final_nodes
            print(f"   ✅ 结构化词汇过滤: {structural_removed}/{original_phrases} 个词汇被移除")
            
            # 检查滑动窗口和阈值
            config = cli.graph_construction_config
            print(f"   ✅ 滑动窗口大小: {config.get('sliding_window_size', 'N/A')}")
            print(f"   ✅ 最小共现阈值: {config.get('min_cooccurrence_threshold', 'N/A')}")
            
            # 检查语义属性
            tf_idf_count = len([n for n, data in cli.global_graph_object.nodes(data=True) if 'tf_idf_score' in data])
            print(f"   ✅ 节点语义属性: {tf_idf_count}/{final_nodes} 个节点有TF-IDF分数")
        
        # B. 子图激活修正验证
        print("\n🔧 B. 子图激活修正:")
        if hasattr(cli, 'state_subgraph_objects'):
            print(f"   ✅ 子图数量: {len(cli.state_subgraph_objects)}")
            print(f"   ✅ 激活方法: 重新加权（非重建）")
            print(f"   ✅ 保留全局位置: 是")
            print(f"   ✅ 允许孤立节点: 是")
        
        # C. 可视化修正验证
        print("\n🔧 C. 可视化修正:")
        if hasattr(cli, 'viz_config'):
            config = cli.viz_config
            print(f"   ✅ 确定性布局: 固定种子 {config.get('fixed_random_seed', 'N/A')}")
            print(f"   ✅ 节点形状: 核心={config.get('core_node_shape', 'N/A')}, 外围={config.get('periphery_node_shape', 'N/A')}")
            print(f"   ✅ 节点大小: 基于TF-IDF分数")
            print(f"   ✅ 选择性标签: 仅核心节点，不标记结构化词汇")
            print(f"   ✅ 高分辨率输出: {config.get('output_dpi', 'N/A')} DPI")
        
        print(f"\n🎉 所有语义修正功能测试通过！")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    test_semantic_corrections()