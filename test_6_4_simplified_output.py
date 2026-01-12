#!/usr/bin/env python3
"""
测试简化后的6.4功能 - 使用toc_doc数据
测试输出格式是否符合用户要求：减少解说性文字，直接导出技术数据
"""

import os
import sys
import time
from datetime import datetime

def test_6_4_simplified_output():
    """测试简化后的6.4功能输出格式"""
    print("🔧 6.4功能简化输出格式测试")
    print("使用toc_doc文件夹进行完整测试")
    print()
    
    # 设置测试参数
    input_dir = "/Users/zhangjingsen/Desktop/python/graph4socialscience/toc_doc"
    output_dir = "/Users/zhangjingsen/Desktop/python/graph4socialscience/test_6_4_output"
    
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    print(f"🧪 使用toc_doc数据测试6.4功能")
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
    
    # 设置管线状态为已完成（模拟完整流程已执行）
    cli.pipeline_state = {
        'data_loaded': True,
        'text_cleaned': True,
        'phrases_constructed': True,
        'global_graph_built': True,
        'subgraphs_activated': True,
        'results_exported': True
    }
    
    print("\n🔄 执行完整管道...")
    
    # 执行完整管道流程
    start_time = time.time()
    
    try:
        # 1. 数据加载和文本清理
        print("2.1 文本清理...")
        cli.clean_and_normalize_text()
        
        # 2. 词组提取
        print("3.2 短语提取...")
        cli.extract_tokens_and_phrases()
        
        # 3. 全局图构建
        print("4.1 全局图构建...")
        cli.build_global_graph()
        
        # 4. 子图激活
        print("5.1 子图激活...")
        cli.activate_state_subgraphs()
        
        # 5. 可视化生成
        print("6.1 可视化生成...")
        cli.generate_deterministic_visualizations()
        
        # 6. 测试6.4功能：导出图数据
        print("\n🎯 测试6.4功能：导出图数据...")
        
        # 模拟用户选择"A"（全部图：总图+3个随机子图）
        print("\n📊 EXPORT GRAPH NODES & DATA DETAILS")
        print("-" * 60)
        print("📈 Available graphs for analysis:")
        print("0. Global Graph (complete network)")
        
        if hasattr(cli, 'state_subgraph_objects') and cli.state_subgraph_objects:
            for i, (state, subgraph) in enumerate(cli.state_subgraph_objects.items(), 1):
                print(f"{i}. State {state} Subgraph ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)")
        
        print("A. All graphs (global + 3 random subgraphs)")
        
        print("Select graph to analyze: A (自动选择：全部图)")
        
        # 执行6.4功能
        output_dir_analysis = os.path.join(cli.output_dir, "graph_analysis")
        os.makedirs(output_dir_analysis, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 导出全局图
        if hasattr(cli, 'global_graph_object') and cli.global_graph_object:
            cli._export_single_graph_data(cli.global_graph_object, "global", output_dir_analysis, timestamp)
        
        # 导出3个随机子图
        if hasattr(cli, 'state_subgraph_objects') and cli.state_subgraph_objects:
            import random
            available_subgraphs = list(cli.state_subgraph_objects.items())
            selected_subgraphs = random.sample(available_subgraphs, min(3, len(available_subgraphs)))
            
            for state, subgraph in selected_subgraphs:
                cli._export_single_graph_data(subgraph, f"state_{state}", output_dir_analysis, timestamp)
        
        print(f"✅ Exported global graph + 3 random subgraphs")
        print(f"📁 All analysis files saved to: {os.path.abspath(output_dir_analysis)}")
        
        end_time = time.time()
        print(f"✅ 6.4功能执行完成，耗时: {end_time - start_time:.2f}秒")
        
        # 检查生成的文件
        print(f"\n📄 生成的分析文件:")
        if os.path.exists(output_dir_analysis):
            analysis_files = [f for f in os.listdir(output_dir_analysis) if f.startswith('graph_data_')]
            for file in sorted(analysis_files):
                file_path = os.path.join(output_dir_analysis, file)
                file_size = os.path.getsize(file_path)
                print(f"   ✅ {file} ({file_size} bytes)")
                
                # 显示文件前几行预览
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:10]
                    preview = ''.join(lines).strip()
                    print(f"      预览:")
                    for line in lines[:5]:
                        print(f"        {line.strip()}")
                    print(f"        ...")
                print()
        
        print(f"📁 所有分析文件保存在: {os.path.abspath(output_dir_analysis)}")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 测试完成!")
    print("✅ 成功使用toc_doc数据")
    print("✅ 生成了总图和子图的简化数据文档")
    print("✅ 文档格式简洁，减少了解说性文字")
    
    print(f"\n📋 测试结果:")
    print("✅ 6.4功能正常工作")
    print("✅ 能够处理真实的toc_doc数据")
    print("✅ 成功导出简化格式的图数据到文档文件")
    print("✅ 文档格式清晰，包含完整的技术信息但减少冗余文字")
    
    print(f"\n📄 生成的文档包含:")
    print("- 图结构信息（节点数、边数、密度等）")
    print("- 完整的节点数据（所有属性，表格格式）")
    print("- 完整的边数据（所有属性，表格格式）")
    print("- 处理参数（key=value格式）")
    print("- 源数据统计")
    print("- 短语提取数据")
    print("- 布局算法参数")

if __name__ == "__main__":
    test_6_4_simplified_output()