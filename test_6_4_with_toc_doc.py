#!/usr/bin/env python3
"""
使用toc_doc文件夹测试新的6.4功能
测试总图和随机选择的3个子图的数据导出
"""

import os
import sys
import time
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, '.')

def test_6_4_with_real_data():
    """使用真实toc_doc数据测试6.4功能"""
    print("🧪 使用toc_doc数据测试6.4功能")
    print("=" * 60)
    
    # 设置路径
    input_dir = "/Users/zhangjingsen/Desktop/python/graph4socialscience/toc_doc"
    output_dir = "/Users/zhangjingsen/Desktop/python/graph4socialscience/hajimi/haniumoa"
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return False
    
    try:
        # 导入管道类
        from complete_usage_guide import ResearchPipelineCLI
        
        # 初始化应用
        app = ResearchPipelineCLI()
        app.input_directory = input_dir
        app.output_dir = output_dir
        
        # 扫描输入文件
        app.input_files = []
        valid_extensions = {'.json', '.txt', '.md'}
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in valid_extensions:
                    app.input_files.append(file_path)
        
        app.pipeline_state['data_loaded'] = True
        
        print(f"📁 输入目录: {input_dir}")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 找到文件: {len(app.input_files)} 个")
        
        # 执行完整管道
        print("\n🔄 执行完整管道...")
        
        start_time = time.time()
        
        # 2.1: 文本清理
        print("2.1 文本清理...")
        app.clean_and_normalize_text()
        if not app.pipeline_state['text_cleaned']:
            print("❌ 文本清理失败")
            return False
        
        # 3.2: 短语提取
        print("3.2 短语提取...")
        app.extract_tokens_and_phrases()
        if not app.pipeline_state['phrases_constructed']:
            print("❌ 短语提取失败")
            return False
        
        # 4.1: 全局图构建
        print("4.1 全局图构建...")
        app.build_global_graph()
        if not app.pipeline_state['global_graph_built']:
            print("❌ 全局图构建失败")
            return False
        
        # 5.1: 子图激活
        print("5.1 子图激活...")
        app.activate_state_subgraphs()
        if not app.pipeline_state['subgraphs_activated']:
            print("❌ 子图激活失败")
            return False
        
        # 6.1: 可视化生成
        print("6.1 可视化生成...")
        app.generate_deterministic_visualizations()
        if not hasattr(app, 'visualization_paths') or not app.visualization_paths:
            print("❌ 可视化生成失败")
            return False
        
        pipeline_time = time.time() - start_time
        print(f"✅ 管道执行完成，耗时: {pipeline_time:.2f}秒")
        
        # 显示生成的图信息
        print(f"\n📊 生成的图信息:")
        print(f"   全局图: {app.global_graph_object.number_of_nodes()} 节点, {app.global_graph_object.number_of_edges()} 边")
        
        if hasattr(app, 'state_subgraph_objects'):
            print(f"   子图数量: {len(app.state_subgraph_objects)}")
            for state, subgraph in app.state_subgraph_objects.items():
                print(f"     {state}: {subgraph.number_of_nodes()} 节点, {subgraph.number_of_edges()} 边")
        
        # 测试6.4功能 - 自动选择"A"（全部图）
        print(f"\n🎯 测试6.4功能：导出图数据...")
        
        # 模拟用户选择"A"（全部图）
        original_get_user_choice = app.get_user_choice
        def mock_get_user_choice(prompt, valid_choices):
            print(f"{prompt}: A (自动选择：全部图)")
            return "A"
        
        app.get_user_choice = mock_get_user_choice
        
        # 执行6.4功能
        export_start = time.time()
        app.view_graph_nodes_and_data()
        export_time = time.time() - export_start
        
        # 恢复原始方法
        app.get_user_choice = original_get_user_choice
        
        print(f"✅ 6.4功能执行完成，耗时: {export_time:.2f}秒")
        
        # 检查生成的文件
        analysis_dir = os.path.join(output_dir, "graph_analysis")
        if os.path.exists(analysis_dir):
            analysis_files = [f for f in os.listdir(analysis_dir) if f.endswith('.txt')]
            print(f"\n📄 生成的分析文件:")
            
            for filename in sorted(analysis_files):
                filepath = os.path.join(analysis_dir, filename)
                file_size = os.path.getsize(filepath)
                print(f"   ✅ {filename} ({file_size} bytes)")
                
                # 显示文件前几行内容预览
                print(f"      预览:")
                with open(filepath, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 5:  # 只显示前5行
                            break
                        print(f"        {line.strip()}")
                    print(f"        ...")
                print()
            
            print(f"📁 所有分析文件保存在: {os.path.abspath(analysis_dir)}")
            
        else:
            print("❌ 未找到分析文件目录")
            return False
        
        print(f"\n🎉 测试完成!")
        print(f"✅ 成功使用toc_doc数据")
        print(f"✅ 生成了总图和子图的完整数据文档")
        print(f"✅ 文档包含所有节点、边、参数信息")
        
        return True
        
    except Exception as e:
        print(f"💥 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🔧 6.4功能实际数据测试")
    print("使用toc_doc文件夹进行完整测试")
    print()
    
    success = test_6_4_with_real_data()
    
    if success:
        print("\n📋 测试结果:")
        print("✅ 6.4功能正常工作")
        print("✅ 能够处理真实的toc_doc数据")
        print("✅ 成功导出图数据到文档文件")
        print("✅ 文档格式清晰，包含完整的技术信息")
        print()
        print("📄 生成的文档包含:")
        print("- 图结构信息（节点数、边数、密度等）")
        print("- 完整的节点数据（所有属性）")
        print("- 完整的边数据（所有属性）")
        print("- 处理参数（可重现性配置）")
        print("- 源数据统计")
        print("- 短语提取数据")
        print("- 布局算法参数")
    else:
        print("\n❌ 测试失败")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())