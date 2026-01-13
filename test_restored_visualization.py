#!/usr/bin/env python3
"""
测试恢复后的可视化效果

这个脚本测试恢复后的可视化方法，验证是否能生成更丰富的图像输出。
"""

import os
import sys
from datetime import datetime
from complete_usage_guide import ResearchPipelineCLI

def test_restored_visualization():
    """测试恢复后的可视化效果"""
    
    print("🧪 测试恢复后的可视化效果")
    print("=" * 50)
    
    try:
        # 初始化pipeline
        print("🔄 初始化pipeline...")
        app = ResearchPipelineCLI()
        
        # 设置输出目录
        output_dir = "test_output"
        app.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 输出目录: {output_dir}")
        
        # 创建示例数据
        print("📊 创建示例数据...")
        app.create_sample_research_data()
        
        # 运行pipeline步骤
        print("\\n🔄 运行pipeline步骤...")
        
        # 步骤1: 文本清理
        print("1️⃣ 文本清理...")
        app.clean_and_normalize_text()
        
        if not app.pipeline_state['text_cleaned']:
            print("❌ 文本清理失败")
            return False
        
        # 步骤2: 词组提取
        print("2️⃣ 词组提取...")
        app.extract_tokens_and_phrases()
        
        if not app.pipeline_state['phrases_constructed']:
            print("❌ 词组提取失败")
            return False
        
        # 步骤3: 全局图构建
        print("3️⃣ 全局图构建...")
        app.build_global_graph()
        
        if not app.pipeline_state['global_graph_built']:
            print("❌ 全局图构建失败")
            return False
        
        # 检查图的规模
        if hasattr(app, 'global_graph_object') and app.global_graph_object:
            nodes = app.global_graph_object.number_of_nodes()
            edges = app.global_graph_object.number_of_edges()
            print(f"   📊 全局图规模: {nodes} 节点, {edges} 边")
            
            if nodes < 50:
                print("   ⚠️ 节点数较少，可能需要调整参数")
            else:
                print("   ✅ 节点数合理，应该能生成丰富的可视化")
        
        # 步骤4: 子图激活
        print("4️⃣ 子图激活...")
        app.activate_state_subgraphs()
        
        if not app.pipeline_state['subgraphs_activated']:
            print("❌ 子图激活失败")
            return False
        
        # 步骤5: 生成可视化（使用恢复的方法）
        print("5️⃣ 生成可视化（恢复版本）...")
        
        # 检查是否有generate_deterministic_visualizations方法
        if hasattr(app, 'generate_deterministic_visualizations'):
            print("   ✅ 找到恢复的generate_deterministic_visualizations方法")
            app.generate_deterministic_visualizations()
        else:
            print("   ❌ 未找到generate_deterministic_visualizations方法")
            print("   🔄 尝试使用scientific visualization...")
            app.generate_scientific_visualizations()
        
        # 检查生成的可视化
        if hasattr(app, 'visualization_paths') and app.visualization_paths:
            print(f"\\n✅ 成功生成 {len(app.visualization_paths)} 个可视化文件:")
            
            for viz_name, viz_path in app.visualization_paths.items():
                abs_path = os.path.abspath(viz_path)
                file_size = os.path.getsize(abs_path) if os.path.exists(abs_path) else 0
                print(f"   📊 {viz_name}:")
                print(f"      路径: {abs_path}")
                print(f"      大小: {file_size:,} bytes ({file_size/1024:.1f} KB)")
                
                if file_size > 100000:  # > 100KB
                    print(f"      ✅ 文件大小合理，可能包含丰富内容")
                else:
                    print(f"      ⚠️ 文件较小，可能内容简化")
            
            return True
        else:
            print("❌ 未生成任何可视化文件")
            return False
    
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_with_scientific_version():
    """对比scientific版本的效果"""
    
    print("\\n🔍 对比scientific版本效果")
    print("-" * 40)
    
    try:
        # 初始化pipeline
        app = ResearchPipelineCLI()
        
        # 检查scientific配置
        print("📋 当前scientific配置:")
        for key, value in app.scientific_config.items():
            print(f"   {key}: {value}")
        
        print("\\n📋 当前graph construction配置:")
        if hasattr(app, 'graph_construction_config'):
            for key, value in app.graph_construction_config.items():
                print(f"   {key}: {value}")
        
        # 分析配置的影响
        print("\\n📊 配置分析:")
        
        if not app.scientific_config.get('enable_lcc_extraction', True):
            print("   ✅ LCC extraction已禁用 - 保留更多节点")
        else:
            print("   ⚠️ LCC extraction仍启用 - 可能过度简化")
        
        if not app.scientific_config.get('enable_community_pruning', True):
            print("   ✅ Community pruning已禁用 - 保留更多社区")
        else:
            print("   ⚠️ Community pruning仍启用 - 可能合并小社区")
        
        edge_retention = app.scientific_config.get('edge_retention_rate', 0.05)
        if edge_retention >= 0.2:
            print(f"   ✅ Edge retention rate: {edge_retention*100:.0f}% - 保留更多边")
        else:
            print(f"   ⚠️ Edge retention rate: {edge_retention*100:.0f}% - 可能过度稀疏")
        
        min_edge_weight = app.graph_construction_config.get('min_edge_weight', 2) if hasattr(app, 'graph_construction_config') else 2
        if min_edge_weight <= 1:
            print(f"   ✅ Min edge weight: {min_edge_weight} - 保留更多弱连接")
        else:
            print(f"   ⚠️ Min edge weight: {min_edge_weight} - 可能过滤太多边")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置检查失败: {e}")
        return False

def main():
    """主函数"""
    print("🔄 恢复可视化效果测试")
    print("=" * 60)
    
    # 对比配置
    compare_success = compare_with_scientific_version()
    
    if not compare_success:
        print("❌ 配置检查失败，可能恢复不完整")
        return 1
    
    # 测试可视化
    test_success = test_restored_visualization()
    
    if test_success:
        print("\\n🎉 测试成功！")
        print("📊 恢复后的可视化应该包含更多节点和边")
        print("🎯 现在可以用这个版本处理真实数据了")
        return 0
    else:
        print("\\n❌ 测试失败")
        print("🔧 可能需要进一步调整参数")
        return 1

if __name__ == "__main__":
    sys.exit(main())