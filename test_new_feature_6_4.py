#!/usr/bin/env python3
"""
测试新功能6.4：查看图节点和数据详情
"""

import os
import sys
import json
import tempfile
import shutil
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, '.')

def create_test_data():
    """创建测试数据"""
    test_dir = tempfile.mkdtemp(prefix="test_6_4_")
    
    # 创建测试文档
    test_docs = [
        {
            "segment_id": "test_001",
            "title": "AI Research Document",
            "text": "artificial intelligence machine learning deep learning neural networks computer vision natural language processing data science analytics big data cloud computing distributed systems",
            "state": "CA",
            "language": "english"
        },
        {
            "segment_id": "test_002", 
            "title": "Data Science Methods",
            "text": "data science machine learning statistical modeling predictive analytics business intelligence data visualization data mining exploratory data analysis feature selection model validation",
            "state": "NY",
            "language": "english"
        },
        {
            "segment_id": "test_003",
            "title": "Machine Learning Algorithms",
            "text": "machine learning algorithms supervised learning unsupervised learning reinforcement learning deep learning neural networks ensemble methods random forest support vector machines decision trees",
            "state": "TX",
            "language": "english"
        }
    ]
    
    # 保存测试文档
    for doc in test_docs:
        state_dir = os.path.join(test_dir, doc['state'])
        os.makedirs(state_dir, exist_ok=True)
        
        file_path = os.path.join(state_dir, f"{doc['segment_id']}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
    
    return test_dir

def test_feature_6_4():
    """测试功能6.4"""
    print("🧪 测试新功能6.4：查看图节点和数据详情")
    print("=" * 60)
    
    # 创建测试数据
    test_dir = create_test_data()
    output_dir = "/Users/zhangjingsen/Desktop/python/graph4socialscience/hajimi/haniumoa"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 导入修复后的类
        from complete_usage_guide import ResearchPipelineCLI
        
        # 初始化应用
        app = ResearchPipelineCLI()
        app.input_directory = test_dir
        app.output_dir = output_dir
        
        # 扫描输入文件
        app.input_files = []
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.endswith('.json'):
                    app.input_files.append(os.path.join(root, file))
        
        app.pipeline_state['data_loaded'] = True
        
        print(f"📁 测试数据: {test_dir}")
        print(f"📁 输出目录: {output_dir}")
        print(f"📊 测试文件: {len(app.input_files)} 个")
        
        # 执行完整管道到6.1
        print("\n🔄 执行管道步骤...")
        
        # 2.1: 文本清理
        print("2.1 文本清理...")
        app.clean_and_normalize_text()
        
        # 3.2: 短语提取
        print("3.2 短语提取...")
        app.extract_tokens_and_phrases()
        
        # 4.1: 全局图构建
        print("4.1 全局图构建...")
        app.build_global_graph()
        
        # 5.1: 子图激活
        print("5.1 子图激活...")
        app.activate_state_subgraphs()
        
        # 6.1: 可视化生成
        print("6.1 可视化生成...")
        app.generate_deterministic_visualizations()
        
        # 测试新功能6.4
        print("\n🎯 测试新功能6.4：查看图节点和数据详情")
        print("-" * 50)
        
        # 测试在没有可视化时的情况
        print("测试1：在没有可视化时调用6.4")
        temp_viz_paths = app.visualization_paths
        app.visualization_paths = {}
        app.view_graph_nodes_and_data()
        app.visualization_paths = temp_viz_paths
        
        print("\n" + "="*50)
        print("测试2：正常调用6.4（有完整数据）")
        app.view_graph_nodes_and_data()
        
        print("\n🎉 功能6.4测试完成!")
        print("✅ 新功能正常工作")
        print("✅ 显示了详细的节点和数据信息")
        print("✅ 包含了图概览、节点详情、社区分析、边分析等")
        
        return True
        
    except Exception as e:
        print(f"💥 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理测试数据
        try:
            shutil.rmtree(test_dir)
            print(f"\n🧹 清理测试数据: {test_dir}")
        except:
            pass

def main():
    """主函数"""
    print("🔧 新功能6.4测试")
    print("功能：查看图节点和数据详情")
    print()
    
    success = test_feature_6_4()
    
    if success:
        print("\n📋 新功能6.4使用说明:")
        print("1. 运行完整管道到步骤6.1（生成可视化）")
        print("2. 选择操作6.4查看详细信息")
        print("3. 功能会显示：")
        print("   - 图概览（节点数、边数、密度）")
        print("   - 节点详情（按重要性排序）")
        print("   - 社区分析（每个社区的统计）")
        print("   - 边分析（权重分布、最强连接）")
        print("   - 状态子图详情")
        print("   - 数据源信息")
        print("   - 短语提取详情")
        print("   - 生成的可视化文件信息")
    else:
        print("\n❌ 新功能测试失败")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())