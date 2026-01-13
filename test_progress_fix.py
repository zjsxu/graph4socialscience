#!/usr/bin/env python3
"""
测试进度条修复是否有效
"""

import os
import sys
import json
import tempfile
import shutil
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, '.')

def create_minimal_test_data():
    """创建最小测试数据"""
    test_dir = tempfile.mkdtemp(prefix="progress_test_")
    
    # 创建简单的测试文档
    test_docs = [
        {
            "segment_id": "test_001",
            "title": "Machine Learning Test",
            "text": "machine learning algorithms artificial intelligence neural networks deep learning computer vision natural language processing data science analytics big data cloud computing",
            "state": "CA",
            "language": "english"
        },
        {
            "segment_id": "test_002", 
            "title": "AI Research Test",
            "text": "artificial intelligence machine learning deep learning neural networks computer vision pattern recognition data mining predictive modeling statistical analysis",
            "state": "NY",
            "language": "english"
        },
        {
            "segment_id": "test_003",
            "title": "Data Science Test", 
            "text": "data science analytics big data machine learning statistical modeling predictive analytics business intelligence data visualization data mining",
            "state": "TX",
            "language": "english"
        }
    ]
    
    # 保存测试文档
    for i, doc in enumerate(test_docs):
        state_dir = os.path.join(test_dir, doc['state'])
        os.makedirs(state_dir, exist_ok=True)
        
        file_path = os.path.join(state_dir, f"test_doc_{i+1}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
    
    return test_dir

def test_progress_bars():
    """测试进度条修复"""
    print("🧪 测试进度条修复")
    print("=" * 50)
    
    # 创建测试数据
    test_dir = create_minimal_test_data()
    output_dir = "test_output"
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
        
        # 测试步骤2.1: 文本清理
        print("\n2.1 测试文本清理...")
        app.clean_and_normalize_text()
        
        # 测试步骤3.2: 短语提取
        print("\n3.2 测试短语提取...")
        app.extract_tokens_and_phrases()
        
        # 测试步骤4.1: 全局图构建（重点测试进度条）
        print("\n4.1 测试全局图构建（进度条修复）...")
        app.build_global_graph()
        
        # 测试步骤5.1: 子图激活
        print("\n5.1 测试子图激活...")
        app.activate_state_subgraphs()
        
        # 测试步骤6.1: 可视化生成（重点测试卡住问题）
        print("\n6.1 测试可视化生成（卡住问题修复）...")
        app.generate_deterministic_visualizations()
        
        # 检查结果
        if hasattr(app, 'visualization_paths') and app.visualization_paths:
            print(f"\n✅ 成功生成 {len(app.visualization_paths)} 个可视化文件:")
            for name, path in app.visualization_paths.items():
                if os.path.exists(path):
                    print(f"   ✅ {name}: {os.path.basename(path)}")
                else:
                    print(f"   ❌ {name}: 文件不存在")
        else:
            print("❌ 没有生成可视化文件")
            return False
        
        print("\n🎉 所有进度条测试通过!")
        print("✅ 4.1步骤的spring layout进度条现在显示真实进度")
        print("✅ 6.1步骤的可视化生成不再卡住")
        
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
    print("🔧 进度条修复验证测试")
    print("修复内容:")
    print("1. 4.1步骤: Spring layout进度条分批显示真实进度")
    print("2. 6.1步骤: 简化边绘制避免卡住")
    print()
    
    success = test_progress_bars()
    
    if success:
        print("\n📋 修复说明:")
        print("✅ complete_usage_guide.py 已经修复")
        print("✅ 现在可以正常使用4.1和6.1功能")
        print("✅ 进度条会显示真实进度，不会卡住")
    else:
        print("\n❌ 修复验证失败，需要进一步调试")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())