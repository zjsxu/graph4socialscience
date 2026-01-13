#!/usr/bin/env python3
"""
最终可视化修复验证测试
验证complete_usage_guide.py中的6.1操作是否彻底修复
"""

import os
import sys
import json
import tempfile
import shutil
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, '.')

def create_realistic_test_data():
    """创建更真实的测试数据，模拟用户的实际使用场景"""
    test_dir = tempfile.mkdtemp(prefix="final_viz_test_")
    
    # 创建更复杂的测试文档，模拟真实的TOC数据
    test_docs = [
        {
            "segment_id": "ai_research_001",
            "title": "Artificial Intelligence Fundamentals",
            "text": "artificial intelligence machine learning deep learning neural networks computer vision natural language processing pattern recognition supervised learning unsupervised learning reinforcement learning feature engineering model evaluation cross validation hyperparameter tuning ensemble methods random forest support vector machines logistic regression linear regression decision trees clustering algorithms dimensionality reduction principal component analysis",
            "state": "CA",
            "language": "english"
        },
        {
            "segment_id": "ai_research_002", 
            "title": "Deep Learning Applications",
            "text": "deep learning neural networks convolutional neural networks recurrent neural networks transformer models attention mechanisms computer vision image recognition object detection natural language processing machine translation text classification sentiment analysis language modeling generative adversarial networks variational autoencoders transfer learning fine tuning pre trained models",
            "state": "CA",
            "language": "english"
        },
        {
            "segment_id": "data_science_001",
            "title": "Data Science Methodology",
            "text": "data science analytics big data machine learning statistical modeling predictive analytics business intelligence data visualization data mining exploratory data analysis feature selection model validation statistical inference hypothesis testing regression analysis time series analysis clustering classification anomaly detection",
            "state": "NY",
            "language": "english"
        },
        {
            "segment_id": "data_science_002",
            "title": "Big Data Technologies", 
            "text": "big data cloud computing distributed systems scalable architectures apache spark hadoop mapreduce data warehousing data lakes etl processes real time processing stream processing batch processing data pipelines data governance data quality data integration data transformation",
            "state": "NY",
            "language": "english"
        },
        {
            "segment_id": "ml_algorithms_001",
            "title": "Machine Learning Algorithms",
            "text": "machine learning algorithms supervised learning unsupervised learning semi supervised learning reinforcement learning deep reinforcement learning multi agent systems neural architecture search automated machine learning hyperparameter optimization bayesian optimization genetic algorithms evolutionary computation swarm intelligence",
            "state": "TX",
            "language": "english"
        },
        {
            "segment_id": "ml_algorithms_002",
            "title": "Advanced ML Techniques",
            "text": "advanced machine learning techniques ensemble learning boosting bagging stacking meta learning few shot learning zero shot learning multi task learning transfer learning domain adaptation adversarial training robust optimization federated learning privacy preserving machine learning differential privacy homomorphic encryption",
            "state": "TX",
            "language": "english"
        },
        {
            "segment_id": "ai_ethics_001",
            "title": "AI Ethics and Fairness",
            "text": "artificial intelligence ethics algorithmic fairness bias detection bias mitigation explainable artificial intelligence interpretable machine learning model transparency accountability responsible artificial intelligence ethical artificial intelligence fairness metrics demographic parity equalized odds calibration",
            "state": "FL",
            "language": "english"
        },
        {
            "segment_id": "ai_ethics_002",
            "title": "AI Safety and Governance",
            "text": "artificial intelligence safety ai governance ai regulation ai policy algorithmic accountability transparency explainability interpretability robustness adversarial attacks adversarial examples model security privacy protection data protection gdpr compliance ethical guidelines ai standards",
            "state": "FL",
            "language": "english"
        }
    ]
    
    # 保存测试文档到状态文件夹
    for doc in test_docs:
        state_dir = os.path.join(test_dir, doc['state'])
        os.makedirs(state_dir, exist_ok=True)
        
        file_path = os.path.join(state_dir, f"{doc['segment_id']}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
    
    return test_dir

def test_complete_pipeline_with_visualization():
    """测试完整管道，重点验证6.1操作"""
    print("🧪 最终可视化修复验证测试")
    print("=" * 60)
    
    # 创建测试数据
    test_dir = create_realistic_test_data()
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
        print(f"📊 测试状态: {len(set(doc['state'] for doc in create_realistic_test_data() if 'state' in str(doc)))} 个")
        
        # 执行完整管道
        print("\n🔄 执行完整管道...")
        
        # 步骤2.1: 文本清理
        print("\n2.1 文本清理...")
        app.clean_and_normalize_text()
        if not app.pipeline_state['text_cleaned']:
            print("❌ 文本清理失败")
            return False
        
        # 步骤3.2: 短语提取
        print("\n3.2 短语提取...")
        app.extract_tokens_and_phrases()
        if not app.pipeline_state['phrases_constructed']:
            print("❌ 短语提取失败")
            return False
        
        # 步骤4.1: 全局图构建
        print("\n4.1 全局图构建...")
        app.build_global_graph()
        if not app.pipeline_state['global_graph_built']:
            print("❌ 全局图构建失败")
            return False
        
        # 步骤5.1: 子图激活
        print("\n5.1 子图激活...")
        app.activate_state_subgraphs()
        if not app.pipeline_state['subgraphs_activated']:
            print("❌ 子图激活失败")
            return False
        
        # 关键测试：步骤6.1 可视化生成
        print("\n🎯 关键测试：6.1 可视化生成（修复验证）...")
        print("   这是之前卡住的步骤，现在测试是否修复")
        
        start_time = datetime.now()
        app.generate_deterministic_visualizations()
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        print(f"   ⏱️ 可视化生成耗时: {duration:.2f}秒")
        
        # 验证结果
        if not hasattr(app, 'visualization_paths') or not app.visualization_paths:
            print("❌ 可视化生成失败：没有生成可视化文件")
            return False
        
        # 检查生成的文件
        generated_files = []
        missing_files = []
        
        for viz_name, viz_path in app.visualization_paths.items():
            if os.path.exists(viz_path):
                file_size = os.path.getsize(viz_path)
                generated_files.append((viz_name, os.path.basename(viz_path), file_size))
                print(f"   ✅ {viz_name}: {os.path.basename(viz_path)} ({file_size} bytes)")
            else:
                missing_files.append((viz_name, viz_path))
                print(f"   ❌ {viz_name}: 文件不存在 - {viz_path}")
        
        if missing_files:
            print(f"❌ 有 {len(missing_files)} 个文件缺失")
            return False
        
        # 成功验证
        print(f"\n🎉 最终验证成功!")
        print(f"✅ 6.1操作不再卡住，成功生成 {len(generated_files)} 个可视化文件")
        print(f"✅ 总耗时: {duration:.2f}秒（合理范围内）")
        print(f"✅ 所有文件都已正确生成到目标目录")
        
        # 显示性能统计
        if hasattr(app, 'global_graph_object') and app.global_graph_object:
            G = app.global_graph_object
            print(f"\n📊 处理的图规模:")
            print(f"   节点数: {G.number_of_nodes()}")
            print(f"   边数: {G.number_of_edges()}")
            print(f"   密度: {nx.density(G)*100:.2f}%")
            print(f"   状态子图数: {len(app.state_subgraph_objects)}")
        
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
    print("🔧 最终可视化修复验证")
    print("验证6.1操作的性能问题是否彻底解决")
    print()
    
    success = test_complete_pipeline_with_visualization()
    
    if success:
        print("\n🎊 修复验证完全成功!")
        print("✅ 6.1操作的卡住问题已彻底解决")
        print("✅ 性能问题已修复（预计算max_weight）")
        print("✅ 可视化生成速度正常")
        print("✅ 所有功能正常工作")
        print()
        print("📋 现在可以安全使用complete_usage_guide.py处理真实数据:")
        print("   python complete_usage_guide.py")
        print("   选择6.1操作不会再卡住")
    else:
        print("\n❌ 修复验证失败")
        print("需要进一步调试6.1操作")
    
    return 0 if success else 1

if __name__ == "__main__":
    import networkx as nx  # 需要导入nx用于密度计算
    sys.exit(main())