#!/usr/bin/env python3
"""
快速开始脚本

这个脚本帮助新用户快速开始使用语义增强共词网络分析管线。
它会自动检查环境、创建示例数据并运行完整的演示流程。
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要Python 3.8+")
        return False
    
    print(f"✅ Python版本: {sys.version}")
    
    # 检查关键依赖
    required_packages = ['numpy', 'pandas', 'nltk', 'matplotlib', 'networkx', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\\n📦 需要安装缺失的包: {', '.join(missing_packages)}")
        print("运行以下命令安装:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def download_nltk_data():
    """下载NLTK数据"""
    print("\\n📚 下载NLTK数据...")
    
    try:
        import nltk
        
        # 下载必要的NLTK数据
        nltk_data = ['punkt', 'stopwords']
        
        for data in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
                print(f"✅ NLTK {data} 已存在")
            except LookupError:
                print(f"📥 下载 NLTK {data}...")
                nltk.download(data, quiet=True)
                print(f"✅ NLTK {data} 下载完成")
        
        return True
        
    except Exception as e:
        print(f"❌ NLTK数据下载失败: {e}")
        return False

def create_demo_data():
    """创建演示数据"""
    print("\\n📊 创建演示数据...")
    
    # 创建输出目录
    demo_output_dir = "quick_start_output"
    os.makedirs(demo_output_dir, exist_ok=True)
    
    # 创建演示输入数据
    demo_input_dir = "quick_start_input"
    os.makedirs(demo_input_dir, exist_ok=True)
    
    demo_data = [
        {
            "segment_id": "demo_001",
            "title": "Introduction to Network Analysis",
            "level": 1,
            "order": 1,
            "text": "Network analysis is a powerful method for understanding complex relationships in social science research. Graph theory provides the mathematical foundation for analyzing social networks, communication patterns, and information flow.",
            "state": "CA",
            "language": "en"
        },
        {
            "segment_id": "demo_002",
            "title": "Social Network Theory",
            "level": 2,
            "order": 2,
            "text": "Social network theory examines social structures through the lens of graph theory. Nodes represent individual actors within the network, while edges represent relationships between actors. This approach reveals patterns of social interaction and influence.",
            "state": "CA",
            "language": "en"
        },
        {
            "segment_id": "demo_003",
            "title": "Community Detection Methods",
            "level": 2,
            "order": 3,
            "text": "Community detection algorithms identify groups of densely connected nodes within networks. These methods help researchers understand social clustering, information communities, and organizational structures in complex social systems.",
            "state": "NY",
            "language": "en"
        },
        {
            "segment_id": "demo_004",
            "title": "网络分析在社会科学中的应用",
            "level": 1,
            "order": 4,
            "text": "网络分析方法在社会科学研究中发挥着重要作用。通过分析社会关系网络，研究者可以深入理解社会结构、信息传播模式和群体行为特征。这种方法为社会科学研究提供了新的视角和工具。",
            "state": "NY",
            "language": "zh"
        },
        {
            "segment_id": "demo_005",
            "title": "文本挖掘与共词分析",
            "level": 2,
            "order": 5,
            "text": "文本挖掘技术结合共词分析方法，能够从大量文本数据中提取知识结构和主题关联。通过构建词汇共现网络，研究者可以识别研究领域的核心概念、发展趋势和知识演化路径。",
            "state": "TX",
            "language": "zh"
        },
        {
            "segment_id": "demo_006",
            "title": "Visualization Techniques",
            "level": 1,
            "order": 6,
            "text": "Network visualization techniques transform complex graph structures into interpretable visual representations. Force-directed layouts, community coloring, and node sizing based on centrality measures help researchers communicate network insights effectively.",
            "state": "TX",
            "language": "en"
        }
    ]
    
    # 保存演示数据
    for i, doc in enumerate(demo_data):
        doc_file = os.path.join(demo_input_dir, f"demo_doc_{i+1:02d}.json")
        with open(doc_file, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 创建了 {len(demo_data)} 个演示文档")
    print(f"📁 输入目录: {demo_input_dir}")
    print(f"📁 输出目录: {demo_output_dir}")
    
    return demo_input_dir, demo_output_dir

def run_demo_pipeline(input_dir, output_dir):
    """运行演示管线"""
    print("\\n🚀 运行演示管线...")
    
    try:
        from complete_usage_guide import ResearchPipelineCLI
        
        # 初始化管线
        app = ResearchPipelineCLI()
        
        # 设置输入输出目录
        app.input_directory = input_dir
        app.output_dir = output_dir
        
        # 扫描输入文件
        app.input_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.json'):
                    app.input_files.append(os.path.join(root, file))
        
        app.pipeline_state['data_loaded'] = True
        print(f"✅ 加载了 {len(app.input_files)} 个文件")
        
        # 运行管线步骤
        print("\\n📋 执行管线步骤:")
        
        # 1. 文本清理
        print("1️⃣ 文本清理...")
        app.clean_and_normalize_text()
        
        # 2. 词组提取
        print("2️⃣ 词组提取...")
        app.extract_tokens_and_phrases()
        
        # 3. 全局图构建
        print("3️⃣ 全局图构建...")
        app.build_global_graph()
        
        # 4. 子图激活
        print("4️⃣ 子图激活...")
        app.activate_state_subgraphs()
        
        # 5. 可视化生成
        print("5️⃣ 可视化生成...")
        if hasattr(app, 'generate_deterministic_visualizations'):
            app.generate_deterministic_visualizations()
        else:
            app.generate_scientific_visualizations()
        
        # 6. 结果导出
        print("6️⃣ 结果导出...")
        app.export_complete_results()
        
        print("\\n🎉 演示管线运行完成！")
        
        # 显示结果
        if hasattr(app, 'visualization_paths') and app.visualization_paths:
            print("\\n📊 生成的可视化文件:")
            for name, path in app.visualization_paths.items():
                abs_path = os.path.abspath(path)
                print(f"   {name}: {abs_path}")
        
        print(f"\\n📁 所有结果保存在: {os.path.abspath(output_dir)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示管线运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_next_steps():
    """显示后续步骤"""
    print("\\n" + "=" * 60)
    print("🎯 后续步骤")
    print("=" * 60)
    
    print("\\n✅ 快速开始演示已完成！现在您可以:")
    print("\\n1. 📊 查看生成的可视化结果")
    print("   - 打开 quick_start_output/visualizations/ 目录")
    print("   - 查看网络图像文件")
    
    print("\\n2. 🔍 探索详细结果")
    print("   - 查看 quick_start_output/ 目录下的所有输出")
    print("   - 包括清理文本、图数据、统计报告等")
    
    print("\\n3. 🚀 使用自己的数据")
    print("   - 准备TOC JSON格式的数据文件")
    print("   - 运行: python complete_usage_guide.py")
    print("   - 或使用: python quick_pipeline_setup.py")
    
    print("\\n4. 📚 学习更多功能")
    print("   - 阅读 README.md 了解详细功能")
    print("   - 查看 docs/ 目录的文档")
    print("   - 运行测试: pytest")
    
    print("\\n5. 🔧 自定义配置")
    print("   - 修改 config/default_config.json")
    print("   - 调整可视化参数和处理选项")
    
    print("\\n💡 提示:")
    print("   - 如果遇到问题，请查看GitHub Issues")
    print("   - 欢迎贡献代码和反馈")

def main():
    """主函数"""
    print("🚀 Graph4SocialScience 快速开始")
    print("=" * 60)
    print("欢迎使用语义增强共词网络分析管线！")
    print("这个脚本将帮助您快速体验系统功能。")
    print("=" * 60)
    
    # 1. 检查环境
    if not check_environment():
        print("\\n❌ 环境检查失败，请先安装依赖")
        return 1
    
    # 2. 下载NLTK数据
    if not download_nltk_data():
        print("\\n❌ NLTK数据下载失败")
        return 1
    
    # 3. 创建演示数据
    try:
        input_dir, output_dir = create_demo_data()
    except Exception as e:
        print(f"\\n❌ 创建演示数据失败: {e}")
        return 1
    
    # 4. 运行演示管线
    if not run_demo_pipeline(input_dir, output_dir):
        print("\\n❌ 演示管线运行失败")
        return 1
    
    # 5. 显示后续步骤
    show_next_steps()
    
    print("\\n🎉 快速开始完成！享受使用Graph4SocialScience！")
    return 0

if __name__ == "__main__":
    sys.exit(main())