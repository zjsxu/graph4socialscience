#!/usr/bin/env python3
"""
语义增强共词网络分析管线简化使用指南

这个文件展示了如何使用现有的demo文件和主管线来运行完整的分析流程。

作者: 语义增强共词网络分析团队
版本: 1.0.0
日期: 2024年
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

# 导入主管线
from semantic_coword_pipeline.pipeline import SemanticCowordPipeline


def create_sample_input_data():
    """创建示例输入数据"""
    print("创建示例输入数据...")
    
    # 创建输入目录
    input_dir = Path("demo_input")
    input_dir.mkdir(exist_ok=True)
    
    # 示例TOC文档数据
    sample_documents = [
        {
            "segment_id": "demo_001",
            "title": "Introduction to Machine Learning",
            "level": 1,
            "order": 1,
            "text": "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models. Natural language processing is an important application of machine learning.",
            "state": "CA",
            "language": "en"
        },
        {
            "segment_id": "demo_002", 
            "title": "Deep Learning Applications",
            "level": 2,
            "order": 2,
            "text": "Deep learning has revolutionized computer vision and natural language processing. Neural networks and machine learning algorithms are key components of artificial intelligence systems.",
            "state": "CA",
            "language": "en"
        },
        {
            "segment_id": "demo_003",
            "title": "数据科学基础",
            "level": 1,
            "order": 3,
            "text": "数据科学是一个跨学科领域，结合了统计学、计算机科学和领域专业知识。机器学习和人工智能是数据科学的重要组成部分。",
            "state": "NY",
            "language": "zh"
        },
        {
            "segment_id": "demo_004",
            "title": "自然语言处理技术",
            "level": 2,
            "order": 4,
            "text": "自然语言处理技术包括文本分析、语义理解和机器翻译。深度学习方法在自然语言处理领域取得了显著进展。",
            "state": "NY", 
            "language": "zh"
        },
        {
            "segment_id": "demo_005",
            "title": "AI Ethics and Future",
            "level": 1,
            "order": 5,
            "text": "Artificial intelligence ethics is becoming increasingly important as machine learning systems are deployed in critical applications. The future of AI depends on responsible development of natural language processing and computer vision technologies.",
            "state": "TX",
            "language": "en"
        }
    ]
    
    # 保存示例文档
    for i, doc in enumerate(sample_documents):
        doc_file = input_dir / f"document_{i+1}.json"
        with open(doc_file, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 创建了 {len(sample_documents)} 个示例文档在 {input_dir}")
    return str(input_dir)


def demonstrate_main_pipeline():
    """演示主管线的使用"""
    print("\n" + "=" * 80)
    print("语义增强共词网络分析管线使用演示")
    print("=" * 80)
    
    try:
        # 1. 创建示例输入数据
        input_dir = create_sample_input_data()
        output_dir = "pipeline_demo_output"
        
        print(f"\n1. 输入目录: {input_dir}")
        print(f"   输出目录: {output_dir}")
        
        # 2. 初始化主管线
        print("\n2. 初始化主管线...")
        pipeline = SemanticCowordPipeline()
        print("   ✓ 主管线初始化完成")
        
        # 3. 运行完整处理
        print("\n3. 运行完整处理流程...")
        print("   这可能需要几分钟时间，请耐心等待...")
        
        result = pipeline.run(input_dir, output_dir)
        
        # 4. 显示结果
        print("\n4. 处理结果:")
        print(f"   ✓ 总文件数: {result.total_files}")
        print(f"   ✓ 处理成功: {result.processed_files}")
        print(f"   ✓ 处理失败: {result.failed_files}")
        print(f"   ✓ 处理时间: {result.processing_time:.2f} 秒")
        print(f"   ✓ 输出文件: {len(result.output_files)} 个")
        
        if result.global_graph:
            print(f"   ✓ 全局图节点数: {result.global_graph.get_node_count()}")
            print(f"   ✓ 词汇表大小: {len(result.global_graph.vocabulary)}")
        
        print(f"   ✓ 州级子图数: {len(result.state_subgraphs)}")
        for state in result.state_subgraphs.keys():
            print(f"     - {state} 州")
        
        if result.error_summary.get('summary', {}).get('total_errors', 0) > 0:
            print(f"   ⚠ 错误数量: {result.error_summary['summary']['total_errors']}")
        
        # 5. 显示输出文件
        print(f"\n5. 输出文件列表:")
        for file_path in result.output_files:
            print(f"   - {file_path}")
        
        print(f"\n✓ 所有结果已保存到: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_existing_demos():
    """演示现有的demo文件使用"""
    print("\n" + "=" * 80)
    print("现有Demo文件使用演示")
    print("=" * 80)
    
    # 查找现有的demo文件
    demo_files = [
        "demo.py",
        "demo_document_generator.py", 
        "demo_easygraph_interface.py",
        "demo_layout_engine.py",
        "demo_network_analyzer.py"
    ]
    
    print("\n可用的Demo文件:")
    for demo_file in demo_files:
        if Path(demo_file).exists():
            print(f"   ✓ {demo_file}")
        else:
            print(f"   ✗ {demo_file} (不存在)")
    
    print("\n使用方法:")
    print("1. 基础演示:")
    print("   python demo.py")
    
    print("\n2. 文档生成器演示:")
    print("   python demo_document_generator.py")
    
    print("\n3. EasyGraph接口演示:")
    print("   python demo_easygraph_interface.py")
    
    print("\n4. 布局引擎演示:")
    print("   python demo_layout_engine.py")
    
    print("\n5. 网络分析器演示:")
    print("   python demo_network_analyzer.py")


def demonstrate_cli_usage():
    """演示命令行接口使用"""
    print("\n" + "=" * 80)
    print("命令行接口使用演示")
    print("=" * 80)
    
    print("\n1. 基本处理命令:")
    print("   python -m semantic_coword_pipeline.pipeline input_data/ output_results/")
    
    print("\n2. 使用自定义配置:")
    print("   python -m semantic_coword_pipeline.pipeline input_data/ output_results/ --config config/custom.json")
    
    print("\n3. 创建默认配置文件:")
    print("   python -m semantic_coword_pipeline.pipeline --create-config")
    
    print("\n4. 验证配置文件:")
    print("   python -m semantic_coword_pipeline.pipeline --validate-config config/pipeline.json")
    
    print("\n5. 详细日志模式:")
    print("   python -m semantic_coword_pipeline.pipeline input_data/ output_results/ --verbose")
    
    print("\n6. 使用CLI接口:")
    print("   python -m semantic_coword_pipeline.cli process input_data/ output_results/")
    print("   python -m semantic_coword_pipeline.cli config create --output config/my_config.json")
    print("   python -m semantic_coword_pipeline.cli config show config/my_config.json")


def demonstrate_batch_processing():
    """演示批处理功能"""
    print("\n" + "=" * 80)
    print("批处理功能演示")
    print("=" * 80)
    
    print("\n如果你有现有的批处理脚本，可以这样使用:")
    print("   python test_batch_processing.py")
    
    print("\n或者使用主管线的批处理功能:")
    print("   python run_performance_benchmarks.py")


def show_project_structure():
    """显示项目结构"""
    print("\n" + "=" * 80)
    print("项目结构说明")
    print("=" * 80)
    
    print("\n核心组件:")
    print("├── semantic_coword_pipeline/")
    print("│   ├── core/                    # 核心模块")
    print("│   │   ├── config.py           # 配置管理")
    print("│   │   ├── logger.py           # 日志系统")
    print("│   │   ├── error_handler.py    # 错误处理")
    print("│   │   ├── data_models.py      # 数据模型")
    print("│   │   └── performance.py      # 性能监控")
    print("│   ├── processors/             # 处理器模块")
    print("│   │   ├── text_processor.py   # 文本处理")
    print("│   │   ├── phrase_extractor.py # 词组抽取")
    print("│   │   ├── dynamic_stopword_discoverer.py # 停词发现")
    print("│   │   ├── global_graph_builder.py # 总图构建")
    print("│   │   ├── state_subgraph_activator.py # 子图激活")
    print("│   │   ├── deterministic_layout_engine.py # 布局引擎")
    print("│   │   ├── batch_processor.py  # 批处理")
    print("│   │   ├── output_manager.py   # 输出管理")
    print("│   │   ├── document_generator.py # 文档生成")
    print("│   │   └── easygraph_interface.py # EasyGraph接口")
    print("│   ├── analyzers/              # 分析器模块")
    print("│   │   └── network_analyzer.py # 网络分析")
    print("│   ├── pipeline.py             # 主管线")
    print("│   └── cli.py                  # 命令行接口")
    print("├── tests/                      # 测试文件")
    print("├── docs/                       # 文档")
    print("├── config/                     # 配置文件")
    print("└── demo_*.py                   # 演示文件")


def main():
    """主函数"""
    print("语义增强共词网络分析管线 - 简化使用指南")
    print("=" * 80)
    
    # 显示项目结构
    show_project_structure()
    
    # 演示现有demo文件
    demonstrate_existing_demos()
    
    # 演示命令行接口
    demonstrate_cli_usage()
    
    # 演示批处理功能
    demonstrate_batch_processing()
    
    # 演示主管线使用
    success = demonstrate_main_pipeline()
    
    # 总结
    print("\n" + "=" * 80)
    print("使用指南总结")
    print("=" * 80)
    
    if success:
        print("✓ 主管线演示成功完成！")
    else:
        print("⚠ 主管线演示遇到问题，但这是正常的，因为某些组件可能需要额外配置")
    
    print("\n推荐的使用方式:")
    print("1. 首先运行基础演示: python demo.py")
    print("2. 然后尝试各个组件演示: python demo_*.py")
    print("3. 最后使用主管线处理真实数据")
    
    print("\n如果遇到问题:")
    print("1. 检查依赖是否安装完整")
    print("2. 查看日志文件了解详细错误信息")
    print("3. 参考docs/目录下的文档")
    print("4. 运行测试确保组件正常工作: pytest tests/")
    
    print("\n系统特性:")
    print("• 可复现性: 固定随机种子确保结果一致")
    print("• 多语言支持: 中英文文本处理")
    print("• 模块化设计: 各组件可独立使用")
    print("• 高性能: EasyGraph后端优化")
    print("• 完整追溯: 处理过程完全可追踪")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())