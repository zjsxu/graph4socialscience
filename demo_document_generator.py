#!/usr/bin/env python3
"""
文档生成器演示脚本

演示DocumentGenerator和TraceabilityManager的功能，包括：
- 实验追溯记录
- 技术选择记录
- 处理步骤追踪
- 对比分析生成
- 结构化文档生成
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'Easy-Graph'))

from semantic_coword_pipeline.processors.document_generator import (
    DocumentGenerator, 
    TraceabilityManager,
    ComparisonMetrics
)
from semantic_coword_pipeline.core.data_models import GlobalGraph
from semantic_coword_pipeline.core.logger import PipelineLogger


def create_mock_global_graph(node_count: int = 10, edge_count: int = 15) -> GlobalGraph:
    """创建模拟全局图用于演示"""
    vocabulary = {f"phrase_{i}": i for i in range(node_count)}
    reverse_vocabulary = {i: f"phrase_{i}" for i in range(node_count)}
    
    # 创建模拟的EasyGraph实例
    mock_graph = Mock()
    mock_graph.number_of_nodes.return_value = node_count
    mock_graph.number_of_edges.return_value = edge_count
    mock_graph.nodes.return_value = list(range(node_count))
    mock_graph.degree.side_effect = lambda n: 2 if n < edge_count else 0
    
    return GlobalGraph(
        easygraph_instance=mock_graph,
        vocabulary=vocabulary,
        reverse_vocabulary=reverse_vocabulary,
        cooccurrence_matrix=None,
        metadata={'created_at': '2024-01-01T00:00:00'}
    )


def demo_document_generator():
    """演示文档生成器功能"""
    print("=== 文档生成器演示 ===\n")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"临时输出目录: {temp_dir}")
    
    try:
        # 配置
        config = {
            'documentation': {
                'output_path': temp_dir,
                'template_path': 'templates/'
            },
            'text_processing': {
                'tokenizer': 'nltk',
                'language': 'english'
            },
            'phrase_extraction': {
                'ngram_size': 2,
                'min_frequency': 3
            },
            'stopword_discovery': {
                'tfidf_threshold': 0.1,
                'method': 'dynamic'
            },
            'graph_construction': {
                'strategy': 'global_first',
                'preserve_isolated': True
            },
            'layout': {
                'algorithm': 'force_directed',
                'seed': 42
            }
        }
        
        # 创建模拟日志记录器
        mock_logger = Mock(spec=PipelineLogger)
        
        # 初始化文档生成器
        generator = DocumentGenerator(config, mock_logger)
        
        print("1. 开始实验追溯")
        experiment_id = "demo_experiment_001"
        input_files = ["state_A_docs.json", "state_B_docs.json", "state_C_docs.json"]
        generator.start_experiment_trace(experiment_id, input_files)
        print(f"   实验ID: {experiment_id}")
        print(f"   输入文件: {input_files}")
        
        print("\n2. 记录技术选择")
        technical_choices = [
            {
                'component': 'text_processor',
                'choice': 'NLTK + jieba',
                'rationale': 'NLTK提供成熟的英文处理能力，jieba提供高质量的中文分词',
                'alternatives': ['spaCy', 'Stanford CoreNLP', '自定义分词器'],
                'parameters': config['text_processing']
            },
            {
                'component': 'phrase_extractor',
                'choice': '2-gram + 统计筛选',
                'rationale': '2-gram提供基础的词组单位，统计筛选确保质量',
                'alternatives': ['单词级别', '3-gram', '基于依存句法的短语'],
                'parameters': config['phrase_extraction']
            },
            {
                'component': 'graph_builder',
                'choice': '总图优先 + 州级激活',
                'rationale': '确保跨州对比的一致性和可比性',
                'alternatives': ['独立构建各州图', '层次化图构建', '增量图构建'],
                'parameters': config['graph_construction']
            }
        ]
        
        for choice in technical_choices:
            generator.record_technical_choice(**choice)
            print(f"   记录技术选择: {choice['component']} -> {choice['choice']}")
        
        print("\n3. 处理步骤追踪")
        processing_steps = [
            {
                'name': 'text_preprocessing',
                'input': 'Raw TOC documents',
                'output': 'Cleaned and tokenized text',
                'params': {'language_detection': True, 'normalization': True}
            },
            {
                'name': 'phrase_extraction',
                'input': 'Tokenized text',
                'output': 'Phrase candidates with statistical scores',
                'params': {'ngram_size': 2, 'statistical_filters': ['mutual_info', 't_score']}
            },
            {
                'name': 'stopword_discovery',
                'input': 'Phrase candidates',
                'output': 'Filtered phrases and dynamic stopword list',
                'params': {'tfidf_threshold': 0.1, 'frequency_threshold': 0.8}
            },
            {
                'name': 'global_graph_construction',
                'input': 'Filtered phrases',
                'output': 'Global cooccurrence graph',
                'params': {'window_type': 'segment', 'preserve_isolated': True}
            }
        ]
        
        for step in processing_steps:
            step_id = generator.start_processing_step(
                step['name'], step['input'], step['params']
            )
            print(f"   开始步骤: {step['name']}")
            
            # 模拟处理时间
            import time
            time.sleep(0.1)
            
            generator.end_processing_step(
                step['name'], step['output'], 'completed'
            )
            print(f"   完成步骤: {step['name']} -> {step['output']}")
        
        print("\n4. 生成对比分析")
        # 创建模拟图数据
        global_graph = create_mock_global_graph(100, 200)
        state_a_graph = create_mock_global_graph(80, 150)
        state_b_graph = create_mock_global_graph(90, 180)
        
        # 模拟EasyGraph函数
        import unittest.mock
        
        with unittest.mock.patch('easygraph.connected_components') as mock_components, \
             unittest.mock.patch('easygraph.average_clustering') as mock_clustering, \
             unittest.mock.patch('easygraph.degree_centrality') as mock_centrality:
            
            mock_components.return_value = [list(range(100))]
            mock_clustering.return_value = 0.35
            mock_centrality.return_value = {i: 0.1 * (100 - i) / 100 for i in range(100)}
            
            # 生成对比指标
            comparison_results = []
            
            scenarios = [
                ('全局共现网络', global_graph),
                ('州A子图', state_a_graph),
                ('州B子图', state_b_graph)
            ]
            
            for scenario_name, graph in scenarios:
                # 手动创建对比指标（因为模拟图的限制）
                metrics = ComparisonMetrics(
                    scenario_name=scenario_name,
                    node_count=graph.easygraph_instance.number_of_nodes(),
                    edge_count=graph.easygraph_instance.number_of_edges(),
                    density=0.04 if 'global' in scenario_name.lower() else 0.06,
                    isolated_nodes=5 if 'global' in scenario_name.lower() else 2,
                    connected_components=1,
                    clustering_coefficient=0.35,
                    average_path_length=3.2,
                    centrality_ranking=[
                        (f"phrase_{i}", 0.1 * (10 - i) / 10) for i in range(5)
                    ]
                )
                comparison_results.append(metrics)
                generator.add_comparison_result(metrics)
                print(f"   生成对比指标: {scenario_name} (节点: {metrics.node_count}, 边: {metrics.edge_count})")
        
        print("\n5. 结束实验追溯")
        output_files = [
            "global_graph.json",
            "state_subgraphs.json",
            "comparison_analysis.csv",
            "visualization_plots.png"
        ]
        trace = generator.end_experiment_trace(output_files)
        print(f"   输出文件: {output_files}")
        
        print("\n6. 生成结构化文档")
        doc_path = generator.generate_structured_document(trace)
        print(f"   实验文档: {doc_path}")
        
        # 显示文档内容片段
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            print("\n   文档内容预览:")
            for i, line in enumerate(lines[:20]):  # 显示前20行
                print(f"   {i+1:2d}: {line}")
            if len(lines) > 20:
                print(f"   ... (还有 {len(lines) - 20} 行)")
        
        print("\n7. 生成对比报告")
        report_path = generator.generate_comparison_report(comparison_results)
        print(f"   对比报告: {report_path}")
        
        # 显示报告内容片段
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            print("\n   报告内容预览:")
            for i, line in enumerate(lines[:15]):  # 显示前15行
                print(f"   {i+1:2d}: {line}")
            if len(lines) > 15:
                print(f"   ... (还有 {len(lines) - 15} 行)")
        
        print(f"\n8. 生成的文件")
        output_files = list(Path(temp_dir).glob("*"))
        for file_path in output_files:
            file_size = file_path.stat().st_size
            print(f"   {file_path.name}: {file_size} bytes")
        
    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\n清理临时目录: {temp_dir}")


def demo_traceability_manager():
    """演示追溯管理器功能"""
    print("\n=== 追溯管理器演示 ===\n")
    
    # 配置
    config = {}
    mock_logger = Mock(spec=PipelineLogger)
    
    # 初始化追溯管理器
    manager = TraceabilityManager(config, mock_logger)
    
    print("1. 记录数据转换过程")
    
    # 模拟数据转换链
    raw_text = ["This is a sample document.", "Another document here."]
    tokens = [["this", "is", "a", "sample", "document"], ["another", "document", "here"]]
    bigrams = [("this", "is"), ("is", "a"), ("a", "sample"), ("sample", "document"), ("another", "document"), ("document", "here")]
    filtered_bigrams = [("sample", "document"), ("another", "document")]
    
    transformations = [
        ("tokenization", raw_text, tokens, {"tokenizer": "nltk", "lowercase": True}),
        ("bigram_extraction", tokens, bigrams, {"ngram_size": 2}),
        ("statistical_filtering", bigrams, filtered_bigrams, {"min_frequency": 2, "tfidf_threshold": 0.1})
    ]
    
    for step_name, input_data, output_data, params in transformations:
        manager.record_data_transformation(step_name, input_data, output_data, params)
        print(f"   记录转换: {step_name}")
        print(f"     输入: {manager._summarize_data(input_data)}")
        print(f"     输出: {manager._summarize_data(output_data)}")
        print(f"     参数: {params}")
    
    print("\n2. 获取处理历史")
    history = manager.get_processing_history()
    print(f"   处理步骤数: {len(history)}")
    for i, record in enumerate(history, 1):
        print(f"   步骤 {i}: {record['step_name']} ({record['timestamp']})")
    
    print("\n3. 追溯数据血缘")
    final_data_id = id(filtered_bigrams)
    lineage = manager.trace_data_lineage(final_data_id)
    print(f"   最终数据的血缘链长度: {len(lineage)}")
    for i, lineage_info in enumerate(lineage, 1):
        print(f"   血缘 {i}: {lineage_info['transformation']} ({lineage_info['timestamp']})")


def main():
    """主函数"""
    print("语义增强共词网络分析 - 文档生成和追溯系统演示")
    print("=" * 60)
    
    try:
        demo_document_generator()
        demo_traceability_manager()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("\n主要功能:")
        print("✓ 实验追溯记录 - 完整记录实验过程")
        print("✓ 技术选择记录 - 记录关键技术决策和理由")
        print("✓ 处理步骤追踪 - 追踪每个处理步骤的输入输出")
        print("✓ 对比分析生成 - 自动生成网络结构对比指标")
        print("✓ 结构化文档生成 - 生成完整的实验报告")
        print("✓ 对比报告生成 - 生成详细的对比分析报告")
        print("✓ 数据血缘追溯 - 追踪数据转换的完整链路")
        
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()