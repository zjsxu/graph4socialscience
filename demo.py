#!/usr/bin/env python3
"""
语义增强共词网络分析管线演示脚本

展示基础架构和核心数据模型的使用方法。
"""

import json
from semantic_coword_pipeline.core.config import Config
from semantic_coword_pipeline.core.logger import setup_logger
from semantic_coword_pipeline.core.error_handler import ErrorHandler
from semantic_coword_pipeline.core.data_models import (
    TOCDocument, ProcessedDocument, Window, Phrase, GlobalGraph, StateSubgraph
)


def main():
    """主演示函数"""
    print("=== 语义增强共词网络分析管线演示 ===\n")
    
    # 1. 配置管理演示
    print("1. 配置管理系统演示")
    config = Config()
    print(f"   默认N-gram大小: {config.get('text_processing.ngram_size')}")
    print(f"   随机种子: {config.get('layout_engine.random_seed')}")
    
    # 修改配置
    config.set('text_processing.ngram_size', 3)
    print(f"   修改后N-gram大小: {config.get('text_processing.ngram_size')}")
    print()
    
    # 2. 日志系统演示
    print("2. 日志系统演示")
    logger = setup_logger('demo', config.get_section('logging'))
    logger.info("演示系统启动")
    logger.log_processing_step(
        'demo_step', 
        {'input_docs': 5}, 
        {'processed_docs': 5}, 
        1.2
    )
    print("   日志记录完成")
    print()
    
    # 3. 错误处理演示
    print("3. 错误处理系统演示")
    error_handler = ErrorHandler()
    
    try:
        # 模拟一个错误
        raise ValueError("演示错误")
    except Exception as e:
        try:
            error_handler.handle_error(e, "demo_context")
        except Exception:
            print("   错误已被捕获和处理")
    
    error_report = error_handler.generate_error_report()
    print(f"   错误报告生成: {error_report['summary']['total_errors']} 个错误")
    print()
    
    # 4. 数据模型演示
    print("4. 核心数据模型演示")
    
    # TOC文档
    toc_data = {
        "segment_id": "demo_001",
        "title": "演示文档",
        "level": 1,
        "order": 1,
        "text": "这是一个演示文档，用于展示系统功能。包含多个句子和词组。",
        "state": "CA",
        "language": "zh"
    }
    
    toc_doc = TOCDocument.from_json(toc_data)
    print(f"   TOC文档创建: {toc_doc.title} (ID: {toc_doc.segment_id})")
    
    # 词组
    phrases = [
        Phrase("演示文档", frequency=1, tfidf_score=0.8),
        Phrase("系统功能", frequency=1, tfidf_score=0.6),
        Phrase("多个句子", frequency=1, tfidf_score=0.4)
    ]
    print(f"   创建词组: {len(phrases)} 个")
    
    # 窗口
    window = Window(
        window_id="demo_win_001",
        phrases=[p.text for p in phrases],
        source_doc=toc_doc.segment_id,
        state=toc_doc.state,
        segment_id=toc_doc.segment_id
    )
    print(f"   创建窗口: {len(window)} 个词组")
    
    # 处理后文档
    processed_doc = ProcessedDocument(
        original_doc=toc_doc,
        cleaned_text="演示文档 系统功能 多个句子",
        tokens=["演示", "文档", "系统", "功能", "多个", "句子"],
        phrases=[p.text for p in phrases],
        windows=[window]
    )
    print(f"   处理后文档: {processed_doc.get_phrase_count()} 个词组, {processed_doc.get_window_count()} 个窗口")
    
    # 全局图
    vocab = {phrase.text: i for i, phrase in enumerate(phrases)}
    reverse_vocab = {i: phrase.text for i, phrase in enumerate(phrases)}
    
    global_graph = GlobalGraph(
        vocabulary=vocab,
        reverse_vocabulary=reverse_vocab
    )
    print(f"   全局图创建: {global_graph.get_node_count()} 个节点")
    
    # 州级子图
    import numpy as np
    activation_mask = np.array([True, True, False])  # 激活前两个节点
    
    state_subgraph = StateSubgraph(
        state_name="CA",
        parent_global_graph=global_graph,
        activation_mask=activation_mask
    )
    
    active_nodes = state_subgraph.get_active_nodes()
    print(f"   州级子图创建: {len(active_nodes)} 个激活节点")
    print()
    
    # 5. 系统集成演示
    print("5. 系统集成演示")
    
    # 验证数据一致性
    assert global_graph.has_phrase("演示文档")
    assert state_subgraph.is_node_active(0)
    assert not state_subgraph.is_node_active(2)
    
    print("   ✓ 数据模型一致性验证通过")
    print("   ✓ 配置管理功能正常")
    print("   ✓ 日志系统工作正常")
    print("   ✓ 错误处理机制有效")
    print()
    
    print("=== 演示完成 ===")
    print("基础架构和核心数据模型已成功建立！")
    print("系统已准备好进行下一步的功能开发。")


if __name__ == "__main__":
    main()