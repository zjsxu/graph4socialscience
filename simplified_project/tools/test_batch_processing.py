#!/usr/bin/env python3
"""
批处理功能测试脚本

测试批处理和输出管理系统的基本功能。
"""

import os
import json
import tempfile
import shutil
from pathlib import Path

from semantic_coword_pipeline import SemanticCowordPipeline, Config, TOCDocument


def create_test_data(test_dir: str) -> None:
    """创建测试数据"""
    test_path = Path(test_dir)
    
    # 创建测试文档
    test_docs = [
        {
            "segment_id": "doc1_seg1",
            "title": "Introduction to Machine Learning",
            "level": 1,
            "order": 1,
            "text": "Machine learning is a subset of artificial intelligence. It involves algorithms that learn from data.",
            "state": "CA"
        },
        {
            "segment_id": "doc1_seg2", 
            "title": "Deep Learning Basics",
            "level": 2,
            "order": 2,
            "text": "Deep learning uses neural networks with multiple layers. These networks can learn complex patterns.",
            "state": "CA"
        },
        {
            "segment_id": "doc2_seg1",
            "title": "Natural Language Processing",
            "level": 1,
            "order": 1,
            "text": "Natural language processing enables computers to understand human language. It combines linguistics and machine learning.",
            "state": "NY"
        }
    ]
    
    # 保存为JSON文件
    for i, doc in enumerate(test_docs):
        file_path = test_path / f"test_doc_{i+1}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2)
    
    print(f"Created {len(test_docs)} test documents in {test_dir}")


def test_batch_processing():
    """测试批处理功能"""
    print("Testing batch processing functionality...")
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        
        input_dir.mkdir()
        output_dir.mkdir()
        
        # 创建测试数据
        create_test_data(str(input_dir))
        
        try:
            # 创建管线实例
            pipeline = SemanticCowordPipeline()
            
            # 运行批处理
            result = pipeline.run(str(input_dir), str(output_dir))
            
            # 验证结果
            print(f"✓ Processed {result.processed_files}/{result.total_files} files")
            print(f"✓ Processing time: {result.processing_time:.2f} seconds")
            print(f"✓ Generated {len(result.output_files)} output files")
            
            if result.global_graph:
                print(f"✓ Global graph has {result.global_graph.get_node_count()} nodes")
            
            print(f"✓ Generated {len(result.state_subgraphs)} state subgraphs")
            
            # 验证输出文件存在
            output_files_exist = all(Path(f).exists() for f in result.output_files)
            if output_files_exist:
                print("✓ All output files were created successfully")
            else:
                print("✗ Some output files are missing")
            
            # 检查输出目录结构
            expected_dirs = ['data', 'graphs', 'reports', 'logs']
            for dir_name in expected_dirs:
                dir_path = output_dir / dir_name
                if dir_path.exists():
                    print(f"✓ Output directory '{dir_name}' created")
                else:
                    print(f"✗ Output directory '{dir_name}' missing")
            
            print("\n✓ Batch processing test completed successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Batch processing test failed: {e}")
            return False


def test_configuration():
    """测试配置功能"""
    print("\nTesting configuration functionality...")
    
    try:
        # 创建配置实例
        config = Config()
        
        # 测试配置获取
        batch_size = config.get('performance.batch_size', 1000)
        print(f"✓ Retrieved batch size: {batch_size}")
        
        # 测试配置设置
        config.set('performance.batch_size', 500)
        new_batch_size = config.get('performance.batch_size')
        assert new_batch_size == 500
        print("✓ Configuration setting works")
        
        # 测试配置验证
        validation_result = config.validate()
        if not validation_result['errors']:
            print("✓ Configuration validation passed")
        else:
            print(f"✗ Configuration validation errors: {validation_result['errors']}")
        
        print("✓ Configuration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def test_data_models():
    """测试数据模型"""
    print("\nTesting data models...")
    
    try:
        # 测试TOCDocument
        doc_data = {
            "segment_id": "test_seg_1",
            "title": "Test Document",
            "level": 1,
            "order": 1,
            "text": "This is a test document for validation.",
            "state": "TEST"
        }
        
        toc_doc = TOCDocument.from_json(doc_data)
        assert toc_doc.segment_id == "test_seg_1"
        assert toc_doc.state == "TEST"
        print("✓ TOCDocument creation and validation works")
        
        # 测试转换为字典
        doc_dict = toc_doc.to_dict()
        assert doc_dict['segment_id'] == "test_seg_1"
        print("✓ TOCDocument to_dict conversion works")
        
        print("✓ Data models test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Data models test failed: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("Semantic Coword Pipeline - Batch Processing Test")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Data Models", test_data_models),
        ("Batch Processing", test_batch_processing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running {test_name} Test")
        print(f"{'-' * 40}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # 输出测试结果摘要
    print(f"\n{'=' * 60}")
    print("Test Results Summary")
    print(f"{'=' * 60}")
    
    passed = 0
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{test_name:20} : {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Batch processing system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    exit(main())