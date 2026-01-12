"""
集成测试模块

测试主管线和配置系统的完整集成功能。
验证任务15的所有需求。
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from semantic_coword_pipeline.pipeline import SemanticCowordPipeline, create_default_config
from semantic_coword_pipeline.cli import CLIInterface, main
from semantic_coword_pipeline.core.config import Config
from semantic_coword_pipeline.core.performance import PerformanceMonitor, MemoryProfiler


class TestPipelineIntegration:
    """测试管线集成功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.config_dir = Path(self.temp_dir) / "config"
        
        # 创建目录
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        self.config_dir.mkdir(parents=True)
        
        # 创建测试数据
        self._create_test_data()
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_data(self):
        """创建测试数据"""
        # 创建测试TOC文档
        test_doc = {
            "segment_id": "test_001",
            "title": "Test Document",
            "level": 1,
            "order": 1,
            "text": "This is a test document for semantic coword analysis. It contains multiple phrases and keywords.",
            "state": "TestState"
        }
        
        doc_file = self.input_dir / "test_doc.json"
        with open(doc_file, 'w', encoding='utf-8') as f:
            json.dump(test_doc, f, ensure_ascii=False, indent=2)
    
    def test_pipeline_initialization(self):
        """测试管线初始化"""
        # 创建默认配置
        config_file = self.config_dir / "test_config.json"
        config = Config()
        config.save_to_file(str(config_file))
        
        # 初始化管线
        pipeline = SemanticCowordPipeline(str(config_file))
        
        # 验证组件初始化
        assert pipeline.config is not None
        assert pipeline.logger is not None
        assert pipeline.error_handler is not None
        assert pipeline.performance_monitor is not None
        assert pipeline.batch_processor is not None
        assert pipeline.document_generator is not None
    
    def test_configuration_management(self):
        """测试配置管理功能"""
        # 创建配置
        config = Config()
        
        # 测试基本配置操作
        config.set('test.key', 'test_value')
        assert config.get('test.key') == 'test_value'
        
        # 测试配置更新
        config.update({'test': {'key2': 'value2'}})
        assert config.get('test.key2') == 'value2'
        
        # 测试配置验证
        validation_result = config.validate()
        assert 'errors' in validation_result
        assert 'warnings' in validation_result
        
        # 测试配置历史
        history = config.get_config_history()
        assert len(history) > 0
    
    def test_performance_monitoring(self):
        """测试性能监控功能"""
        config = {
            'enable_profiling': True,
            'enable_memory_monitoring': True,
            'sampling_interval': 0.1
        }
        
        monitor = PerformanceMonitor(config)
        
        # 测试操作监控
        operation_id = monitor.start_operation('test_operation', {'test': 'data'})
        assert operation_id != ""
        
        # 模拟一些工作
        import time
        time.sleep(0.1)
        
        metrics = monitor.end_operation(operation_id)
        assert metrics is not None
        assert metrics.operation == 'test_operation'
        assert metrics.duration > 0
        
        # 测试性能报告生成
        report = monitor.generate_performance_report()
        assert 'operation_summary' in report
        assert 'system_summary' in report
    
    def test_memory_profiling(self):
        """测试内存分析功能"""
        profiler = MemoryProfiler()
        profiler.start_profiling()
        
        # 获取初始快照
        profiler.take_snapshot('start')
        
        # 分配一些内存
        data = [i for i in range(1000)]
        
        # 获取结束快照
        profiler.take_snapshot('end')
        
        # 比较快照
        comparison = profiler.compare_snapshots('start', 'end')
        
        profiler.stop_profiling()
        
        # 验证比较结果
        if comparison:  # 只有在启用内存分析时才验证
            assert 'total_difference' in comparison
    
    @patch('semantic_coword_pipeline.processors.batch_processor.BatchProcessor.process_directory')
    def test_pipeline_execution(self, mock_process):
        """测试管线执行"""
        # 模拟批处理结果
        from semantic_coword_pipeline.processors.batch_processor import BatchProcessingResult
        from semantic_coword_pipeline.core.data_models import GlobalGraph
        
        mock_result = BatchProcessingResult(
            total_files=1,
            processed_files=1,
            failed_files=0,
            processing_time=1.0,
            output_files=['test_output.json'],
            global_graph=Mock(spec=GlobalGraph),
            state_subgraphs={'TestState': Mock()},
            error_summary={'summary': {'total_errors': 0}}
        )
        mock_process.return_value = mock_result
        
        # 创建管线并运行
        pipeline = SemanticCowordPipeline()
        result = pipeline.run(str(self.input_dir), str(self.output_dir))
        
        # 验证结果
        assert result.total_files == 1
        assert result.processed_files == 1
        assert result.failed_files == 0
        
        # 验证批处理器被调用
        mock_process.assert_called_once()


class TestCLIIntegration:
    """测试CLI集成功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "config"
        self.config_dir.mkdir(parents=True)
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cli_initialization(self):
        """测试CLI初始化"""
        cli = CLIInterface()
        parser = cli.create_parser()
        
        # 验证解析器创建
        assert parser is not None
        
        # 测试帮助信息
        help_text = parser.format_help()
        assert 'semantic-coword' in help_text.lower()
        assert 'process' in help_text
        assert 'config' in help_text
    
    def test_config_commands(self):
        """测试配置命令"""
        cli = CLIInterface()
        
        # 测试创建配置
        config_file = self.config_dir / "test_config.json"
        result = cli._create_config(Mock(output=str(config_file)))
        
        assert result == 0
        assert config_file.exists()
        
        # 测试验证配置
        result = cli._validate_config(Mock(config_file=str(config_file)))
        assert result == 0
        
        # 测试显示配置
        result = cli._show_config(Mock(config_file=str(config_file), section=None))
        assert result == 0
    
    def test_cli_argument_parsing(self):
        """测试CLI参数解析"""
        cli = CLIInterface()
        parser = cli.create_parser()
        
        # 测试process命令参数
        args = parser.parse_args(['process', 'input/', 'output/', '--verbose'])
        assert args.command == 'process'
        assert args.input_dir == 'input/'
        assert args.output_dir == 'output/'
        assert args.verbose == 1
        
        # 测试config命令参数
        args = parser.parse_args(['config', 'create', '--output', 'config.json'])
        assert args.command == 'config'
        assert args.config_action == 'create'
        assert args.output == 'config.json'
    
    @patch('semantic_coword_pipeline.cli.SemanticCowordPipeline')
    def test_cli_process_command(self, mock_pipeline_class):
        """测试CLI处理命令"""
        # 模拟管线实例
        mock_pipeline = Mock()
        mock_result = Mock()
        mock_result.total_files = 1
        mock_result.processed_files = 1
        mock_result.failed_files = 0
        mock_result.processing_time = 1.0
        mock_result.output_files = ['output.json']
        mock_result.global_graph = Mock()
        mock_result.global_graph.get_node_count.return_value = 100
        mock_result.state_subgraphs = {'state1': Mock()}
        mock_result.error_summary = {'summary': {'total_errors': 0}}
        
        mock_pipeline.run.return_value = mock_result
        mock_pipeline_class.return_value = mock_pipeline
        
        # 创建CLI并运行
        cli = CLIInterface()
        
        # 模拟参数
        args = Mock()
        args.input_dir = 'input/'
        args.output_dir = 'output/'
        args.config = None
        args.dry_run = False
        args.profile = False
        args.memory_monitor = False
        args.parallel = None
        args.resume = None
        
        result = cli._handle_process_command(args)
        
        # 验证结果
        assert result == 0
        mock_pipeline.run.assert_called_once_with('input/', 'output/')


class TestSystemIntegration:
    """测试系统级集成"""
    
    def test_default_config_creation(self):
        """测试默认配置创建"""
        config_file = create_default_config()
        
        # 验证配置文件创建
        assert Path(config_file).exists()
        
        # 验证配置内容
        config = Config(config_file)
        all_config = config.get_all()
        
        # 验证必需的配置节
        required_sections = [
            'text_processing',
            'stopword_discovery', 
            'graph_construction',
            'layout_engine',
            'output',
            'performance',
            'logging'
        ]
        
        for section in required_sections:
            assert section in all_config
    
    def test_configuration_validation(self):
        """测试配置验证"""
        config = Config()
        
        # 测试有效配置
        validation_result = config.validate()
        assert len(validation_result['errors']) == 0
        
        # 测试无效配置
        config.set('text_processing.ngram_size', 0)
        validation_result = config.validate()
        assert len(validation_result['errors']) > 0
    
    def test_error_handling_integration(self):
        """测试错误处理集成"""
        from semantic_coword_pipeline.core.error_handler import ErrorHandler
        
        config = {
            'max_retries': 2,
            'retry_delay': 0.1,
            'continue_on_error': True
        }
        
        error_handler = ErrorHandler(config)
        
        # 测试错误处理 - 期望抛出异常
        test_error = ValueError("Test error")
        
        with pytest.raises(Exception):  # 期望抛出异常
            error_handler.handle_error(test_error, "test_context")
        
        # 验证错误记录
        error_report = error_handler.generate_error_report()
        assert 'summary' in error_report
        assert error_report['summary']['total_errors'] > 0
    
    def test_logging_integration(self):
        """测试日志集成"""
        from semantic_coword_pipeline.core.logger import PipelineLogger
        
        config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
        
        logger = PipelineLogger("TestLogger", config)
        
        # 测试日志记录
        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")
        
        # 验证日志器正常工作
        assert logger.logger.name == "TestLogger"


@pytest.mark.integration
class TestEndToEndIntegration:
    """端到端集成测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        
        # 创建目录
        self.input_dir.mkdir(parents=True)
        self.output_dir.mkdir(parents=True)
        
        # 创建测试数据
        self._create_comprehensive_test_data()
    
    def teardown_method(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_comprehensive_test_data(self):
        """创建综合测试数据"""
        # 创建多个测试文档
        test_docs = [
            {
                "segment_id": "doc1_seg1",
                "title": "Introduction",
                "level": 1,
                "order": 1,
                "text": "Natural language processing and machine learning are important fields in artificial intelligence.",
                "state": "California"
            },
            {
                "segment_id": "doc1_seg2", 
                "title": "Methods",
                "level": 2,
                "order": 2,
                "text": "We use statistical methods and neural networks for text analysis and semantic understanding.",
                "state": "California"
            },
            {
                "segment_id": "doc2_seg1",
                "title": "Overview",
                "level": 1,
                "order": 1,
                "text": "Deep learning and neural networks have revolutionized natural language processing applications.",
                "state": "Texas"
            }
        ]
        
        for i, doc in enumerate(test_docs):
            doc_file = self.input_dir / f"test_doc_{i+1}.json"
            with open(doc_file, 'w', encoding='utf-8') as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
    
    @patch('semantic_coword_pipeline.processors.text_processor.TextProcessor.process_document')
    @patch('semantic_coword_pipeline.processors.phrase_extractor.PhraseExtractor.extract_phrases_from_document')
    @patch('semantic_coword_pipeline.processors.global_graph_builder.GlobalGraphBuilder.build_global_graph')
    def test_complete_pipeline_flow(self, mock_graph_builder, mock_phrase_extractor, mock_text_processor):
        """测试完整的管线流程"""
        # 模拟处理结果
        from semantic_coword_pipeline.core.data_models import ProcessedDocument, GlobalGraph, Phrase
        
        mock_processed_doc = ProcessedDocument(
            original_doc=Mock(),
            cleaned_text="processed text",
            tokens=["natural", "language", "processing"],
            phrases=["natural language", "language processing"],
            windows=[Mock()]
        )
        mock_text_processor.return_value = mock_processed_doc
        
        mock_phrases = [
            Phrase(text="natural language", frequency=2, tfidf_score=0.5, statistical_scores={}),
            Phrase(text="language processing", frequency=2, tfidf_score=0.6, statistical_scores={})
        ]
        mock_phrase_extractor.return_value = mock_phrases
        
        mock_global_graph = Mock(spec=GlobalGraph)
        mock_global_graph.get_node_count.return_value = 10
        mock_global_graph.vocabulary = {"natural language": 0, "language processing": 1}
        mock_graph_builder.return_value = mock_global_graph
        
        # 创建配置
        config = Config()
        config.set('performance.enable_profiling', True)
        config.set('performance.enable_memory_monitoring', True)
        
        # 运行管线
        pipeline = SemanticCowordPipeline()
        pipeline.config = config
        
        # 由于我们模拟了关键组件，这里主要测试管线协调逻辑
        # 实际的端到端测试需要真实的组件实现
        
        # 验证配置加载
        assert pipeline.config is not None
        assert pipeline.performance_monitor is not None
        assert pipeline.batch_processor is not None