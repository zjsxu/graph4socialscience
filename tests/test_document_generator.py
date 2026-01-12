"""
文档生成器测试

测试DocumentGenerator和TraceabilityManager的各种功能，包括结构化文档生成、
技术选择记录、处理过程追溯和对比报告生成。
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Easy-Graph'))

from hypothesis import given, strategies as st, settings, HealthCheck

try:
    import easygraph as eg
except ImportError:
    eg = None

from semantic_coword_pipeline.processors.document_generator import (
    DocumentGenerator,
    TraceabilityManager,
    TechnicalChoice,
    ProcessingStep,
    ComparisonMetrics,
    ExperimentTrace
)
from semantic_coword_pipeline.core.data_models import GlobalGraph, StateSubgraph
from semantic_coword_pipeline.core.logger import PipelineLogger


class TestDocumentGenerator:
    """DocumentGenerator单元测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'documentation': {
                'output_path': self.temp_dir,
                'template_path': 'templates/'
            }
        }
        
        # 创建模拟日志记录器
        self.mock_logger = Mock(spec=PipelineLogger)
        
        self.generator = DocumentGenerator(self.config, self.mock_logger)
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_global_graph(self, node_count: int = 5, edge_count: int = 4) -> GlobalGraph:
        """创建模拟全局图"""
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
            metadata={'created_at': datetime.now().isoformat()}
        )
    
    def test_start_experiment_trace(self):
        """测试开始实验追溯"""
        experiment_id = "test_exp_001"
        input_files = ["file1.json", "file2.json"]
        
        self.generator.start_experiment_trace(experiment_id, input_files)
        
        assert self.generator.current_trace is not None
        assert self.generator.current_trace.experiment_id == experiment_id
        assert self.generator.current_trace.input_files == input_files
        assert self.generator.current_trace.start_time != ""
        assert len(self.generator.technical_choices) == 0
        assert len(self.generator.processing_steps) == 0
    
    def test_record_technical_choice(self):
        """测试记录技术选择"""
        self.generator.start_experiment_trace("test_exp", ["file1.json"])
        
        component = "text_processor"
        choice = "nltk_tokenizer"
        rationale = "Better performance for English text"
        alternatives = ["spacy_tokenizer", "custom_tokenizer"]
        parameters = {"language": "english", "preserve_case": False}
        
        self.generator.record_technical_choice(
            component, choice, rationale, alternatives, parameters
        )
        
        assert len(self.generator.technical_choices) == 1
        tech_choice = self.generator.technical_choices[0]
        assert tech_choice.component == component
        assert tech_choice.choice == choice
        assert tech_choice.rationale == rationale
        assert tech_choice.alternatives == alternatives
        assert tech_choice.parameters == parameters
    
    def test_processing_step_lifecycle(self):
        """测试处理步骤生命周期"""
        self.generator.start_experiment_trace("test_exp", ["file1.json"])
        
        step_name = "text_processing"
        input_desc = "Raw TOC documents"
        parameters = {"language": "english"}
        
        # 开始步骤
        step_id = self.generator.start_processing_step(step_name, input_desc, parameters)
        
        assert len(self.generator.processing_steps) == 1
        step = self.generator.processing_steps[0]
        assert step.step_name == step_name
        assert step.input_description == input_desc
        assert step.parameters == parameters
        assert step.status == "running"
        
        # 结束步骤
        output_desc = "Processed documents with tokens"
        self.generator.end_processing_step(step_name, output_desc, "completed")
        
        assert step.output_description == output_desc
        assert step.status == "completed"
        assert step.duration_seconds >= 0
    
    def test_generate_comparison_metrics(self):
        """测试生成对比指标"""
        mock_graph = self.create_mock_global_graph(10, 15)
        
        with patch('easygraph.connected_components') as mock_components, \
             patch('easygraph.average_clustering') as mock_clustering, \
             patch('easygraph.degree_centrality') as mock_centrality:
            
            mock_components.return_value = [list(range(10))]  # 一个连通分量
            mock_clustering.return_value = 0.3
            mock_centrality.return_value = {i: 0.1 * (10 - i) for i in range(10)}
            
            metrics = self.generator.generate_comparison_metrics("test_scenario", mock_graph)
            
            assert metrics.scenario_name == "test_scenario"
            assert metrics.node_count == 10
            assert metrics.edge_count == 15
            assert metrics.density > 0
            assert metrics.connected_components == 1
            assert metrics.clustering_coefficient == 0.3
            assert len(metrics.centrality_ranking) > 0
    
    def test_end_experiment_trace(self):
        """测试结束实验追溯"""
        self.generator.start_experiment_trace("test_exp", ["file1.json"])
        
        # 添加一些记录
        self.generator.record_technical_choice(
            "tokenizer", "nltk", "Standard choice", ["spacy"], {}
        )
        
        step_id = self.generator.start_processing_step("test_step", "input", {})
        self.generator.end_processing_step("test_step", "output", "completed")
        
        output_files = ["output1.json", "output2.csv"]
        trace = self.generator.end_experiment_trace(output_files)
        
        assert trace.end_time != ""
        assert trace.output_files == output_files
        assert len(trace.technical_choices) == 1
        assert len(trace.processing_steps) == 1
        
        # 检查是否保存了追溯文件
        trace_files = list(Path(self.temp_dir).glob("experiment_trace_*.json"))
        assert len(trace_files) == 1
    
    def test_generate_structured_document(self):
        """测试生成结构化文档"""
        # 创建完整的实验追溯
        self.generator.start_experiment_trace("test_exp", ["file1.json"])
        
        self.generator.record_technical_choice(
            "tokenizer", "nltk", "Standard choice", ["spacy"], {"language": "english"}
        )
        
        step_id = self.generator.start_processing_step("test_step", "input", {})
        self.generator.end_processing_step("test_step", "output", "completed")
        
        # 添加对比结果
        mock_graph = self.create_mock_global_graph()
        with patch('easygraph.connected_components') as mock_components, \
             patch('easygraph.average_clustering') as mock_clustering, \
             patch('easygraph.degree_centrality') as mock_centrality:
            
            mock_components.return_value = [list(range(5))]
            mock_clustering.return_value = 0.3
            mock_centrality.return_value = {i: 0.1 * (5 - i) for i in range(5)}
            
            metrics = self.generator.generate_comparison_metrics("test_scenario", mock_graph)
            self.generator.add_comparison_result(metrics)
        
        trace = self.generator.end_experiment_trace(["output1.json"])
        
        # 生成文档
        doc_path = self.generator.generate_structured_document(trace)
        
        assert Path(doc_path).exists()
        
        # 检查文档内容
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "语义增强共词网络分析实验报告" in content
        assert "test_exp" in content
        assert "nltk" in content
        assert "test_step" in content
        assert "test_scenario" in content
    
    def test_generate_comparison_report(self):
        """测试生成对比报告"""
        # 创建多个对比指标
        metrics1 = ComparisonMetrics(
            scenario_name="单词节点",
            node_count=100,
            edge_count=200,
            density=0.04,
            isolated_nodes=10,
            connected_components=5,
            clustering_coefficient=0.3,
            average_path_length=3.5,
            centrality_ranking=[("word1", 0.1), ("word2", 0.08)]
        )
        
        metrics2 = ComparisonMetrics(
            scenario_name="词组节点",
            node_count=80,
            edge_count=180,
            density=0.057,
            isolated_nodes=5,
            connected_components=3,
            clustering_coefficient=0.4,
            average_path_length=3.2,
            centrality_ranking=[("phrase1", 0.12), ("phrase2", 0.09)]
        )
        
        comparison_results = [metrics1, metrics2]
        
        report_path = self.generator.generate_comparison_report(comparison_results)
        
        assert Path(report_path).exists()
        
        # 检查报告内容
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert "网络结构对比分析报告" in content
        assert "单词节点" in content
        assert "词组节点" in content
        assert "100" in content  # 节点数
        assert "0.04" in content  # 密度


class TestTraceabilityManager:
    """TraceabilityManager单元测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.config = {}
        self.mock_logger = Mock(spec=PipelineLogger)
        self.manager = TraceabilityManager(self.config, self.mock_logger)
    
    def test_record_data_transformation(self):
        """测试记录数据转换"""
        input_data = ["word1", "word2", "word3"]
        output_data = [("word1", "word2"), ("word2", "word3")]
        parameters = {"ngram_size": 2}
        
        self.manager.record_data_transformation(
            "bigram_extraction", input_data, output_data, parameters
        )
        
        assert len(self.manager.processing_history) == 1
        
        record = self.manager.processing_history[0]
        assert record['step_name'] == "bigram_extraction"
        assert record['parameters'] == parameters
        assert record['input_summary']['size'] == 3
        assert record['output_summary']['size'] == 2
        
        # 检查数据血缘
        output_id = id(output_data)
        assert output_id in self.manager.data_lineage
        assert self.manager.data_lineage[output_id]['transformation'] == "bigram_extraction"
    
    def test_data_summarization(self):
        """测试数据摘要生成"""
        # 测试列表摘要
        list_data = [1, 2, 3, 4, 5]
        summary = self.manager._summarize_data(list_data)
        assert summary['type'] == 'list'
        assert summary['size'] == 5
        assert summary['properties']['first_item_type'] == 'int'
        
        # 测试字典摘要
        dict_data = {'a': 1, 'b': 2, 'c': 3}
        summary = self.manager._summarize_data(dict_data)
        assert summary['type'] == 'dict'
        assert summary['size'] == 3
        assert 'keys' in summary['properties']
    
    def test_get_processing_history(self):
        """测试获取处理历史"""
        # 记录几个转换
        self.manager.record_data_transformation("step1", [1, 2], [3, 4], {})
        self.manager.record_data_transformation("step2", [3, 4], [5, 6], {})
        
        history = self.manager.get_processing_history()
        
        assert len(history) == 2
        assert history[0]['step_name'] == "step1"
        assert history[1]['step_name'] == "step2"
        
        # 确保返回的是副本
        history.append({'test': 'data'})
        assert len(self.manager.processing_history) == 2
    
    def test_trace_data_lineage(self):
        """测试数据血缘追溯"""
        data1 = [1, 2, 3]
        data2 = [4, 5, 6]
        data3 = [7, 8, 9]
        
        # 建立转换链
        self.manager.record_data_transformation("step1", data1, data2, {})
        self.manager.record_data_transformation("step2", data2, data3, {})
        
        # 追溯最终数据的血缘
        lineage = self.manager.trace_data_lineage(id(data3))
        
        assert len(lineage) == 2
        assert lineage[0]['transformation'] == "step2"
        assert lineage[1]['transformation'] == "step1"


# 属性测试
class TestDocumentGeneratorProperties:
    """DocumentGenerator属性测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'documentation': {
                'output_path': self.temp_dir
            }
        }
        self.mock_logger = Mock(spec=PipelineLogger)
        self.generator = DocumentGenerator(self.config, self.mock_logger)
    
    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @given(
        experiment_id=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='._-'), min_size=1, max_size=50),
        input_files=st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=10)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_experiment_trace_completeness(self, experiment_id, input_files):
        """
        属性测试：实验追溯完整性
        
        **验证：需求 10.5**
        
        对于任何实验ID和输入文件列表，开始和结束实验追溯应该生成完整的追溯记录。
        """
        # 开始追溯
        self.generator.start_experiment_trace(experiment_id, input_files)
        
        # 验证追溯已开始
        assert self.generator.current_trace is not None
        assert self.generator.current_trace.experiment_id == experiment_id
        assert self.generator.current_trace.input_files == input_files
        
        # 结束追溯
        output_files = ["output1.json", "output2.csv"]
        trace = self.generator.end_experiment_trace(output_files)
        
        # 验证追溯完整性
        assert trace.experiment_id == experiment_id
        assert trace.input_files == input_files
        assert trace.output_files == output_files
        assert trace.start_time != ""
        assert trace.end_time != ""
        assert trace.start_time <= trace.end_time
    
    @given(
        component=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='._-'), min_size=1, max_size=50),
        choice=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), whitelist_characters='._- '), min_size=1, max_size=50),
        rationale=st.text(alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs'), whitelist_characters='._-,'), min_size=1, max_size=200)
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_technical_choice_recording(self, component, choice, rationale):
        """
        属性测试：技术选择记录完整性
        
        **验证：需求 10.3**
        
        对于任何技术选择，记录应该包含所有必要信息且可追溯。
        """
        self.generator.start_experiment_trace("test_exp", ["file1.json"])
        
        alternatives = ["alt1", "alt2"]
        parameters = {"param1": "value1"}
        
        self.generator.record_technical_choice(
            component, choice, rationale, alternatives, parameters
        )
        
        # 验证记录完整性
        assert len(self.generator.technical_choices) == 1
        tech_choice = self.generator.technical_choices[0]
        
        assert tech_choice.component == component
        assert tech_choice.choice == choice
        assert tech_choice.rationale == rationale
        assert tech_choice.alternatives == alternatives
        assert tech_choice.parameters == parameters
        assert tech_choice.timestamp != ""


if __name__ == "__main__":
    pytest.main([__file__])