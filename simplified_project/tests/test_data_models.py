"""
数据模型测试

测试核心数据模型的功能和正确性。
"""

import pytest
import json
import numpy as np
from hypothesis import given, strategies as st

from semantic_coword_pipeline.core.data_models import (
    TOCDocument, ProcessedDocument, Window, Phrase, GlobalGraph, StateSubgraph,
    validate_toc_json, create_phrase_mapping
)
from semantic_coword_pipeline.core.error_handler import InputValidationError


class TestTOCDocument:
    """TOC文档测试"""
    
    def test_create_toc_document(self, sample_toc_document):
        """测试创建TOC文档"""
        assert sample_toc_document.segment_id == "seg_001"
        assert sample_toc_document.title == "Introduction"
        assert sample_toc_document.level == 1
        assert sample_toc_document.order == 1
        assert "sample text" in sample_toc_document.text
        assert sample_toc_document.state == "CA"
        assert sample_toc_document.language == "en"
    
    def test_from_json_valid(self, sample_toc_json):
        """测试从有效JSON创建文档"""
        doc = TOCDocument.from_json(sample_toc_json)
        assert doc.segment_id == sample_toc_json["segment_id"]
        assert doc.title == sample_toc_json["title"]
        assert doc.level == sample_toc_json["level"]
        assert doc.order == sample_toc_json["order"]
        assert doc.text == sample_toc_json["text"]
    
    def test_from_json_invalid(self, invalid_toc_json):
        """测试从无效JSON创建文档"""
        with pytest.raises(ValueError, match="Missing required field"):
            TOCDocument.from_json(invalid_toc_json)
    
    def test_to_dict(self, sample_toc_document):
        """测试转换为字典"""
        doc_dict = sample_toc_document.to_dict()
        assert doc_dict["segment_id"] == sample_toc_document.segment_id
        assert doc_dict["title"] == sample_toc_document.title
        assert doc_dict["level"] == sample_toc_document.level
        assert doc_dict["order"] == sample_toc_document.order
        assert doc_dict["text"] == sample_toc_document.text


class TestWindow:
    """窗口测试"""
    
    def test_create_window(self, sample_window):
        """测试创建窗口"""
        assert sample_window.window_id == "win_001"
        assert len(sample_window.phrases) == 3
        assert sample_window.source_doc == "seg_001"
        assert sample_window.state == "CA"
    
    def test_window_length(self, sample_window):
        """测试窗口长度"""
        assert len(sample_window) == 3
    
    def test_empty_window(self):
        """测试空窗口"""
        empty_window = Window(
            window_id="empty",
            phrases=[],
            source_doc="doc",
            state="CA",
            segment_id="seg"
        )
        assert empty_window.is_empty()
        assert len(empty_window) == 0


class TestPhrase:
    """词组测试"""
    
    def test_create_phrase(self):
        """测试创建词组"""
        phrase = Phrase("test phrase", frequency=5, tfidf_score=0.8)
        assert phrase.text == "test phrase"
        assert phrase.frequency == 5
        assert phrase.tfidf_score == 0.8
        assert not phrase.is_stopword
    
    def test_statistical_scores(self):
        """测试统计分数"""
        phrase = Phrase("test phrase")
        phrase.add_statistical_score("mutual_info", 0.5)
        phrase.add_statistical_score("t_score", 2.3)
        
        assert phrase.get_statistical_score("mutual_info") == 0.5
        assert phrase.get_statistical_score("t_score") == 2.3
        assert phrase.get_statistical_score("nonexistent") is None


class TestProcessedDocument:
    """处理后文档测试"""
    
    def test_create_processed_document(self, sample_processed_document):
        """测试创建处理后文档"""
        assert sample_processed_document.get_phrase_count() == 3
        assert sample_processed_document.get_window_count() == 1
        assert "processed_at" in sample_processed_document.processing_metadata
    
    def test_metadata_initialization(self, sample_toc_document):
        """测试元数据初始化"""
        processed_doc = ProcessedDocument(
            original_doc=sample_toc_document,
            cleaned_text="test",
            tokens=["test"],
            phrases=["test"],
            windows=[]
        )
        
        metadata = processed_doc.processing_metadata
        assert metadata["token_count"] == 1
        assert metadata["phrase_count"] == 1
        assert metadata["window_count"] == 0


class TestGlobalGraph:
    """全局图测试"""
    
    def test_create_global_graph(self):
        """测试创建全局图"""
        vocab = {"phrase1": 0, "phrase2": 1}
        reverse_vocab = {0: "phrase1", 1: "phrase2"}
        
        graph = GlobalGraph(
            vocabulary=vocab,
            reverse_vocabulary=reverse_vocab
        )
        
        assert graph.get_node_count() == 2
        assert graph.get_node_id("phrase1") == 0
        assert graph.get_phrase(0) == "phrase1"
        assert graph.has_phrase("phrase1")
        assert not graph.has_phrase("nonexistent")
    
    def test_add_phrase(self):
        """测试添加词组"""
        graph = GlobalGraph(vocabulary={}, reverse_vocabulary={})
        
        node_id = graph.add_phrase("new phrase")
        assert node_id == 0
        assert graph.has_phrase("new phrase")
        assert graph.get_phrase(node_id) == "new phrase"
    
    def test_vocabulary_consistency_error(self):
        """测试词表一致性错误"""
        with pytest.raises(ValueError, match="Vocabulary and reverse vocabulary size mismatch"):
            GlobalGraph(
                vocabulary={"phrase1": 0},
                reverse_vocabulary={0: "phrase1", 1: "phrase2"}  # 不一致
            )


class TestStateSubgraph:
    """州级子图测试"""
    
    def test_create_state_subgraph(self):
        """测试创建州级子图"""
        parent_graph = GlobalGraph(
            vocabulary={"phrase1": 0, "phrase2": 1},
            reverse_vocabulary={0: "phrase1", 1: "phrase2"}
        )
        
        subgraph = StateSubgraph(
            state_name="CA",
            parent_global_graph=parent_graph
        )
        
        assert subgraph.state_name == "CA"
        assert subgraph.metadata["state_name"] == "CA"
        assert subgraph.metadata["parent_graph_nodes"] == 2
    
    def test_activation_mask(self):
        """测试激活掩码"""
        parent_graph = GlobalGraph(
            vocabulary={"phrase1": 0, "phrase2": 1},
            reverse_vocabulary={0: "phrase1", 1: "phrase2"}
        )
        
        subgraph = StateSubgraph(
            state_name="CA",
            parent_global_graph=parent_graph,
            activation_mask=np.array([True, False])
        )
        
        active_nodes = subgraph.get_active_nodes()
        assert active_nodes == {0}
        assert subgraph.is_node_active(0)
        assert not subgraph.is_node_active(1)
    
    def test_node_positions(self):
        """测试节点位置"""
        parent_graph = GlobalGraph(
            vocabulary={"phrase1": 0},
            reverse_vocabulary={0: "phrase1"}
        )
        
        subgraph = StateSubgraph(
            state_name="CA",
            parent_global_graph=parent_graph
        )
        
        subgraph.set_node_position(0, (1.0, 2.0))
        position = subgraph.get_node_position(0)
        assert position == (1.0, 2.0)
        assert subgraph.get_node_position(999) is None
    
    def test_statistics(self):
        """测试统计指标"""
        parent_graph = GlobalGraph(
            vocabulary={"phrase1": 0},
            reverse_vocabulary={0: "phrase1"}
        )
        
        subgraph = StateSubgraph(
            state_name="CA",
            parent_global_graph=parent_graph
        )
        
        subgraph.set_statistic("density", 0.5)
        subgraph.set_statistic("clustering", 0.3)
        
        assert subgraph.get_statistic("density") == 0.5
        assert subgraph.get_statistic("clustering") == 0.3
        assert subgraph.get_statistic("nonexistent") is None


class TestUtilityFunctions:
    """工具函数测试"""
    
    def test_validate_toc_json_valid(self, sample_toc_json):
        """测试有效JSON验证"""
        assert validate_toc_json(sample_toc_json) is True
    
    def test_validate_toc_json_invalid(self, invalid_toc_json):
        """测试无效JSON验证"""
        assert validate_toc_json(invalid_toc_json) is False
    
    def test_validate_toc_json_type_error(self):
        """测试类型错误JSON验证"""
        invalid_json = {
            "segment_id": "seg_001",
            "title": "Title",
            "level": "not_a_number",  # 应该是数字
            "order": 1,
            "text": "Text"
        }
        assert validate_toc_json(invalid_json) is False
    
    def test_create_phrase_mapping(self):
        """测试创建词组映射"""
        phrases = ["phrase2", "phrase1", "phrase3", "phrase1"]  # 包含重复
        vocab, reverse_vocab = create_phrase_mapping(phrases)
        
        # 应该去重并排序
        assert len(vocab) == 3
        assert len(reverse_vocab) == 3
        
        # 验证映射一致性
        for phrase, node_id in vocab.items():
            assert reverse_vocab[node_id] == phrase
        
        # 验证排序
        phrases_sorted = sorted(set(phrases))
        for i, phrase in enumerate(phrases_sorted):
            assert vocab[phrase] == i


# 属性测试
class TestDataModelsProperties:
    """数据模型属性测试"""
    
    @given(st.text(min_size=1), st.integers(min_value=0), st.integers(min_value=0))
    def test_toc_document_creation_property(self, text, level, order):
        """属性测试：TOC文档创建"""
        doc = TOCDocument(
            segment_id="test",
            title="Test Title",
            level=level,
            order=order,
            text=text
        )
        
        assert doc.segment_id == "test"
        assert doc.level == level
        assert doc.order == order
        assert doc.text == text
    
    @given(st.lists(st.text(min_size=1), min_size=0, max_size=10))
    def test_phrase_mapping_property(self, phrases):
        """属性测试：词组映射唯一性"""
        if not phrases:
            return
        
        vocab, reverse_vocab = create_phrase_mapping(phrases)
        
        # 验证映射唯一性
        assert len(vocab) == len(reverse_vocab)
        
        # 验证双向映射一致性
        for phrase, node_id in vocab.items():
            assert reverse_vocab[node_id] == phrase
        
        for node_id, phrase in reverse_vocab.items():
            assert vocab[phrase] == node_id
    
    @given(st.lists(st.text(min_size=1), min_size=1, max_size=5))
    def test_window_properties(self, phrases):
        """属性测试：窗口属性"""
        window = Window(
            window_id="test",
            phrases=phrases,
            source_doc="doc",
            state="CA",
            segment_id="seg"
        )
        
        assert len(window) == len(phrases)
        assert window.is_empty() == (len(phrases) == 0)
        assert window.phrases == phrases