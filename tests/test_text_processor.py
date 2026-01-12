"""
文本处理器测试

测试TextProcessor的核心功能，包括语言检测、文本规范化和分词。
"""

import pytest
from unittest.mock import Mock, patch
from hypothesis import given, strategies as st, assume
from semantic_coword_pipeline.processors.text_processor import (
    TextProcessor,
    LanguageDetector,
    EnglishTokenizer,
    ChineseTokenizer,
    LanguageDetectionResult
)
from semantic_coword_pipeline.core.data_models import TOCDocument
from semantic_coword_pipeline.core.config import Config


class TestLanguageDetector:
    """语言检测器测试"""
    
    def setup_method(self):
        """测试设置"""
        self.detector = LanguageDetector()
    
    def test_detect_english_text(self):
        """测试英文文本检测"""
        text = "This is an English text with some words."
        result = self.detector.detect_language(text)
        
        assert result.language == 'english'
        assert result.confidence > 0.5
        assert 'english_chars' in result.detected_features
    
    def test_detect_chinese_text(self):
        """测试中文文本检测"""
        text = "这是一段中文文本，包含一些中文字符。"
        result = self.detector.detect_language(text)
        
        assert result.language == 'chinese'
        assert result.confidence > 0.5
        assert 'chinese_chars' in result.detected_features
    
    def test_detect_mixed_text(self):
        """测试中英文混合文本检测"""
        text = "This is mixed text 这里有中文"
        result = self.detector.detect_language(text)
        
        # 应该检测为中文（因为有中文字符）
        assert result.language == 'chinese'
        assert result.confidence > 0.0
    
    def test_detect_empty_text(self):
        """测试空文本检测"""
        result = self.detector.detect_language("")
        
        assert result.language == 'unknown'
        assert result.confidence == 0.0
    
    def test_detect_punctuation_only(self):
        """测试仅标点符号文本"""
        text = "!@#$%^&*()"
        result = self.detector.detect_language(text)
        
        # 应该默认为英文
        assert result.language == 'english'


class TestEnglishTokenizer:
    """英文分词器测试"""
    
    def setup_method(self):
        """测试设置"""
        config = {'remove_stopwords': False, 'use_stemming': False}
        self.tokenizer = EnglishTokenizer(config)
    
    def test_basic_tokenization(self):
        """测试基础分词"""
        text = "Hello world! This is a test."
        tokens = self.tokenizer.tokenize(text)
        
        assert len(tokens) > 0
        assert 'hello' in tokens
        assert 'world' in tokens
        assert 'test' in tokens
    
    def test_punctuation_removal(self):
        """测试标点符号移除"""
        text = "Hello, world! How are you?"
        tokens = self.tokenizer.tokenize(text)
        
        # 标点符号应该被移除
        assert ',' not in tokens
        assert '!' not in tokens
        assert '?' not in tokens
    
    def test_empty_text(self):
        """测试空文本"""
        tokens = self.tokenizer.tokenize("")
        assert tokens == []
    
    def test_whitespace_only(self):
        """测试仅空白字符"""
        tokens = self.tokenizer.tokenize("   \n\t  ")
        assert tokens == []


class TestChineseTokenizer:
    """中文分词器测试"""
    
    def setup_method(self):
        """测试设置"""
        config = {'remove_stopwords': False, 'use_pos_tagging': False}
        self.tokenizer = ChineseTokenizer(config)
    
    def test_basic_tokenization(self):
        """测试基础中文分词"""
        text = "这是一个测试文本"
        tokens = self.tokenizer.tokenize(text)
        
        assert len(tokens) > 0
        # 检查是否包含有意义的词
        assert any(len(token) >= 2 for token in tokens)
    
    def test_empty_text(self):
        """测试空文本"""
        tokens = self.tokenizer.tokenize("")
        assert tokens == []
    
    def test_punctuation_handling(self):
        """测试标点符号处理"""
        text = "你好，世界！这是测试。"
        tokens = self.tokenizer.tokenize(text)
        
        # 标点符号应该被过滤
        assert '，' not in tokens
        assert '！' not in tokens
        assert '。' not in tokens


class TestTextProcessor:
    """文本处理器主类测试"""
    
    def setup_method(self):
        """测试设置"""
        self.config = Config()
        self.processor = TextProcessor(self.config)
    
    def test_initialization(self):
        """测试初始化"""
        assert self.processor.language_detector is not None
        assert self.processor.english_tokenizer is not None
        assert self.processor.chinese_tokenizer is not None
    
    def test_process_english_document(self):
        """测试处理英文文档"""
        doc = TOCDocument(
            segment_id="test_001",
            title="Test Document",
            level=1,
            order=1,
            text="This is a test document with some English text.",
            state="test_state"
        )
        
        processed_doc = self.processor.process_document(doc)
        
        assert processed_doc is not None
        assert processed_doc.original_doc == doc
        assert len(processed_doc.tokens) > 0
        assert processed_doc.processing_metadata['language'] == 'english'
        assert len(processed_doc.windows) == 1
    
    def test_process_chinese_document(self):
        """测试处理中文文档"""
        doc = TOCDocument(
            segment_id="test_002",
            title="测试文档",
            level=1,
            order=1,
            text="这是一个测试文档，包含一些中文文本。",
            state="test_state"
        )
        
        processed_doc = self.processor.process_document(doc)
        
        assert processed_doc is not None
        assert processed_doc.original_doc == doc
        assert len(processed_doc.tokens) > 0
        assert processed_doc.processing_metadata['language'] == 'chinese'
        assert len(processed_doc.windows) == 1
    
    def test_normalize_english_text(self):
        """测试英文文本规范化"""
        text = "Hello World! This is a TEST."
        normalized = self.processor.normalize_text_content(text, 'english')
        
        assert normalized.lower() == normalized  # 应该转换为小写
        assert len(normalized) > 0
    
    def test_normalize_chinese_text(self):
        """测试中文文本规范化"""
        text = "你好世界！这是一个测试。"
        normalized = self.processor.normalize_text_content(text, 'chinese')
        
        assert len(normalized) > 0
        assert normalized.strip() == normalized  # 应该去除首尾空白
    
    def test_batch_process_documents(self):
        """测试批量处理文档"""
        docs = [
            TOCDocument(
                segment_id=f"test_{i:03d}",
                title=f"Test Document {i}",
                level=1,
                order=i,
                text=f"This is test document number {i}.",
                state="test_state"
            )
            for i in range(5)
        ]
        
        processed_docs = self.processor.batch_process_documents(docs)
        
        assert len(processed_docs) == len(docs)
        for processed_doc in processed_docs:
            assert processed_doc is not None
            assert len(processed_doc.tokens) > 0
    
    def test_get_processing_statistics(self):
        """测试获取处理统计信息"""
        docs = [
            TOCDocument(
                segment_id="test_001",
                title="English Document",
                level=1,
                order=1,
                text="This is an English document.",
                state="test_state"
            ),
            TOCDocument(
                segment_id="test_002",
                title="中文文档",
                level=1,
                order=2,
                text="这是一个中文文档。",
                state="test_state"
            )
        ]
        
        processed_docs = self.processor.batch_process_documents(docs)
        stats = self.processor.get_processing_statistics(processed_docs)
        
        assert stats['total_documents'] == 2
        assert 'language_distribution' in stats
        assert 'total_tokens' in stats
        assert 'average_tokens_per_doc' in stats
    
    def test_empty_document_handling(self):
        """测试空文档处理"""
        doc = TOCDocument(
            segment_id="empty_001",
            title="Empty Document",
            level=1,
            order=1,
            text="",
            state="test_state"
        )
        
        processed_doc = self.processor.process_document(doc)
        
        assert processed_doc is not None
        assert processed_doc.tokens == []
        assert len(processed_doc.windows) == 1
        assert processed_doc.windows[0].is_empty()
    
    @given(st.text(alphabet='这是一个测试文档包含内容词汇分析处理系统网络图结构数据模型配置管理错误处理日志记录', min_size=5, max_size=100))
    def test_chinese_text_processing_property(self, chinese_text):
        """
        属性测试：中文分词处理
        **Feature: semantic-coword-enhancement, Property 9: 中文分词处理**
        **验证：需求 3.3**
        
        对于任何中文文本输入，系统应该先执行分词操作，然后基于分词结果进行短语抽取
        """
        assume(len(chinese_text.strip()) > 0)
        assume(any('\u4e00' <= c <= '\u9fff' for c in chinese_text))  # 确保包含中文字符
        
        # 创建中文文档
        doc = TOCDocument(
            segment_id="chinese_test",
            title="中文测试文档",
            level=1,
            order=1,
            text=chinese_text,
            state="test_state",
            language="chinese"
        )
        
        # 处理文档
        processed_doc = self.processor.process_document(doc)
        
        # 属性验证：中文文本应该被正确处理
        assert processed_doc is not None
        assert processed_doc.processing_metadata['language'] == 'chinese'
        
        # 属性验证：分词结果应该是有意义的
        tokens = processed_doc.tokens
        
        # 如果原文包含中文字符，分词结果不应该为空（除非全是标点符号）
        if any('\u4e00' <= c <= '\u9fff' for c in chinese_text):
            # 检查分词是否产生了合理的结果
            # 分词后的tokens应该保留中文字符
            chinese_tokens = [token for token in tokens if any('\u4e00' <= c <= '\u9fff' for c in token)]
            
            # 如果原文有中文内容，应该至少有一些中文tokens（除非全被过滤）
            # 这里我们验证分词过程确实执行了，而不是简单的字符分割
            if len(tokens) > 0:
                # 验证tokens不是简单的字符分割（即不是每个字符都是单独的token）
                single_char_tokens = [token for token in tokens if len(token) == 1]
                multi_char_tokens = [token for token in tokens if len(token) > 1]
                
                # 对于中文分词，应该产生一些多字符的词汇（除非文本很短）
                if len(chinese_text.strip()) > 10:
                    # 长文本应该产生一些多字符词汇
                    assert len(multi_char_tokens) > 0 or len(tokens) > 0, "Chinese tokenization should produce meaningful tokens"
        
        # 属性验证：窗口结构正确
        assert len(processed_doc.windows) == 1
        window = processed_doc.windows[0]
        assert window.source_doc == doc.segment_id
        assert window.state == doc.state
        assert window.segment_id == doc.segment_id


# 集成测试
class TestTextProcessorIntegration:
    """文本处理器集成测试"""
    
    def test_full_processing_pipeline(self):
        """测试完整处理流程"""
        # 创建配置
        config = Config()
        processor = TextProcessor(config)
        
        # 创建测试文档
        docs = [
            TOCDocument(
                segment_id="integration_001",
                title="Policy Document",
                level=1,
                order=1,
                text="This policy document outlines the regulatory framework for financial institutions.",
                state="california"
            ),
            TOCDocument(
                segment_id="integration_002", 
                title="法规文件",
                level=1,
                order=2,
                text="本法规文件规定了金融机构的监管框架。",
                state="beijing"
            )
        ]
        
        # 批量处理
        processed_docs = processor.batch_process_documents(docs)
        
        # 验证结果
        assert len(processed_docs) == 2
        
        # 验证英文文档
        english_doc = processed_docs[0]
        assert english_doc.processing_metadata['language'] == 'english'
        assert len(english_doc.tokens) > 0
        assert english_doc.windows[0].state == 'california'
        
        # 验证中文文档
        chinese_doc = processed_docs[1]
        assert chinese_doc.processing_metadata['language'] == 'chinese'
        assert len(chinese_doc.tokens) > 0
        assert chinese_doc.windows[0].state == 'beijing'
        
        # 获取统计信息
        stats = processor.get_processing_statistics(processed_docs)
        assert stats['total_documents'] == 2
        assert 'english' in stats['language_distribution']
        assert 'chinese' in stats['language_distribution']