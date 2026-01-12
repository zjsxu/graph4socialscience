"""
动态停词发现器测试

测试DynamicStopwordDiscoverer的功能，包括TF-IDF计算、停词识别和合并等。
"""

import pytest
import tempfile
import json
from pathlib import Path
from hypothesis import given, strategies as st, settings
from hypothesis import HealthCheck

from semantic_coword_pipeline.processors.dynamic_stopword_discoverer import (
    DynamicStopwordDiscoverer,
    TFIDFScore,
    StopwordDiscoveryResult
)
from semantic_coword_pipeline.core.data_models import TOCDocument, ProcessedDocument, Window


class TestDynamicStopwordDiscoverer:
    """动态停词发现器测试类"""
    
    def setup_method(self):
        """测试设置"""
        self.config = {
            'tfidf_threshold': 0.1,
            'frequency_threshold': 0.8,
            'min_document_frequency': 2,
            'enable_dynamic_discovery': True,
            'static_stopwords_path': None
        }
        self.discoverer = DynamicStopwordDiscoverer(self.config)
    
    def test_init_with_config(self):
        """测试初始化配置"""
        assert self.discoverer.tfidf_threshold == 0.1
        assert self.discoverer.frequency_threshold == 0.8
        assert self.discoverer.min_document_frequency == 2
        assert self.discoverer.enable_dynamic_discovery is True
        assert isinstance(self.discoverer.static_stopwords, set)
        assert len(self.discoverer.static_stopwords) > 0  # 应该包含基础停词
    
    def test_load_static_stopwords_file_not_exists(self):
        """测试加载不存在的静态停词文件"""
        config = self.config.copy()
        config['static_stopwords_path'] = 'nonexistent_file.txt'
        discoverer = DynamicStopwordDiscoverer(config)
        
        # 应该仍然包含基础停词
        assert len(discoverer.static_stopwords) > 0
        assert 'the' in discoverer.static_stopwords
        assert '的' in discoverer.static_stopwords
    
    def test_load_static_stopwords_from_file(self):
        """测试从文件加载静态停词"""
        # 创建临时停词文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write('custom_stopword1\n')
            f.write('custom_stopword2\n')
            f.write('# 这是注释\n')
            f.write('\n')  # 空行
            temp_path = f.name
        
        try:
            config = self.config.copy()
            config['static_stopwords_path'] = temp_path
            discoverer = DynamicStopwordDiscoverer(config)
            
            assert 'custom_stopword1' in discoverer.static_stopwords
            assert 'custom_stopword2' in discoverer.static_stopwords
            assert '# 这是注释' not in discoverer.static_stopwords  # 注释应该被忽略
            
        finally:
            Path(temp_path).unlink()
    
    def test_calculate_tfidf_matrix_empty_corpus(self):
        """测试空语料库的TF-IDF计算"""
        result = self.discoverer.calculate_tfidf_matrix([])
        assert result == {}
    
    def test_calculate_tfidf_matrix_single_document(self):
        """测试单文档的TF-IDF计算"""
        phrase_corpus = [['phrase1', 'phrase2', 'phrase1']]
        result = self.discoverer.calculate_tfidf_matrix(phrase_corpus)
        
        # 由于min_document_frequency=2，单文档中的词组应该被过滤掉
        assert len(result) == 0
    
    def test_calculate_tfidf_matrix_multiple_documents(self):
        """测试多文档的TF-IDF计算"""
        phrase_corpus = [
            ['phrase1', 'phrase2', 'phrase1'],
            ['phrase1', 'phrase3'],
            ['phrase2', 'phrase3', 'phrase4']
        ]
        result = self.discoverer.calculate_tfidf_matrix(phrase_corpus)
        
        # phrase1出现在2个文档中，phrase2出现在2个文档中，phrase3出现在2个文档中
        # phrase4只出现在1个文档中，应该被过滤
        assert 'phrase1' in result
        assert 'phrase2' in result
        assert 'phrase3' in result
        assert 'phrase4' not in result
        
        # 验证TF-IDF分数结构
        score = result['phrase1']
        assert isinstance(score, TFIDFScore)
        assert score.phrase == 'phrase1'
        assert score.df == 2  # 出现在2个文档中
        assert score.tf > 0
        assert score.idf > 0
        assert score.tfidf > 0
    
    def test_identify_low_discrimination_phrases(self):
        """测试低区分度词组识别"""
        # 创建模拟的TF-IDF分数
        tfidf_scores = {
            'high_freq_low_tfidf': TFIDFScore('high_freq_low_tfidf', 0.5, 10, 0.1, 0.05),  # 高频低TF-IDF
            'low_freq_high_tfidf': TFIDFScore('low_freq_high_tfidf', 0.1, 2, 1.0, 0.1),   # 低频高TF-IDF
            'high_freq_high_tfidf': TFIDFScore('high_freq_high_tfidf', 0.4, 8, 0.2, 0.15), # 高频高TF-IDF (TF-IDF > threshold)
            'low_freq_low_tfidf': TFIDFScore('low_freq_low_tfidf', 0.1, 1, 2.0, 0.05)      # 低频低TF-IDF
        }
        
        result = self.discoverer.identify_low_discrimination_phrases(tfidf_scores)
        
        # 只有高频低TF-IDF的词组应该被识别为动态停词
        # frequency_threshold = 0.8, max_df = 10, so cutoff = 8.0
        # high_freq_low_tfidf: df=10 >= 8.0 and tfidf=0.05 < 0.1 -> should be identified
        # high_freq_high_tfidf: df=8 >= 8.0 but tfidf=0.15 >= 0.1 -> should NOT be identified
        assert 'high_freq_low_tfidf' in result
        assert 'low_freq_high_tfidf' not in result  # df=2 < 8.0
        assert 'high_freq_high_tfidf' not in result  # tfidf=0.15 >= 0.1
        assert 'low_freq_low_tfidf' not in result    # df=1 < 8.0
    
    def test_merge_stopword_lists(self):
        """测试停词表合并"""
        dynamic_stopwords = {'dynamic1', 'dynamic2'}
        merged = self.discoverer.merge_stopword_lists(dynamic_stopwords)
        
        # 合并后的停词表应该包含静态和动态停词
        assert 'dynamic1' in merged
        assert 'dynamic2' in merged
        assert 'the' in merged  # 静态停词
        assert len(merged) >= len(self.discoverer.static_stopwords) + len(dynamic_stopwords)
    
    def test_apply_stopword_filter(self):
        """测试停词过滤"""
        phrases = ['the', 'good', 'phrase', 'and', 'another']
        filtered = self.discoverer.apply_stopword_filter(phrases)
        
        # 停词应该被过滤掉
        assert 'the' not in filtered
        assert 'and' not in filtered
        assert 'good' in filtered
        assert 'phrase' in filtered
        assert 'another' in filtered
    
    def test_apply_stopword_filter_custom_stopwords(self):
        """测试使用自定义停词过滤"""
        phrases = ['word1', 'word2', 'word3']
        custom_stopwords = {'word1', 'word3'}
        filtered = self.discoverer.apply_stopword_filter(phrases, custom_stopwords)
        
        assert 'word1' not in filtered
        assert 'word2' in filtered
        assert 'word3' not in filtered
    
    def test_discover_stopwords_disabled(self):
        """测试禁用动态停词发现"""
        config = self.config.copy()
        config['enable_dynamic_discovery'] = False
        discoverer = DynamicStopwordDiscoverer(config)
        
        # 创建模拟文档
        processed_docs = self._create_mock_processed_docs()
        
        result = discoverer.discover_stopwords(processed_docs)
        
        assert len(result.dynamic_stopwords) == 0
        assert result.merged_stopwords == discoverer.static_stopwords
        assert result.discovery_metadata['enabled'] is False
    
    def test_discover_stopwords_no_phrases(self):
        """测试没有词组的文档"""
        processed_docs = [
            ProcessedDocument(
                original_doc=TOCDocument('1', 'Title', 1, 1, 'Text'),
                cleaned_text='cleaned',
                tokens=['token1'],
                phrases=[],  # 空词组列表
                windows=[]
            )
        ]
        
        result = self.discoverer.discover_stopwords(processed_docs)
        
        assert len(result.dynamic_stopwords) == 0
        assert 'error' in result.discovery_metadata
    
    def test_discover_stopwords_normal_case(self):
        """测试正常情况的停词发现"""
        processed_docs = self._create_mock_processed_docs()
        
        result = self.discoverer.discover_stopwords(processed_docs)
        
        assert isinstance(result, StopwordDiscoveryResult)
        assert isinstance(result.dynamic_stopwords, set)
        assert isinstance(result.merged_stopwords, set)
        assert isinstance(result.tfidf_scores, dict)
        assert isinstance(result.discovery_metadata, dict)
        
        # 验证元数据
        metadata = result.discovery_metadata
        assert 'total_documents' in metadata
        assert 'total_unique_phrases' in metadata
        assert 'dynamic_stopwords_count' in metadata
        assert 'static_stopwords_count' in metadata
        assert 'merged_stopwords_count' in metadata
    
    def test_get_stopword_explanation(self):
        """测试停词解释功能"""
        tfidf_scores = {
            'test_phrase': TFIDFScore('test_phrase', 0.5, 10, 0.1, 0.05)
        }
        
        explanation = self.discoverer.get_stopword_explanation('test_phrase', tfidf_scores)
        
        assert explanation['phrase'] == 'test_phrase'
        assert 'is_static_stopword' in explanation
        assert 'is_dynamic_stopword' in explanation
        assert 'reason' in explanation
        assert 'tfidf_score' in explanation
        assert 'document_frequency' in explanation
    
    def test_save_stopword_analysis(self):
        """测试保存停词分析结果"""
        result = StopwordDiscoveryResult(
            dynamic_stopwords={'dynamic1'},
            merged_stopwords={'static1', 'dynamic1'},
            tfidf_scores={'phrase1': TFIDFScore('phrase1', 0.1, 2, 1.0, 0.1)},
            discovery_metadata={'test': True}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.discoverer.save_stopword_analysis(result, temp_path)
            
            # 验证文件内容
            with open(temp_path, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            assert 'dynamic_stopwords' in saved_data
            assert 'merged_stopwords' in saved_data
            assert 'tfidf_scores' in saved_data
            assert 'discovery_metadata' in saved_data
            
        finally:
            Path(temp_path).unlink()
    
    def _create_mock_processed_docs(self) -> list:
        """创建模拟的处理后文档"""
        return [
            ProcessedDocument(
                original_doc=TOCDocument('1', 'Title1', 1, 1, 'Text1'),
                cleaned_text='cleaned1',
                tokens=['token1', 'token2'],
                phrases=['common_phrase', 'unique_phrase1', 'common_phrase'],
                windows=[Window('w1', ['common_phrase', 'unique_phrase1'], 'doc1', 'state1', 'seg1')]
            ),
            ProcessedDocument(
                original_doc=TOCDocument('2', 'Title2', 1, 2, 'Text2'),
                cleaned_text='cleaned2',
                tokens=['token3', 'token4'],
                phrases=['common_phrase', 'unique_phrase2'],
                windows=[Window('w2', ['common_phrase', 'unique_phrase2'], 'doc2', 'state2', 'seg2')]
            ),
            ProcessedDocument(
                original_doc=TOCDocument('3', 'Title3', 1, 3, 'Text3'),
                cleaned_text='cleaned3',
                tokens=['token5', 'token6'],
                phrases=['common_phrase', 'unique_phrase3', 'common_phrase'],
                windows=[Window('w3', ['common_phrase', 'unique_phrase3'], 'doc3', 'state3', 'seg3')]
            )
        ]


class TestTFIDFScore:
    """TF-IDF分数测试类"""
    
    def test_tfidf_score_creation(self):
        """测试TF-IDF分数创建"""
        score = TFIDFScore('test_phrase', 0.5, 10, 0.2, 0.1)
        
        assert score.phrase == 'test_phrase'
        assert score.tf == 0.5
        assert score.df == 10
        assert score.idf == 0.2
        assert score.tfidf == 0.1
    
    def test_tfidf_score_to_dict(self):
        """测试TF-IDF分数转换为字典"""
        score = TFIDFScore('test_phrase', 0.5, 10, 0.2, 0.1)
        result = score.to_dict()
        
        expected = {
            'phrase': 'test_phrase',
            'tf': 0.5,
            'df': 10,
            'idf': 0.2,
            'tfidf': 0.1
        }
        
        assert result == expected


class TestStopwordDiscoveryResult:
    """停词发现结果测试类"""
    
    def test_stopword_discovery_result_creation(self):
        """测试停词发现结果创建"""
        result = StopwordDiscoveryResult(
            dynamic_stopwords={'word1'},
            merged_stopwords={'word1', 'word2'},
            tfidf_scores={'word1': TFIDFScore('word1', 0.1, 2, 1.0, 0.1)},
            discovery_metadata={'test': True}
        )
        
        assert result.dynamic_stopwords == {'word1'}
        assert result.merged_stopwords == {'word1', 'word2'}
        assert 'word1' in result.tfidf_scores
        assert result.discovery_metadata['test'] is True
    
    def test_stopword_discovery_result_to_dict(self):
        """测试停词发现结果转换为字典"""
        result = StopwordDiscoveryResult(
            dynamic_stopwords={'word1'},
            merged_stopwords={'word1', 'word2'},
            tfidf_scores={'word1': TFIDFScore('word1', 0.1, 2, 1.0, 0.1)},
            discovery_metadata={'test': True}
        )
        
        result_dict = result.to_dict()
        
        assert 'dynamic_stopwords' in result_dict
        assert 'merged_stopwords' in result_dict
        assert 'tfidf_scores' in result_dict
        assert 'discovery_metadata' in result_dict
        
        assert result_dict['dynamic_stopwords'] == ['word1']
        assert set(result_dict['merged_stopwords']) == {'word1', 'word2'}


# 属性测试
class TestDynamicStopwordDiscovererProperties:
    """动态停词发现器属性测试"""
    
    def setup_method(self):
        """测试设置"""
        self.config = {
            'tfidf_threshold': 0.1,
            'frequency_threshold': 0.8,
            'min_document_frequency': 2,
            'enable_dynamic_discovery': True,
            'static_stopwords_path': None
        }
    
    @given(
        tfidf_threshold=st.floats(min_value=0.01, max_value=1.0),
        frequency_threshold=st.floats(min_value=0.1, max_value=1.0),
        min_doc_freq=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_tfidf_calculation_correctness(self, tfidf_threshold, frequency_threshold, min_doc_freq):
        """
        属性10: TF-IDF计算正确性
        对于任何词组语料库，计算的TF-IDF值应该符合标准TF-IDF公式：tf(t,d) * log(N/df(t))
        验证：需求 4.1
        """
        config = self.config.copy()
        config.update({
            'tfidf_threshold': tfidf_threshold,
            'frequency_threshold': frequency_threshold,
            'min_document_frequency': min_doc_freq
        })
        
        discoverer = DynamicStopwordDiscoverer(config)
        
        # 创建简单的测试语料库
        phrase_corpus = [
            ['phrase_a', 'phrase_b'],
            ['phrase_a', 'phrase_c'],
            ['phrase_b', 'phrase_c']
        ]
        
        tfidf_scores = discoverer.calculate_tfidf_matrix(phrase_corpus)
        
        # 验证TF-IDF计算的数学正确性
        for phrase, score in tfidf_scores.items():
            # 验证基本属性
            assert score.tf >= 0, f"TF should be non-negative for {phrase}"
            assert score.df >= min_doc_freq, f"DF should be >= min_doc_freq for {phrase}"
            assert score.idf >= 0, f"IDF should be non-negative for {phrase}"
            assert score.tfidf >= 0, f"TF-IDF should be non-negative for {phrase}"
            
            # 验证TF-IDF公式: tfidf = tf * idf
            expected_tfidf = score.tf * score.idf
            assert abs(score.tfidf - expected_tfidf) < 1e-10, f"TF-IDF formula incorrect for {phrase}"
    
    @given(
        phrases=st.lists(
            st.lists(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10), min_size=1, max_size=20),
            min_size=2, max_size=10
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_dynamic_stopword_identification_accuracy(self, phrases):
        """
        属性11: 动态停词识别准确性
        对于任何识别为动态停词的词组，其文档频率应该高于设定阈值且TF-IDF值应该低于区分度阈值
        验证：需求 4.2
        """
        discoverer = DynamicStopwordDiscoverer(self.config)
        
        # 过滤掉空的词组列表
        valid_phrases = [doc for doc in phrases if doc]
        if len(valid_phrases) < 2:
            return  # 跳过无效的测试用例
        
        tfidf_scores = discoverer.calculate_tfidf_matrix(valid_phrases)
        dynamic_stopwords = discoverer.identify_low_discrimination_phrases(tfidf_scores)
        
        # 计算频率阈值
        if tfidf_scores:
            max_df = max(score.df for score in tfidf_scores.values())
            frequency_cutoff = max_df * discoverer.frequency_threshold
            
            # 验证每个动态停词都满足条件
            for stopword in dynamic_stopwords:
                if stopword in tfidf_scores:
                    score = tfidf_scores[stopword]
                    
                    # 条件1: TF-IDF分数低于阈值
                    assert score.tfidf < discoverer.tfidf_threshold, \
                        f"Dynamic stopword {stopword} should have low TF-IDF ({score.tfidf} >= {discoverer.tfidf_threshold})"
                    
                    # 条件2: 文档频率高于阈值
                    assert score.df >= frequency_cutoff, \
                        f"Dynamic stopword {stopword} should have high frequency ({score.df} < {frequency_cutoff})"
                    
                    # 条件3: 不是静态停词
                    assert stopword not in discoverer.static_stopwords, \
                        f"Dynamic stopword {stopword} should not be in static stopwords"
    
    @given(
        phrases=st.lists(st.text(alphabet='abcdefghijklmnopqrstuvwxyz', min_size=1, max_size=10), min_size=0, max_size=50)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_property_stopword_filtering_thoroughness(self, phrases):
        """
        属性12: 停词过滤彻底性
        对于任何最终生成的词表，其中不应该包含合并停词表中的任何词组
        验证：需求 4.5
        """
        discoverer = DynamicStopwordDiscoverer(self.config)
        
        # 创建合并停词表
        dynamic_stopwords = {'dynamic_stop1', 'dynamic_stop2'}
        merged_stopwords = discoverer.merge_stopword_lists(dynamic_stopwords)
        
        # 应用停词过滤
        filtered_phrases = discoverer.apply_stopword_filter(phrases, merged_stopwords)
        
        # 验证过滤的彻底性
        for phrase in filtered_phrases:
            assert phrase not in merged_stopwords, \
                f"Filtered phrase list should not contain stopword: {phrase}"
        
        # 验证过滤后的词组都是原始词组的子集
        for phrase in filtered_phrases:
            assert phrase in phrases, \
                f"Filtered phrase {phrase} should be from original phrases"