"""
pytest配置文件

定义测试夹具和配置，支持单元测试和属性测试。
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
import json

from semantic_coword_pipeline.core.config import Config
from semantic_coword_pipeline.core.logger import setup_logger
from semantic_coword_pipeline.core.error_handler import ErrorHandler
from semantic_coword_pipeline.core.data_models import TOCDocument, ProcessedDocument, Window, Phrase


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def test_config():
    """测试配置"""
    return Config()


@pytest.fixture
def test_logger(temp_dir):
    """测试日志记录器"""
    log_config = {
        'level': 'DEBUG',
        'file_path': str(temp_dir / 'test.log')
    }
    return setup_logger('test_logger', log_config)


@pytest.fixture
def error_handler():
    """错误处理器"""
    return ErrorHandler()


@pytest.fixture
def sample_toc_document():
    """示例TOC文档"""
    return TOCDocument(
        segment_id="seg_001",
        title="Introduction",
        level=1,
        order=1,
        text="This is a sample text for testing purposes. It contains multiple sentences.",
        state="CA",
        language="en"
    )


@pytest.fixture
def sample_toc_json():
    """示例TOC JSON数据"""
    return {
        "segment_id": "seg_001",
        "title": "Introduction", 
        "level": 1,
        "order": 1,
        "text": "This is a sample text for testing purposes. It contains multiple sentences.",
        "state": "CA",
        "language": "en"
    }


@pytest.fixture
def sample_phrases():
    """示例词组列表"""
    return [
        Phrase("sample text", frequency=5, tfidf_score=0.8),
        Phrase("testing purposes", frequency=3, tfidf_score=0.6),
        Phrase("multiple sentences", frequency=2, tfidf_score=0.4)
    ]


@pytest.fixture
def sample_window():
    """示例窗口"""
    return Window(
        window_id="win_001",
        phrases=["sample text", "testing purposes", "multiple sentences"],
        source_doc="seg_001",
        state="CA",
        segment_id="seg_001"
    )


@pytest.fixture
def sample_processed_document(sample_toc_document, sample_window):
    """示例处理后文档"""
    return ProcessedDocument(
        original_doc=sample_toc_document,
        cleaned_text="sample text testing purposes multiple sentences",
        tokens=["sample", "text", "testing", "purposes", "multiple", "sentences"],
        phrases=["sample text", "testing purposes", "multiple sentences"],
        windows=[sample_window]
    )


@pytest.fixture
def invalid_toc_json():
    """无效的TOC JSON数据"""
    return {
        "segment_id": "seg_001",
        # 缺少必需字段
        "text": "Some text"
    }


# Hypothesis配置
from hypothesis import settings, HealthCheck

# 属性测试配置
PROPERTY_TEST_CONFIG = {
    'max_examples': 100,
    'deadline': 10000,  # 10秒超时
    'suppress_health_check': [HealthCheck.too_slow]
}

# 应用全局设置
settings.register_profile("default", **PROPERTY_TEST_CONFIG)
settings.load_profile("default")