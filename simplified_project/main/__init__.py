"""
语义增强共词网络分析管线

本包实现了一个完整的语义增强共词网络分析管线，采用"总图优先，州级激活"的两阶段构建策略。
系统以词组/短语为节点单位，通过动态停词发现和确定性布局确保可复现的网络分析结果。
"""

__version__ = "0.1.0"
__author__ = "Semantic Coword Enhancement Team"

from .core.data_models import (
    TOCDocument,
    ProcessedDocument,
    Window,
    Phrase,
    GlobalGraph,
    StateSubgraph
)

from .core.config import Config
from .core.error_handler import ErrorHandler
from .core.logger import setup_logger
from .pipeline import SemanticCowordPipeline
from .analyzers import NetworkAnalyzer

__all__ = [
    "TOCDocument",
    "ProcessedDocument", 
    "Window",
    "Phrase",
    "GlobalGraph",
    "StateSubgraph",
    "Config",
    "ErrorHandler",
    "setup_logger",
    "SemanticCowordPipeline",
    "NetworkAnalyzer"
]