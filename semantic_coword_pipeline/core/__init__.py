"""
核心模块

包含数据模型、配置管理、错误处理和日志系统等核心组件。
"""

from .data_models import (
    TOCDocument,
    ProcessedDocument,
    Window,
    Phrase,
    GlobalGraph,
    StateSubgraph
)

from .config import Config
from .error_handler import ErrorHandler
from .logger import setup_logger

__all__ = [
    "TOCDocument",
    "ProcessedDocument",
    "Window", 
    "Phrase",
    "GlobalGraph",
    "StateSubgraph",
    "Config",
    "ErrorHandler",
    "setup_logger"
]