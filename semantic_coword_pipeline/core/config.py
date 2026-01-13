"""
配置管理系统

提供分层配置管理，支持默认配置、用户配置和运行时配置的覆盖。
根据需求10.5提供可追溯的配置记录。
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import copy
from datetime import datetime


class Config:
    """
    配置管理类
    
    支持分层配置管理：默认配置 < 用户配置 < 运行时配置
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        'text_processing': {
            'english_tokenizer': 'nltk',
            'chinese_tokenizer': 'jieba',
            'ngram_size': 2,
            'min_phrase_frequency': 3,
            'normalize_text': True,
            'remove_punctuation': True,
            'convert_to_lowercase': True
        },
        'stopword_discovery': {
            'tfidf_threshold': 0.1,
            'frequency_threshold': 0.8,
            'static_stopwords_path': 'data/stopwords.txt',
            'enable_dynamic_discovery': True,
            'min_document_frequency': 2
        },
        'graph_construction': {
            'preserve_isolated_nodes': True,
            'edge_weight_method': 'binary',
            'window_type': 'segment',
            'min_cooccurrence_count': 1,
            'use_sparse_matrix': True
        },
        'layout_engine': {
            'algorithm': 'force_directed',
            'random_seed': 42,
            'cache_enabled': True,
            'max_iterations': 1000,
            'convergence_threshold': 1e-6,
            'spring_constant': 1.0,
            'repulsion_strength': 1.0
        },
        'subgraph_activation': {
            'activation_method': 'reweight',
            'preserve_global_positions': True,
            'min_edge_weight': 0.0,
            'normalize_weights': False
        },
        'output': {
            'base_path': 'output/',
            'generate_visualizations': True,
            'export_formats': ['json', 'graphml', 'csv'],
            'save_intermediate_results': True,
            'compression_enabled': False
        },
        'performance': {
            'enable_parallel_processing': True,
            'max_workers': 4,
            'batch_size': 1000,
            'memory_limit_mb': 4096,
            'enable_profiling': False
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file_path': 'logs/pipeline.log',
            'max_file_size_mb': 100,
            'backup_count': 5
        },
        'error_handling': {
            'max_retries': 3,
            'retry_delay': 1.0,
            'fallback_strategies': {
                'text_processing': 'simple_split',
                'phrase_extraction': 'word_level',
                'graph_construction': 'minimal_graph'
            },
            'continue_on_error': False,
            'error_report_path': 'logs/error_report.json'
        },
        'integration': {
            'easygraph_compatibility': True,
            'export_formats': ['json', 'graphml', 'csv', 'pickle'],
            'multi_view_support': True,
            'graph_fusion_ready': True,
            'openrank_integration': True
        },
        'validation': {
            'input_validation': True,
            'output_validation': True,
            'intermediate_validation': False,
            'strict_mode': False
        },
        'enhanced_text_processing': {
            # Phrase extraction settings
            'min_phrase_length': 2,
            'max_phrase_length': 4,
            
            # Static stopword settings
            'english_stopwords_file': None,
            'chinese_stopwords_file': None,
            
            # Dynamic stopword identification
            'df_threshold_ratio': 0.8,  # DF / N >= threshold for high frequency
            'tfidf_threshold': 0.1,     # Low TF-IDF threshold for low discrimination
            'min_frequency_for_stopword': 5,  # Minimum frequency to consider as stopword
            
            # spaCy settings
            'use_spacy': True,
            'spacy_models': {
                'english': 'en_core_web_sm',
                'chinese': 'zh_core_web_sm'
            }
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 用户配置文件路径
        """
        self._config = copy.deepcopy(self.DEFAULT_CONFIG)
        self._config_history = []
        self._config_path = config_path
        
        # 记录初始配置
        self._record_config_change("initialization", "default", self._config)
        
        # 加载用户配置
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        支持点分隔的嵌套键，如 'text_processing.ngram_size'
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        支持点分隔的嵌套键，如 'text_processing.ngram_size'
        """
        keys = key.split('.')
        config = self._config
        
        # 导航到目标位置
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 记录变更前的值
        old_value = config.get(keys[-1])
        
        # 设置新值
        config[keys[-1]] = value
        
        # 记录配置变更
        self._record_config_change("set", key, {
            'old_value': old_value,
            'new_value': value
        })
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        批量更新配置
        """
        old_config = copy.deepcopy(self._config)
        self._deep_update(self._config, config_dict)
        
        # 记录配置变更
        self._record_config_change("update", "batch", {
            'old_config': old_config,
            'new_config': copy.deepcopy(self._config),
            'update_dict': config_dict
        })
    
    def load_from_file(self, file_path: str) -> None:
        """
        从文件加载配置
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            old_config = copy.deepcopy(self._config)
            self._deep_update(self._config, user_config)
            
            # 记录配置变更
            self._record_config_change("load_file", file_path, {
                'old_config': old_config,
                'new_config': copy.deepcopy(self._config),
                'loaded_config': user_config
            })
            
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load config from {file_path}: {e}")
    
    def save_to_file(self, file_path: str) -> None:
        """
        保存配置到文件
        """
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            
            # 记录配置保存
            self._record_config_change("save_file", file_path, {
                'saved_config': copy.deepcopy(self._config)
            })
            
        except Exception as e:
            raise ValueError(f"Failed to save config to {file_path}: {e}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置节
        """
        return self._config.get(section, {})
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有配置
        """
        return copy.deepcopy(self._config)
    
    def reset_to_default(self) -> None:
        """
        重置为默认配置
        """
        old_config = copy.deepcopy(self._config)
        self._config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        # 记录配置重置
        self._record_config_change("reset", "default", {
            'old_config': old_config,
            'new_config': copy.deepcopy(self._config)
        })
    
    def get_config_history(self) -> list:
        """
        获取配置变更历史
        
        根据需求10.5提供可追溯的配置记录。
        """
        return copy.deepcopy(self._config_history)
    
    def validate(self) -> Dict[str, list]:
        """
        验证配置的有效性
        """
        errors = []
        warnings = []
        
        # 验证必需的配置项
        required_sections = ['text_processing', 'graph_construction', 'layout_engine', 'output']
        for section in required_sections:
            if section not in self._config:
                errors.append(f"Missing required section: {section}")
        
        # 验证数值范围
        if self.get('text_processing.ngram_size', 0) < 1:
            errors.append("text_processing.ngram_size must be >= 1")
        
        if self.get('layout_engine.random_seed') is None:
            warnings.append("layout_engine.random_seed is not set, results may not be reproducible")
        
        if self.get('layout_engine.max_iterations', 0) < 1:
            errors.append("layout_engine.max_iterations must be >= 1")
        
        # 验证文件路径
        output_path = self.get('output.base_path')
        if output_path and not os.path.exists(os.path.dirname(output_path)):
            warnings.append(f"Output directory does not exist: {os.path.dirname(output_path)}")
        
        return {
            'errors': errors,
            'warnings': warnings
        }
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        深度更新字典
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _record_config_change(self, operation: str, target: str, details: Any) -> None:
        """
        记录配置变更
        """
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'target': target,
            'details': details
        }
        self._config_history.append(change_record)
    
    def __str__(self) -> str:
        """字符串表示"""
        try:
            # 创建一个可序列化的配置副本
            serializable_config = self._make_serializable(self._config)
            return json.dumps(serializable_config, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            # 如果序列化失败，返回简单的字符串表示
            return f"Config(sections={list(self._config.keys())})"
    
    def _make_serializable(self, obj):
        """将配置对象转换为可序列化的格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif callable(obj):
            return f"<function {obj.__name__}>"
        else:
            return obj
    
    def copy(self) -> Dict[str, Any]:
        """
        创建配置的深拷贝
        
        Returns:
            配置的深拷贝
        """
        return copy.deepcopy(self._config)
    
    def __repr__(self) -> str:
        """对象表示"""
        return f"Config(sections={list(self._config.keys())})"


def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载配置的便捷函数
    """
    return Config(config_path)


def create_default_config_file(file_path: str) -> None:
    """
    创建默认配置文件
    """
    config = Config()
    config.save_to_file(file_path)