"""
配置管理测试

测试配置系统的功能和正确性。
"""

import pytest
import json
import tempfile
from pathlib import Path

from semantic_coword_pipeline.core.config import Config, load_config, create_default_config_file


class TestConfig:
    """配置管理测试"""
    
    def test_default_config_initialization(self):
        """测试默认配置初始化"""
        config = Config()
        
        # 验证默认配置存在
        assert config.get('text_processing.ngram_size') == 2
        assert config.get('layout_engine.random_seed') == 42
        assert config.get('graph_construction.preserve_isolated_nodes') is True
        assert config.get('output.generate_visualizations') is True
    
    def test_get_nested_config(self):
        """测试获取嵌套配置"""
        config = Config()
        
        # 测试存在的配置
        assert config.get('text_processing.english_tokenizer') == 'nltk'
        assert config.get('stopword_discovery.tfidf_threshold') == 0.1
        
        # 测试不存在的配置
        assert config.get('nonexistent.key') is None
        assert config.get('nonexistent.key', 'default') == 'default'
    
    def test_set_nested_config(self):
        """测试设置嵌套配置"""
        config = Config()
        
        # 设置现有配置
        config.set('text_processing.ngram_size', 3)
        assert config.get('text_processing.ngram_size') == 3
        
        # 设置新配置
        config.set('new_section.new_key', 'new_value')
        assert config.get('new_section.new_key') == 'new_value'
    
    def test_update_config(self):
        """测试批量更新配置"""
        config = Config()
        
        update_dict = {
            'text_processing': {
                'ngram_size': 4,
                'new_option': True
            },
            'new_section': {
                'key1': 'value1',
                'key2': 42
            }
        }
        
        config.update(update_dict)
        
        assert config.get('text_processing.ngram_size') == 4
        assert config.get('text_processing.new_option') is True
        assert config.get('new_section.key1') == 'value1'
        assert config.get('new_section.key2') == 42
        
        # 验证其他配置未被影响
        assert config.get('text_processing.english_tokenizer') == 'nltk'
    
    def test_get_section(self):
        """测试获取配置节"""
        config = Config()
        
        text_processing = config.get_section('text_processing')
        assert isinstance(text_processing, dict)
        assert text_processing['ngram_size'] == 2
        assert text_processing['english_tokenizer'] == 'nltk'
        
        # 测试不存在的节
        nonexistent = config.get_section('nonexistent')
        assert nonexistent == {}
    
    def test_get_all_config(self):
        """测试获取所有配置"""
        config = Config()
        all_config = config.get_all()
        
        assert isinstance(all_config, dict)
        assert 'text_processing' in all_config
        assert 'layout_engine' in all_config
        assert 'output' in all_config
        
        # 验证是深拷贝
        all_config['text_processing']['ngram_size'] = 999
        assert config.get('text_processing.ngram_size') == 2
    
    def test_reset_to_default(self):
        """测试重置为默认配置"""
        config = Config()
        
        # 修改配置
        config.set('text_processing.ngram_size', 5)
        assert config.get('text_processing.ngram_size') == 5
        
        # 重置
        config.reset_to_default()
        assert config.get('text_processing.ngram_size') == 2
    
    def test_config_history(self):
        """测试配置历史记录"""
        config = Config()
        
        # 初始历史应该包含初始化记录
        history = config.get_config_history()
        assert len(history) >= 1
        assert history[0]['operation'] == 'initialization'
        
        # 进行一些配置变更
        config.set('test.key', 'value')
        config.update({'another': {'key': 'value'}})
        
        # 验证历史记录
        history = config.get_config_history()
        assert len(history) >= 3
        
        # 验证记录包含必要信息
        for record in history:
            assert 'timestamp' in record
            assert 'operation' in record
            assert 'target' in record
    
    def test_save_and_load_config(self, temp_dir):
        """测试保存和加载配置"""
        config = Config()
        config.set('test.key', 'test_value')
        
        config_file = temp_dir / 'test_config.json'
        
        # 保存配置
        config.save_to_file(str(config_file))
        assert config_file.exists()
        
        # 验证文件内容
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        assert saved_config['test']['key'] == 'test_value'
        
        # 加载配置到新实例
        new_config = Config(str(config_file))
        assert new_config.get('test.key') == 'test_value'
    
    def test_load_invalid_config_file(self):
        """测试加载无效配置文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json content')
            invalid_file = f.name
        
        with pytest.raises(ValueError, match="Failed to load config"):
            Config(invalid_file)
    
    def test_config_validation(self):
        """测试配置验证"""
        config = Config()
        
        # 默认配置应该通过验证
        validation = config.validate()
        assert len(validation['errors']) == 0
        
        # 设置无效配置
        config.set('text_processing.ngram_size', -1)
        config.set('layout_engine.max_iterations', 0)
        
        validation = config.validate()
        assert len(validation['errors']) >= 2
        
        # 验证错误消息
        error_messages = [error for error in validation['errors']]
        assert any('ngram_size must be >= 1' in msg for msg in error_messages)
        assert any('max_iterations must be >= 1' in msg for msg in error_messages)


class TestConfigUtilityFunctions:
    """配置工具函数测试"""
    
    def test_load_config_function(self):
        """测试load_config函数"""
        config = load_config()
        assert isinstance(config, Config)
        assert config.get('text_processing.ngram_size') == 2
    
    def test_create_default_config_file(self, temp_dir):
        """测试创建默认配置文件"""
        config_file = temp_dir / 'default_config.json'
        
        create_default_config_file(str(config_file))
        assert config_file.exists()
        
        # 验证文件内容
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        assert 'text_processing' in config_data
        assert 'layout_engine' in config_data
        assert config_data['text_processing']['ngram_size'] == 2


class TestConfigEdgeCases:
    """配置边界情况测试"""
    
    def test_deep_nested_config(self):
        """测试深层嵌套配置"""
        config = Config()
        
        # 设置深层嵌套配置
        config.set('level1.level2.level3.level4.key', 'deep_value')
        assert config.get('level1.level2.level3.level4.key') == 'deep_value'
        
        # 验证中间层级被正确创建
        assert isinstance(config.get('level1'), dict)
        assert isinstance(config.get('level1.level2'), dict)
        assert isinstance(config.get('level1.level2.level3'), dict)
    
    def test_config_with_none_values(self):
        """测试包含None值的配置"""
        config = Config()
        
        config.set('test.none_value', None)
        assert config.get('test.none_value') is None
        assert config.get('test.none_value', 'default') is None
    
    def test_config_with_complex_types(self):
        """测试复杂类型配置"""
        config = Config()
        
        complex_value = {
            'list': [1, 2, 3],
            'dict': {'nested': True},
            'tuple': (1, 2, 3)  # 注意：JSON序列化会将tuple转为list
        }
        
        config.set('test.complex', complex_value)
        retrieved = config.get('test.complex')
        
        assert retrieved['list'] == [1, 2, 3]
        assert retrieved['dict']['nested'] is True
    
    def test_config_overwrite_behavior(self):
        """测试配置覆盖行为"""
        config = Config()
        
        # 设置初始值
        config.set('test.key', 'initial')
        assert config.get('test.key') == 'initial'
        
        # 覆盖值
        config.set('test.key', 'overwritten')
        assert config.get('test.key') == 'overwritten'
        
        # 覆盖整个节
        config.update({'test': {'key': 'updated', 'new_key': 'new_value'}})
        assert config.get('test.key') == 'updated'
        assert config.get('test.new_key') == 'new_value'