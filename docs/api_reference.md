# API 参考文档

## 概述

本文档提供语义增强共词网络分析管线的完整API参考，包括所有核心模块、类和函数的详细说明。

## 核心模块

### semantic_coword_pipeline.core

#### Config 类

配置管理类，支持分层配置和运行时修改。

```python
from semantic_coword_pipeline.core.config import Config

# 创建配置实例
config = Config()

# 获取配置值
ngram_size = config.get('text_processing.ngram_size')  # 返回: 2

# 设置配置值
config.set('text_processing.ngram_size', 3)

# 批量更新配置
config.update({
    'layout_engine': {
        'random_seed': 123,
        'max_iterations': 2000
    }
})

# 保存配置到文件
config.save_to_file('my_config.json')

# 验证配置
validation_result = config.validate()
print(validation_result['errors'])  # 显示配置错误
```

**方法说明:**

- `get(key: str, default: Any = None) -> Any`: 获取配置值，支持点分隔的嵌套键
- `set(key: str, value: Any) -> None`: 设置配置值
- `update(config_dict: Dict[str, Any]) -> None`: 批量更新配置
- `load_from_file(file_path: str) -> None`: 从文件加载配置
- `save_to_file(file_path: str) -> None`: 保存配置到文件
- `validate() -> Dict[str, list]`: 验证配置有效性
- `get_config_history() -> list`: 获取配置变更历史

#### TOCDocument 类

TOC文档数据模型，表示输入的政策/法规文档片段。

```python
from semantic_coword_pipeline.core.data_models import TOCDocument

# 创建文档实例
doc = TOCDocument(
    segment_id="ca_001",
    title="California Environmental Policy",
    level=1,
    order=1,
    text="Natural language processing applications in environmental monitoring...",
    state="California"
)

# 从JSON创建
json_data = {
    "segment_id": "tx_001",
    "title": "Texas Energy Policy",
    "level": 1,
    "order": 1,
    "text": "Energy efficiency standards and renewable energy targets..."
}
doc = TOCDocument.from_json(json_data)

# 转换为字典
doc_dict = doc.to_dict()
```

**属性说明:**

- `segment_id: str`: 文档片段唯一标识符
- `title: str`: 文档片段标题
- `level: int`: 文档层级（1为顶级）
- `order: int`: 在同级中的顺序
- `text: str`: 文档正文内容
- `state: Optional[str]`: 所属州/地区（可选）
- `language: Optional[str]`: 文档语言（自动检测）

#### GlobalGraph 类

全局共现网络数据模型。

```python
from semantic_coword_pipeline.core.data_models import GlobalGraph

# 创建全局图实例
global_graph = GlobalGraph(
    vocabulary={"natural language": 0, "machine learning": 1},
    reverse_vocabulary={0: "natural language", 1: "machine learning"},
    cooccurrence_matrix=sparse_matrix,
    easygraph_instance=eg_graph,
    metadata={"creation_time": "2024-01-01", "total_documents": 100}
)

# 获取图统计信息
node_count = global_graph.get_node_count()
edge_count = global_graph.get_edge_count()
density = global_graph.get_density()

# 添加新词组
global_graph.add_phrase("deep learning", frequency=50)

# 获取词组信息
phrase_info = global_graph.get_phrase_info("natural language")
```

**方法说明:**

- `get_node_count() -> int`: 获取节点数量
- `get_edge_count() -> int`: 获取边数量
- `get_density() -> float`: 获取图密度
- `add_phrase(phrase: str, frequency: int) -> None`: 添加新词组
- `get_phrase_info(phrase: str) -> Dict`: 获取词组信息
- `to_dict() -> Dict`: 转换为字典格式

#### ErrorHandler 类

统一错误处理和恢复机制。

```python
from semantic_coword_pipeline.core.error_handler import ErrorHandler

# 创建错误处理器
error_handler = ErrorHandler({
    'max_retries': 3,
    'retry_delay': 1.0,
    'continue_on_error': False
})

# 处理错误
try:
    # 可能出错的操作
    result = risky_operation()
except Exception as e:
    # 尝试错误恢复
    recovered_result = error_handler.handle_error(e, 'processing_context')

# 生成错误报告
error_report = error_handler.generate_error_report()
print(f"Total errors: {error_report['summary']['total_errors']}")
```

### semantic_coword_pipeline.processors

#### TextProcessor 类

文本预处理器，支持多语言文本清洗、分词和规范化。

```python
from semantic_coword_pipeline.processors.text_processor import TextProcessor

# 创建处理器
config = {
    'english_tokenizer': 'nltk',
    'chinese_tokenizer': 'jieba',
    'normalize_text': True,
    'remove_punctuation': True
}
processor = TextProcessor(config)

# 处理单个文档
processed_doc = processor.process_document(toc_document)

# 批量处理文档
processed_docs = processor.batch_process_documents([doc1, doc2, doc3])

# 获取处理统计
stats = processor.get_processing_statistics()
print(f"Processed {stats['total_documents']} documents")
```

**方法说明:**

- `process_document(doc: TOCDocument) -> ProcessedDocument`: 处理单个文档
- `batch_process_documents(docs: List[TOCDocument]) -> List[ProcessedDocument]`: 批量处理
- `detect_language(text: str) -> str`: 检测文本语言
- `normalize_text(text: str, language: str) -> str`: 文本规范化
- `tokenize(text: str, language: str) -> List[str]`: 分词处理

#### PhraseExtractor 类

词组抽取器，支持英文2-gram和中文短语抽取。

```python
from semantic_coword_pipeline.processors.phrase_extractor import PhraseExtractor

# 创建抽取器
config = {
    'ngram_size': 2,
    'min_frequency': 3,
    'statistical_filters': ['mutual_information', 't_score']
}
extractor = PhraseExtractor(config)

# 从文档抽取词组
phrases = extractor.extract_phrases_from_document(processed_doc)

# 批量抽取
all_phrases = extractor.batch_extract_phrases(processed_docs)

# 计算语料库统计
corpus_stats = extractor.calculate_corpus_statistics(all_phrases)
```

#### DynamicStopwordDiscoverer 类

动态停词发现器，基于TF-IDF自动识别低区分度词组。

```python
from semantic_coword_pipeline.processors.dynamic_stopword_discoverer import DynamicStopwordDiscoverer

# 创建发现器
config = {
    'tfidf_threshold': 0.1,
    'frequency_threshold': 0.8,
    'enable_dynamic_discovery': True
}
discoverer = DynamicStopwordDiscoverer(config)

# 发现动态停词
phrase_corpus = [doc.phrases for doc in processed_docs]
stopwords = discoverer.discover_stopwords(phrase_corpus)

# 应用停词过滤
filtered_phrases = discoverer.apply_stopword_filter(phrases, stopwords)

# 获取停词解释
explanation = discoverer.get_stopword_explanation()
```

#### GlobalGraphBuilder 类

全局图构建器，创建跨文档的共现网络。

```python
from semantic_coword_pipeline.processors.global_graph_builder import GlobalGraphBuilder

# 创建构建器
config = {
    'preserve_isolated_nodes': True,
    'edge_weight_method': 'binary',
    'window_type': 'segment'
}
builder = GlobalGraphBuilder(config)

# 构建全局图
global_graph = builder.build_global_graph(processed_docs)

# 获取图统计
stats = builder.get_graph_statistics(global_graph)
print(f"Nodes: {stats['node_count']}, Edges: {stats['edge_count']}")
```

#### StateSubgraphActivator 类

州级子图激活器，从全局图提取特定州的子图。

```python
from semantic_coword_pipeline.processors.state_subgraph_activator import StateSubgraphActivator

# 创建激活器
config = {
    'activation_method': 'reweight',
    'preserve_global_positions': True
}
activator = StateSubgraphActivator(config)

# 激活州级子图
california_docs = [doc for doc in processed_docs if doc.original_doc.state == 'California']
ca_subgraph = activator.activate_state_subgraph(global_graph, california_docs)

# 比较子图
texas_subgraph = activator.activate_state_subgraph(global_graph, texas_docs)
comparison = activator.compare_subgraphs([ca_subgraph, texas_subgraph])
```

#### DeterministicLayoutEngine 类

确定性布局引擎，生成可复现的网络可视化。

```python
from semantic_coword_pipeline.processors.deterministic_layout_engine import DeterministicLayoutEngine

# 创建布局引擎
config = {
    'algorithm': 'force_directed',
    'random_seed': 42,
    'cache_enabled': True,
    'max_iterations': 1000
}
layout_engine = DeterministicLayoutEngine(config)

# 计算布局
positions = layout_engine.compute_layout(graph, "my_graph")

# 应用可视化过滤
filtered_positions = layout_engine.apply_visualization_filter(
    positions, 
    min_degree=2, 
    max_nodes=100
)

# 更新子图位置
layout_engine.update_subgraph_positions(subgraph, positions)
```

### semantic_coword_pipeline.analyzers

#### NetworkAnalyzer 类

网络分析器，计算网络统计指标和进行对比分析。

```python
from semantic_coword_pipeline.analyzers.network_analyzer import NetworkAnalyzer

# 创建分析器
config = {
    'enable_community_detection': True,
    'centrality_measures': ['degree', 'betweenness', 'closeness']
}
analyzer = NetworkAnalyzer(config)

# 计算基础统计
basic_stats = analyzer.calculate_basic_statistics(graph)

# 计算高级指标
advanced_metrics = analyzer.calculate_advanced_metrics(graph)

# 社群检测
communities = analyzer.detect_communities(graph)

# 对比分析
comparison = analyzer.compare_network_structures([graph1, graph2, graph3])

# 跨州差异分析
state_analysis = analyzer.analyze_cross_state_differences(state_subgraphs)
```

## 管线类

### SemanticCowordPipeline 类

主管线类，协调所有处理组件。

```python
from semantic_coword_pipeline.pipeline import SemanticCowordPipeline

# 创建管线实例
pipeline = SemanticCowordPipeline(config_path='my_config.json')

# 运行完整管线
result = pipeline.run(
    input_dir='data/toc_documents/',
    output_dir='output/analysis_results/'
)

# 检查结果
print(f"Processed {result.processed_files} files")
print(f"Generated {len(result.output_files)} output files")
print(f"Global graph has {result.global_graph.get_node_count()} nodes")

# 获取州级子图
california_subgraph = result.state_subgraphs.get('California')
if california_subgraph:
    print(f"California subgraph has {california_subgraph.get_node_count()} nodes")
```

### CLIInterface 类

命令行接口，提供便捷的命令行操作。

```python
from semantic_coword_pipeline.cli import CLIInterface

# 创建CLI实例
cli = CLIInterface()

# 程序化调用处理命令
args = type('Args', (), {
    'input_dir': 'data/',
    'output_dir': 'output/',
    'config': 'config.json',
    'verbose': True,
    'dry_run': False
})()

result = cli._handle_process_command(args)
```

**命令行使用:**

```bash
# 处理文档
python -m semantic_coword_pipeline process input/ output/ --config config.json --verbose

# 创建默认配置
python -m semantic_coword_pipeline config create --output default_config.json

# 验证配置
python -m semantic_coword_pipeline config validate --config-file my_config.json

# 显示配置
python -m semantic_coword_pipeline config show --config-file my_config.json --section text_processing
```

## 工具函数

### 配置工具

```python
from semantic_coword_pipeline.pipeline import create_default_config

# 创建默认配置文件
config_file = create_default_config()
print(f"Created config file: {config_file}")
```

### 日志工具

```python
from semantic_coword_pipeline.core.logger import setup_logger

# 设置日志器
logger = setup_logger('my_component', {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': 'logs/my_component.log'
})

logger.info("Processing started")
logger.error("An error occurred", exc_info=True)
```

### 数据验证工具

```python
from semantic_coword_pipeline.core.data_models import validate_toc_json

# 验证TOC JSON数据
json_data = {
    "segment_id": "test_001",
    "title": "Test Document",
    "level": 1,
    "order": 1,
    "text": "Document content..."
}

validation_result = validate_toc_json(json_data)
if validation_result['is_valid']:
    print("Data is valid")
else:
    print(f"Validation errors: {validation_result['errors']}")
```

## 性能监控

### PerformanceMonitor 类

```python
from semantic_coword_pipeline.core.performance import PerformanceMonitor

# 创建性能监控器
monitor = PerformanceMonitor({
    'enable_profiling': True,
    'enable_memory_monitoring': True,
    'sampling_interval': 1.0
})

# 开始监控
monitor.start_monitoring()

# 监控操作
operation_id = monitor.start_operation('text_processing')
# ... 执行操作 ...
metrics = monitor.end_operation(operation_id)

# 生成性能报告
report = monitor.generate_performance_report()
monitor.save_performance_report('performance_report.json')

# 停止监控
monitor.stop_monitoring()
```

## 错误处理

### 异常类型

```python
from semantic_coword_pipeline.core.error_handler import (
    PipelineError,
    InputValidationError,
    ProcessingError,
    GraphConstructionError,
    OutputError
)

# 捕获特定错误类型
try:
    # 处理操作
    result = process_documents(docs)
except InputValidationError as e:
    print(f"Input validation failed: {e}")
except ProcessingError as e:
    print(f"Processing error: {e}")
except GraphConstructionError as e:
    print(f"Graph construction failed: {e}")
```

## 扩展开发

### 自定义处理器

```python
from semantic_coword_pipeline.processors.text_processor import TextProcessor

class CustomTextProcessor(TextProcessor):
    """自定义文本处理器示例"""
    
    def __init__(self, config):
        super().__init__(config)
        self.custom_rules = config.get('custom_rules', [])
    
    def normalize_text(self, text: str, language: str) -> str:
        """重写文本规范化方法"""
        # 调用父类方法
        normalized = super().normalize_text(text, language)
        
        # 应用自定义规则
        for rule in self.custom_rules:
            normalized = self._apply_custom_rule(normalized, rule)
        
        return normalized
    
    def _apply_custom_rule(self, text: str, rule: dict) -> str:
        """应用自定义规则"""
        # 实现自定义逻辑
        return text
```

### 自定义分析器

```python
from semantic_coword_pipeline.analyzers.network_analyzer import NetworkAnalyzer

class CustomNetworkAnalyzer(NetworkAnalyzer):
    """自定义网络分析器示例"""
    
    def calculate_custom_metrics(self, graph) -> dict:
        """计算自定义网络指标"""
        metrics = {}
        
        # 实现自定义分析逻辑
        metrics['custom_centrality'] = self._calculate_custom_centrality(graph)
        metrics['custom_clustering'] = self._calculate_custom_clustering(graph)
        
        return metrics
```

## 最佳实践

### 配置管理

1. **使用分层配置**: 将默认配置、环境配置和用户配置分层管理
2. **配置验证**: 在使用前验证配置的有效性
3. **配置追溯**: 利用配置历史功能追踪配置变更

### 错误处理

1. **统一错误处理**: 使用ErrorHandler类统一处理所有错误
2. **错误恢复**: 为关键操作提供回退策略
3. **错误记录**: 详细记录错误信息用于调试

### 性能优化

1. **启用性能监控**: 在生产环境中启用性能监控
2. **内存管理**: 对大数据集使用分批处理
3. **缓存利用**: 启用布局缓存提高重复计算效率

### 测试策略

1. **单元测试**: 为所有核心功能编写单元测试
2. **属性测试**: 使用hypothesis进行属性测试
3. **集成测试**: 测试组件间的集成
4. **性能测试**: 定期进行性能基准测试

## 版本兼容性

- **Python版本**: 要求Python 3.8+
- **依赖版本**: 详见requirements.txt
- **向后兼容**: 主版本更新可能包含破坏性变更
- **配置兼容**: 配置格式在小版本更新中保持兼容

## 支持和贡献

- **问题报告**: 通过GitHub Issues报告问题
- **功能请求**: 通过GitHub Issues提交功能请求
- **代码贡献**: 遵循项目的贡献指南
- **文档改进**: 欢迎改进文档和示例

---

*本文档随项目更新而更新，最新版本请参考项目仓库。*