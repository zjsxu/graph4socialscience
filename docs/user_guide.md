# 用户指南

## 概述

语义增强共词网络分析管线是一个专门用于政策/法规文档分析的工具，采用"总图优先，州级激活"的两阶段构建策略，以词组/短语为节点单位，通过动态停词发现和确定性布局确保可复现的网络分析结果。

## 快速开始

### 1. 安装系统

#### 环境要求

- Python 3.8 或更高版本
- 至少 4GB 内存
- 2GB 可用磁盘空间

#### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd semantic-coword-pipeline

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装EasyGraph（如果需要）
cd Easy-Graph
pip install -e .
cd ..

# 5. 下载NLTK数据
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 6. 安装本项目
pip install -e .
```

#### 验证安装

```bash
# 运行测试验证安装
python -m pytest tests/test_config.py -v

# 检查CLI是否可用
python -m semantic_coword_pipeline --help
```

### 2. 准备数据

#### 输入数据格式

系统接受TOC分段的JSON文档，每个文档包含以下字段：

```json
{
  "segment_id": "ca_001",
  "title": "California Environmental Policy Section 1",
  "level": 1,
  "order": 1,
  "text": "Natural language processing and machine learning algorithms are essential for environmental data analysis. Statistical methods help identify pollution patterns and climate change indicators.",
  "state": "California"
}
```

**字段说明:**

- `segment_id`: 文档片段的唯一标识符
- `title`: 文档片段标题
- `level`: 文档层级（1为顶级，2为二级等）
- `order`: 在同级中的顺序
- `text`: 文档正文内容（主要分析对象）
- `state`: 所属州/地区（用于州级分析）

#### 数据准备示例

```python
# 创建测试数据
import json
from pathlib import Path

# 准备输入目录
input_dir = Path("data/input")
input_dir.mkdir(parents=True, exist_ok=True)

# 创建示例文档
documents = [
    {
        "segment_id": "ca_001",
        "title": "California Environmental Standards",
        "level": 1,
        "order": 1,
        "text": "Environmental protection requires comprehensive monitoring systems. Data analysis and statistical modeling provide insights into pollution trends and climate impacts.",
        "state": "California"
    },
    {
        "segment_id": "tx_001", 
        "title": "Texas Energy Regulations",
        "level": 1,
        "order": 1,
        "text": "Energy efficiency standards and renewable energy targets drive policy development. Machine learning applications optimize energy distribution and consumption patterns.",
        "state": "Texas"
    }
]

# 保存文档
for i, doc in enumerate(documents):
    doc_file = input_dir / f"document_{i+1}.json"
    with open(doc_file, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

print(f"Created {len(documents)} test documents in {input_dir}")
```

### 3. 基本使用

#### 命令行使用

```bash
# 基本处理命令
python -m semantic_coword_pipeline process data/input/ output/ --verbose

# 使用自定义配置
python -m semantic_coword_pipeline process data/input/ output/ --config my_config.json

# 启用性能监控
python -m semantic_coword_pipeline process data/input/ output/ --profile --memory-monitor

# 并行处理
python -m semantic_coword_pipeline process data/input/ output/ --parallel 4
```

#### Python API使用

```python
from semantic_coword_pipeline import SemanticCowordPipeline
from semantic_coword_pipeline.core.config import Config

# 创建和配置管线
config = Config()
config.set('text_processing.ngram_size', 2)
config.set('layout_engine.random_seed', 42)

pipeline = SemanticCowordPipeline()
pipeline.config = config

# 运行分析
result = pipeline.run('data/input/', 'output/')

# 检查结果
print(f"处理了 {result.processed_files} 个文件")
print(f"生成了 {len(result.output_files)} 个输出文件")
print(f"全局图包含 {result.global_graph.get_node_count()} 个节点")

# 访问州级子图
for state, subgraph in result.state_subgraphs.items():
    print(f"{state}: {subgraph.get_node_count()} 个节点")
```

## 配置管理

### 配置文件结构

```json
{
  "text_processing": {
    "english_tokenizer": "nltk",
    "chinese_tokenizer": "jieba", 
    "ngram_size": 2,
    "min_phrase_frequency": 3,
    "normalize_text": true,
    "remove_punctuation": true,
    "convert_to_lowercase": true
  },
  "stopword_discovery": {
    "tfidf_threshold": 0.1,
    "frequency_threshold": 0.8,
    "enable_dynamic_discovery": true,
    "min_document_frequency": 2
  },
  "graph_construction": {
    "preserve_isolated_nodes": true,
    "edge_weight_method": "binary",
    "window_type": "segment",
    "min_cooccurrence_count": 1,
    "use_sparse_matrix": true
  },
  "layout_engine": {
    "algorithm": "force_directed",
    "random_seed": 42,
    "cache_enabled": true,
    "max_iterations": 1000,
    "convergence_threshold": 1e-6
  },
  "output": {
    "base_path": "output/",
    "generate_visualizations": true,
    "export_formats": ["json", "graphml", "csv"],
    "save_intermediate_results": true
  }
}
```

### 配置管理命令

```bash
# 创建默认配置文件
python -m semantic_coword_pipeline config create --output my_config.json

# 验证配置文件
python -m semantic_coword_pipeline config validate --config-file my_config.json

# 显示配置内容
python -m semantic_coword_pipeline config show --config-file my_config.json

# 显示特定配置节
python -m semantic_coword_pipeline config show --config-file my_config.json --section text_processing
```

### 常用配置调整

#### 文本处理配置

```python
# 调整N-gram大小
config.set('text_processing.ngram_size', 3)  # 使用3-gram

# 调整最小词组频率
config.set('text_processing.min_phrase_frequency', 5)  # 提高质量阈值

# 启用/禁用文本规范化
config.set('text_processing.normalize_text', False)
```

#### 停词发现配置

```python
# 调整TF-IDF阈值
config.set('stopword_discovery.tfidf_threshold', 0.05)  # 更严格的过滤

# 调整频率阈值
config.set('stopword_discovery.frequency_threshold', 0.9)  # 更高的频率要求

# 禁用动态停词发现
config.set('stopword_discovery.enable_dynamic_discovery', False)
```

#### 图构建配置

```python
# 使用加权边
config.set('graph_construction.edge_weight_method', 'frequency')

# 调整共现阈值
config.set('graph_construction.min_cooccurrence_count', 3)

# 禁用孤立节点保留
config.set('graph_construction.preserve_isolated_nodes', False)
```

#### 布局引擎配置

```python
# 使用层级布局
config.set('layout_engine.algorithm', 'hierarchical')

# 调整迭代次数
config.set('layout_engine.max_iterations', 2000)

# 禁用布局缓存
config.set('layout_engine.cache_enabled', False)
```

## 高级功能

### 1. 多语言文档处理

系统支持英文和中文文档的混合处理：

```python
# 中英文混合文档示例
mixed_docs = [
    {
        "segment_id": "en_001",
        "title": "English Policy Document",
        "text": "Natural language processing and machine learning algorithms...",
        "state": "California"
    },
    {
        "segment_id": "cn_001", 
        "title": "中文政策文档",
        "text": "自然语言处理和机器学习算法在政策分析中的应用...",
        "state": "Beijing"
    }
]

# 系统会自动检测语言并应用相应的处理策略
```

### 2. 州级对比分析

```python
from semantic_coword_pipeline.analyzers.network_analyzer import NetworkAnalyzer

# 创建网络分析器
analyzer = NetworkAnalyzer({
    'enable_community_detection': True,
    'centrality_measures': ['degree', 'betweenness', 'closeness']
})

# 对比不同州的网络结构
state_graphs = {
    'California': california_subgraph,
    'Texas': texas_subgraph,
    'New York': newyork_subgraph
}

comparison_results = analyzer.compare_network_structures(list(state_graphs.values()))

# 分析跨州差异
cross_state_analysis = analyzer.analyze_cross_state_differences(state_graphs)

print("州级网络对比结果:")
for state, metrics in cross_state_analysis.items():
    print(f"{state}: {metrics['node_count']} 节点, {metrics['edge_count']} 边")
```

### 3. 自定义词组抽取

```python
from semantic_coword_pipeline.processors.phrase_extractor import PhraseExtractor

# 创建自定义抽取器
config = {
    'ngram_size': 3,  # 使用3-gram
    'min_frequency': 5,
    'statistical_filters': ['mutual_information', 't_score', 'chi_square'],
    'custom_patterns': [
        r'\b(?:machine|deep|artificial)\s+(?:learning|intelligence)\b',
        r'\b(?:natural|computational)\s+language\s+processing\b'
    ]
}

extractor = PhraseExtractor(config)

# 抽取词组
phrases = extractor.extract_phrases_from_document(processed_doc)

# 应用统计过滤
filtered_phrases = extractor.apply_statistical_filters(phrases, corpus_stats)
```

### 4. 确定性可视化

```python
from semantic_coword_pipeline.processors.deterministic_layout_engine import DeterministicLayoutEngine

# 创建确定性布局引擎
layout_config = {
    'algorithm': 'force_directed',
    'random_seed': 42,  # 确保可复现性
    'cache_enabled': True,
    'max_iterations': 1000,
    'spring_constant': 1.0,
    'repulsion_strength': 1.0
}

layout_engine = DeterministicLayoutEngine(layout_config)

# 计算布局
positions = layout_engine.compute_layout(global_graph.easygraph_instance, "global_graph")

# 应用可视化过滤
filtered_positions = layout_engine.apply_visualization_filter(
    positions,
    min_degree=2,      # 最小度数
    max_nodes=100,     # 最大节点数
    edge_threshold=0.1 # 边权重阈值
)

# 保存布局
layout_engine.save_layout(filtered_positions, "output/layouts/global_layout.json")
```

### 5. 性能监控和优化

```python
from semantic_coword_pipeline.core.performance import PerformanceMonitor

# 启用性能监控
monitor = PerformanceMonitor({
    'enable_profiling': True,
    'enable_memory_monitoring': True,
    'sampling_interval': 1.0
})

# 在管线中使用监控
pipeline = SemanticCowordPipeline()
pipeline.performance_monitor = monitor

# 运行并监控
monitor.start_monitoring()
result = pipeline.run('data/input/', 'output/')
monitor.stop_monitoring()

# 生成性能报告
performance_report = monitor.generate_performance_report()
monitor.save_performance_report('output/performance_report.json')

# 查看优化建议
recommendations = performance_report['optimization_recommendations']
for rec in recommendations:
    print(f"{rec['type']}: {rec['recommendation']}")
```

## 输出文件说明

### 目录结构

```
output/
├── graphs/                    # 图数据文件
│   ├── global_graph.json     # 全局图数据
│   ├── global_graph.graphml  # GraphML格式
│   └── state_subgraphs/      # 州级子图
│       ├── California.json
│       └── Texas.json
├── statistics/               # 统计数据
│   ├── vocabulary.json       # 词表统计
│   ├── phrase_frequencies.json
│   ├── tfidf_scores.json
│   └── network_metrics.json
├── visualizations/           # 可视化文件
│   ├── global_network.png
│   ├── state_comparisons.png
│   └── layouts/
│       └── cached_positions.json
├── documentation/            # 文档和报告
│   ├── experiment_report.md
│   ├── technical_choices.json
│   └── processing_trace.json
└── cache/                   # 缓存文件
    ├── stopwords.json
    └── intermediate_results/
```

### 主要输出文件

#### 1. 全局图数据 (global_graph.json)

```json
{
  "metadata": {
    "creation_time": "2024-01-01T12:00:00",
    "total_documents": 150,
    "total_phrases": 2500,
    "processing_time": 45.2
  },
  "vocabulary": {
    "natural language": 0,
    "machine learning": 1,
    "data analysis": 2
  },
  "statistics": {
    "node_count": 2500,
    "edge_count": 8750,
    "density": 0.0028,
    "isolated_nodes": 125
  },
  "cooccurrence_data": "cooccurrence_matrix.npz"
}
```

#### 2. 网络统计 (network_metrics.json)

```json
{
  "global_metrics": {
    "node_count": 2500,
    "edge_count": 8750,
    "density": 0.0028,
    "average_degree": 7.0,
    "clustering_coefficient": 0.45,
    "connected_components": 12,
    "largest_component_size": 2100
  },
  "state_metrics": {
    "California": {
      "node_count": 850,
      "edge_count": 2100,
      "density": 0.0058
    },
    "Texas": {
      "node_count": 720,
      "edge_count": 1650,
      "density": 0.0064
    }
  },
  "centrality_measures": {
    "top_degree_nodes": [
      {"phrase": "machine learning", "degree": 45},
      {"phrase": "data analysis", "degree": 38}
    ],
    "top_betweenness_nodes": [
      {"phrase": "statistical methods", "betweenness": 0.12}
    ]
  }
}
```

#### 3. 实验报告 (experiment_report.md)

自动生成的Markdown格式报告，包含：

- 实验配置和参数
- 处理步骤和时间
- 技术选择说明
- 结果统计和分析
- 对比分析结果
- 可视化图表

#### 4. 词组频率统计 (phrase_frequencies.json)

```json
{
  "total_phrases": 2500,
  "frequency_distribution": {
    "machine learning": {
      "frequency": 156,
      "document_frequency": 89,
      "tfidf_score": 0.234
    },
    "natural language": {
      "frequency": 142,
      "document_frequency": 76,
      "tfidf_score": 0.198
    }
  },
  "statistics": {
    "mean_frequency": 12.5,
    "median_frequency": 8.0,
    "std_frequency": 15.2
  }
}
```

## 故障排除

### 常见问题

#### 1. 内存不足错误

**问题**: 处理大量文档时出现内存不足

**解决方案**:
```python
# 调整批处理大小
config.set('performance.batch_size', 500)  # 减少批处理大小

# 启用稀疏矩阵
config.set('graph_construction.use_sparse_matrix', True)

# 禁用内存监控
config.set('performance.enable_memory_monitoring', False)
```

#### 2. 处理速度慢

**问题**: 文档处理速度过慢

**解决方案**:
```python
# 启用并行处理
config.set('performance.enable_parallel_processing', True)
config.set('performance.max_workers', 4)

# 调整词组频率阈值
config.set('text_processing.min_phrase_frequency', 5)

# 禁用可视化生成
config.set('output.generate_visualizations', False)
```

#### 3. 中文分词问题

**问题**: 中文文档分词效果不佳

**解决方案**:
```python
# 使用自定义词典
import jieba
jieba.load_userdict('custom_dict.txt')

# 调整中文处理参数
config.set('text_processing.chinese_tokenizer', 'jieba')
config.set('text_processing.chinese_custom_dict', 'custom_dict.txt')
```

#### 4. EasyGraph导入错误

**问题**: 无法导入EasyGraph模块

**解决方案**:
```bash
# 重新安装EasyGraph
cd Easy-Graph
pip install -e .

# 或使用备用图库
pip install networkx
```

#### 5. 配置验证失败

**问题**: 配置文件验证不通过

**解决方案**:
```bash
# 检查配置错误
python -m semantic_coword_pipeline config validate --config-file my_config.json

# 重新生成默认配置
python -m semantic_coword_pipeline config create --output new_config.json
```

### 调试技巧

#### 1. 启用详细日志

```python
config.set('logging.level', 'DEBUG')
config.set('logging.file_path', 'logs/debug.log')
```

#### 2. 使用干运行模式

```bash
python -m semantic_coword_pipeline process data/input/ output/ --dry-run
```

#### 3. 分步调试

```python
from semantic_coword_pipeline.processors import TextProcessor, PhraseExtractor

# 单独测试文本处理
processor = TextProcessor(config.get_section('text_processing'))
processed_doc = processor.process_document(test_doc)
print(f"Tokens: {len(processed_doc.tokens)}")

# 单独测试词组抽取
extractor = PhraseExtractor(config.get_section('phrase_extraction'))
phrases = extractor.extract_phrases_from_document(processed_doc)
print(f"Phrases: {len(phrases)}")
```

#### 4. 性能分析

```python
# 启用性能分析
config.set('performance.enable_profiling', True)

# 查看性能瓶颈
performance_report = monitor.generate_performance_report()
bottlenecks = performance_report['optimization_recommendations']
```

## 最佳实践

### 1. 数据准备

- **数据质量**: 确保输入文档格式正确，文本内容完整
- **文档数量**: 建议每个州至少有10个文档以获得有意义的分析结果
- **文本长度**: 每个文档片段建议包含50-500个词
- **编码格式**: 使用UTF-8编码保存所有文件

### 2. 配置优化

- **内存管理**: 根据可用内存调整批处理大小
- **处理速度**: 在质量和速度之间找到平衡点
- **可复现性**: 始终设置随机种子确保结果可复现
- **缓存利用**: 启用缓存减少重复计算

### 3. 结果解释

- **网络密度**: 密度过高可能表明停词过滤不够严格
- **孤立节点**: 适量的孤立节点是正常的，过多可能表明参数设置问题
- **社群结构**: 清晰的社群结构表明主题分离良好
- **中心性指标**: 关注高中心性节点，它们通常代表重要概念

### 4. 性能优化

- **分批处理**: 对大数据集使用分批处理
- **并行计算**: 在多核系统上启用并行处理
- **内存监控**: 定期监控内存使用避免溢出
- **缓存策略**: 合理使用缓存提高重复分析效率

### 5. 质量保证

- **数据验证**: 处理前验证所有输入数据
- **结果检查**: 检查输出文件的完整性和合理性
- **对比验证**: 使用已知数据集验证分析结果
- **文档记录**: 详细记录处理参数和结果

---

*本指南将随着系统更新而持续改进，如有问题请参考API文档或联系开发团队。*