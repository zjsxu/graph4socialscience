# 语义增强共词网络分析管线

## 概述

本项目实现了一个完整的语义增强共词网络分析管线，采用"总图优先，州级激活"的两阶段构建策略。系统以词组/短语为节点单位，通过动态停词发现和确定性布局确保可复现的网络分析结果。

## 主要特性

- **总图优先架构**: 先构建全局共现网络，再通过激活掩码提取州级子图
- **词组级节点**: 使用2-gram和短语作为网络节点，提供更有意义的语义表示
- **动态停词发现**: 自动识别高频低区分度词组，提升网络质量
- **确定性布局**: 使用固定种子确保可复现的网络可视化
- **可追溯处理**: 完整记录处理过程，支持学术研究需求
- **EasyGraph集成**: 与EasyGraph框架兼容，支持图融合分析

## 系统架构

```
TOC JSON输入 → 文本预处理 → 词组抽取 → 动态停词发现 → 总图构建 → 州级子图激活 → 确定性布局 → 网络分析 → EasyGraph兼容输出
```

## 安装

### 环境要求

- Python 3.8+
- EasyGraph >= 1.0
- NLTK >= 3.6
- jieba >= 0.42 (中文文本处理)
- hypothesis >= 6.0 (属性测试)

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd semantic-coword-pipeline
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 安装EasyGraph (从本地Easy-Graph目录)
```bash
cd Easy-Graph
pip install -e .
cd ..
```

4. 下载NLTK数据
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

5. 安装本项目
```bash
pip install -e .
```

## 快速开始

### 基本使用

```python
from semantic_coword_pipeline import Config, setup_logger
from semantic_coword_pipeline.core.data_models import TOCDocument

# 初始化配置和日志
config = Config()
logger = setup_logger('pipeline', config.get_section('logging'))

# 创建TOC文档
doc_data = {
    "segment_id": "seg_001",
    "title": "Introduction",
    "level": 1,
    "order": 1,
    "text": "This is sample text for analysis.",
    "state": "CA"
}

toc_doc = TOCDocument.from_json(doc_data)
print(f"Created document: {toc_doc.title}")
```

### 配置管理

```python
from semantic_coword_pipeline.core.config import Config

# 加载默认配置
config = Config()

# 获取配置值
ngram_size = config.get('text_processing.ngram_size')  # 2
random_seed = config.get('layout_engine.random_seed')  # 42

# 修改配置
config.set('text_processing.ngram_size', 3)
config.set('output.base_path', '/custom/output/path')

# 保存配置
config.save_to_file('my_config.json')
```

### 错误处理

```python
from semantic_coword_pipeline.core.error_handler import ErrorHandler

# 创建错误处理器
error_handler = ErrorHandler()

try:
    # 可能出错的操作
    result = some_processing_function()
except Exception as e:
    # 统一错误处理
    recovered_result = error_handler.handle_error(e, 'processing_context')
```

## 项目结构

```
semantic_coword_pipeline/
├── core/                   # 核心模块
│   ├── data_models.py     # 数据模型定义
│   ├── config.py          # 配置管理
│   ├── error_handler.py   # 错误处理
│   └── logger.py          # 日志系统
├── processors/            # 处理器模块 (待实现)
├── analyzers/            # 分析器模块 (待实现)
└── utils/                # 工具模块 (待实现)

tests/                    # 测试模块
├── test_data_models.py   # 数据模型测试
├── test_config.py        # 配置管理测试
├── test_error_handler.py # 错误处理测试
└── test_logger.py        # 日志系统测试

config/                   # 配置文件
└── default_config.json   # 默认配置

docs/                     # 文档 (待创建)
```

## 测试

项目使用pytest和hypothesis进行单元测试和属性测试：

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_data_models.py

# 运行属性测试
pytest -m property

# 生成覆盖率报告
pytest --cov=semantic_coword_pipeline --cov-report=html
```

## 开发

### 代码风格

项目使用以下工具确保代码质量：

```bash
# 代码格式化
black semantic_coword_pipeline/

# 代码检查
flake8 semantic_coword_pipeline/

# 类型检查
mypy semantic_coword_pipeline/
```

### 添加新功能

1. 在相应模块中实现功能
2. 添加对应的单元测试和属性测试
3. 更新文档和配置
4. 确保所有测试通过

## 配置说明

### 主要配置项

- `text_processing`: 文本处理配置
  - `ngram_size`: N-gram大小 (默认: 2)
  - `min_phrase_frequency`: 最小词组频率 (默认: 3)

- `layout_engine`: 布局引擎配置
  - `random_seed`: 随机种子 (默认: 42)
  - `algorithm`: 布局算法 (默认: "force_directed")

- `output`: 输出配置
  - `base_path`: 输出路径 (默认: "output/")
  - `export_formats`: 导出格式 (默认: ["json", "graphml", "csv"])

完整配置说明请参考 `config/default_config.json`。

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues: <repository-url>/issues
- 邮箱: team@example.com