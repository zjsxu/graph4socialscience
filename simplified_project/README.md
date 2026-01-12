# 语义共词网络分析器 - 简化版

这是语义共词网络分析管线的简化版本，重组了文件结构以便于使用和维护。

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行分析（推荐）
使用整合版脚本（最简单）：
```bash
python semantic_coword_analyzer.py input_data/ output_results/
```

### 3. 运行分析（完整版）
使用完整管线：
```bash
python quick_start.py input_data/ output_results/
```

## 项目结构

```
simplified_project/
├── semantic_coword_analyzer.py    # 整合版分析器（推荐使用）
├── quick_start.py                 # 快速启动脚本
├── core/                          # 核心模块
├── processors/                    # 处理器模块
├── analyzers/                     # 分析器模块
├── main/                          # 主程序
├── tests/                         # 测试文件
├── docs/                          # 文档
├── config/                        # 配置文件
├── examples/                      # 示例脚本
└── tools/                         # 工具脚本
```

## 使用说明

### 输入数据格式
输入目录应包含JSON格式的TOC文档：
```json
[
  {
    "segment_id": "doc_001_seg_001",
    "title": "Introduction",
    "level": 1,
    "order": 1,
    "text": "This is the introduction text...",
    "state": "California"
  }
]
```

### 输出结果
系统会生成以下输出：
- `global_graph.json`: 全局共现网络
- `global_nodes.csv`: 全局节点表
- `global_edges.csv`: 全局边表
- `states/`: 各州子图数据
- `analysis_statistics.json`: 分析统计报告
- `visualizations/`: 可视化图像（如果matplotlib可用）

## 主要特性

- ✅ 支持中英文文本处理
- ✅ 动态停词发现
- ✅ 词组级网络构建
- ✅ 州级子图分析
- ✅ 多格式输出（JSON, CSV）
- ✅ 可视化生成
- ✅ 完整的错误处理
- ✅ 详细的日志记录

## 依赖说明

### 必需依赖
- Python 3.8+

### 可选依赖（推荐安装）
- nltk: 英文文本处理
- jieba: 中文分词
- scipy: 矩阵运算
- matplotlib: 可视化
- easygraph: 高级图分析

## 故障排除

如果遇到依赖问题，可以逐步安装：
```bash
# 基础功能
pip install numpy

# 文本处理
pip install nltk jieba

# 科学计算
pip install scipy matplotlib

# 图分析（可选）
pip install Python-EasyGraph
```

## 版本信息
- 版本: 1.0.0
- 最后更新: 2026-01-11
