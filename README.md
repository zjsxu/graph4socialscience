# Graph4SocialScience - 语义增强共词网络分析管线

Group Repo for the OpenRank Contest, for the group "graph4socialscience", by Prof. Zixi Chen, Jingsen Zhang, Zeqiang Wang.

## 概述

本项目实现了一个完整的语义增强共词网络分析管线，采用"总图优先，州级激活"的两阶段构建策略。系统以词组/短语为节点单位，通过动态停词发现和确定性布局确保可复现的网络分析结果。

## 主要特性

- **总图优先架构**: 先构建全局共现网络，再通过激活掩码提取州级子图
- **词组级节点**: 使用2-gram和短语作为网络节点，提供更有意义的语义表示
- **动态停词发现**: 自动识别高频低区分度词组，提升网络质量
- **确定性布局**: 使用固定种子确保可复现的网络可视化
- **可追溯处理**: 完整记录处理过程，支持学术研究需求
- **EasyGraph集成**: 与EasyGraph框架兼容，支持图融合分析
- **进度可视化**: 全流程tqdm进度条显示
- **图数据导出**: 完整的节点、边和处理参数导出功能

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
- tqdm >= 4.0 (进度条显示)
- networkx >= 2.5
- matplotlib >= 3.3
- numpy >= 1.19
- pandas >= 1.2

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/zjsxu/graph4socialscience.git
cd graph4socialscience
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

## 快速开始

### 基本使用

运行主程序：

```bash
python complete_usage_guide.py
```

程序提供以下功能：

1. **文本预处理** - 清理和标准化输入文本
2. **词组抽取** - 提取有意义的2-gram和短语
3. **动态停词发现** - 自动识别需要过滤的高频词组
4. **全局图构建** - 构建完整的共现网络
5. **子图激活** - 按州提取特定子图
6. **网络可视化** - 生成可读的主题网络图
7. **图数据导出** - 导出完整的图结构和数据

### 使用方法

#### 方法1: 交互式菜单

```bash
python complete_usage_guide.py
```

按照菜单提示依次执行：
1. 选择输入目录 (1.1)
2. 设置输出目录 (1.2)
3. 清理文本 (2.1)
4. 提取词组 (3.2)
5. 构建全局图 (4.1)
6. 激活子图 (5.1)
7. 生成可视化 (6.1)
8. 导出结果 (6.3)

#### 方法2: 自动化脚本

```bash
# 使用预配置路径自动运行
python run_pipeline_with_memo_data.py

# 或使用快速设置
python quick_pipeline_setup.py
```

#### 方法3: 使用示例数据

```bash
# 运行演示
python demo.py

# 或使用简化指南
python simple_usage_guide.py
```

### 数据格式

输入数据应为TOC JSON格式：

```json
{
  "segment_id": "seg_001",
  "title": "Introduction", 
  "level": 1,
  "order": 1,
  "text": "This is sample text for analysis.",
  "state": "CA"
}
```

### 输出结果

- **清理文本**: `output_dir/cleaned_text/`
- **词组数据**: `output_dir/phrases/`
- **停词列表**: `output_dir/stopwords/`
- **全局图**: `output_dir/global_graph/`
- **子图**: `output_dir/subgraphs/`
- **可视化**: `output_dir/visualizations/`
- **图分析**: `output_dir/graph_analysis/` (功能6.4)

## 项目结构

```
graph4socialscience/
├── complete_usage_guide.py    # 主程序入口
├── semantic_coword_pipeline/   # 核心管线模块
│   ├── core/                  # 核心功能
│   ├── processors/            # 数据处理器
│   └── analyzers/             # 网络分析器
├── tests/                     # 测试模块
├── config/                    # 配置文件
├── docs/                      # 文档
├── data/                      # 示例数据
├── Easy-Graph/                # EasyGraph框架
└── requirements.txt           # 依赖列表
```

## 功能详解

### 1. 文本预处理
- 清理HTML标签和特殊字符
- 标准化空白字符
- 保留文档结构信息

### 2. 词组抽取
- 基于NLTK的2-gram提取
- 频率过滤和质量评估
- 支持中英文混合处理

### 3. 动态停词发现
- 基于TF-IDF的低区分度词组识别
- 自适应阈值设定
- 保留语义重要词组

### 4. 全局图构建
- 共现矩阵计算
- 边权重标准化
- 社区检测和中心性分析
- 密度过滤（从98%降至5%）

### 5. 子图激活
- 基于州标签的节点激活
- 保持全局图结构一致性
- 支持孤立节点保留

### 6. 网络可视化
- 社区感知布局
- 角色区分显示（核心/外围）
- 选择性边渲染
- 高分辨率导出（300 DPI）

### 7. 图数据导出 (功能6.4)
- 完整节点属性导出
- 边权重和关系导出
- 处理参数记录
- 支持全局图和子图选择

## 测试

运行测试套件：

```bash
# 运行所有测试
pytest

# 测试特定功能
python test_6_4_with_toc_doc.py  # 测试6.4功能
python final_visualization_test.py  # 测试可视化
python test_structural_improvements.py  # 测试结构改进
```

## 性能优化

- **边处理优化**: 1000x性能提升
- **内存管理**: 大图处理优化
- **进度显示**: 实时处理状态
- **批量处理**: 支持大规模数据

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 联系方式

- GitHub Issues: https://github.com/zjsxu/graph4socialscience/issues
- 项目团队: Prof. Zixi Chen, Jingsen Zhang, Zeqiang Wang
