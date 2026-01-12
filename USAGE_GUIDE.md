# 语义共词网络分析器 - 使用指南

## 项目优化完成

我已经成功优化了项目架构并创建了整合脚本，解决了原有项目结构过于复杂的问题。

## 两种使用方式

### 方式1: 整合脚本（推荐）
最简单的使用方式，单个文件包含所有功能：

```bash
# 基本使用
python semantic_coword_analyzer.py test_input output

# 详细输出
python semantic_coword_analyzer.py test_input output --verbose

# 使用自定义配置
python semantic_coword_analyzer.py test_input output --config config.json

# 查看帮助
python semantic_coword_analyzer.py --help
```

### 方式2: 简化项目结构
重新组织的项目结构，便于开发和维护：

```bash
# 使用简化项目中的整合脚本
python simplified_project/semantic_coword_analyzer.py test_input output

# 或使用完整管线（需要先进入目录）
cd simplified_project
python quick_start.py ../test_input ../output
```

## 主要改进

### 1. 配置问题修复
- ✅ 修复了配置键缺失的问题
- ✅ 实现了配置深度合并功能
- ✅ 调整了停词发现参数，避免过度过滤

### 2. 项目结构优化
- ✅ 创建了单文件整合脚本 `semantic_coword_analyzer.py`
- ✅ 重组了复杂的文件结构到 `simplified_project/`
- ✅ 保持了所有核心功能

### 3. 功能验证
- ✅ 支持中英文文本处理
- ✅ 动态停词发现
- ✅ 词组级网络构建
- ✅ 州级子图分析
- ✅ 多格式输出（JSON, CSV）

## 测试结果

使用测试数据运行成功：
- 处理文档数: 6/6
- 抽取词组数: 121 (唯一: 114)
- 全局图规模: 114 节点, 1327 边
- 州级子图数: 3
- 处理时间: ~0.8 秒

## 输出文件

系统会生成以下输出：
- `global_graph.json`: 全局共现网络
- `global_nodes.csv`: 全局节点表
- `global_edges.csv`: 全局边表
- `states/`: 各州子图数据
- `analysis_statistics.json`: 分析统计报告
- `visualizations/`: 可视化图像（如果matplotlib可用）

## 依赖说明

### 必需依赖
- Python 3.8+

### 可选依赖（已自动检测）
- nltk: 英文文本处理 ✅
- jieba: 中文分词 ✅
- scipy: 矩阵运算 ✅
- matplotlib: 可视化 ✅
- easygraph: 高级图分析 ✅

## 下一步建议

1. **实际使用**: 用你的真实数据测试整合脚本
2. **配置调整**: 根据需要修改配置参数
3. **功能扩展**: 基于简化项目结构添加新功能
4. **部署**: 使用整合脚本进行生产部署

整合脚本现在可以直接用于实际运行，大大简化了使用复杂度！