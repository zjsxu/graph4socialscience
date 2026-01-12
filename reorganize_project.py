#!/usr/bin/env python3
"""
项目结构重组脚本

将复杂的项目结构重组为更简洁的布局，保留核心功能。
"""

import os
import shutil
from pathlib import Path

def reorganize_project():
    """重组项目结构"""
    print("开始重组项目结构...")
    
    # 创建新的简化目录结构
    new_structure = {
        'core/': [
            'semantic_coword_pipeline/core/config.py',
            'semantic_coword_pipeline/core/data_models.py',
            'semantic_coword_pipeline/core/logger.py',
            'semantic_coword_pipeline/core/error_handler.py',
            'semantic_coword_pipeline/core/performance.py'
        ],
        'processors/': [
            'semantic_coword_pipeline/processors/text_processor.py',
            'semantic_coword_pipeline/processors/phrase_extractor.py',
            'semantic_coword_pipeline/processors/dynamic_stopword_discoverer.py',
            'semantic_coword_pipeline/processors/global_graph_builder.py',
            'semantic_coword_pipeline/processors/state_subgraph_activator.py',
            'semantic_coword_pipeline/processors/deterministic_layout_engine.py',
            'semantic_coword_pipeline/processors/output_manager.py',
            'semantic_coword_pipeline/processors/batch_processor.py',
            'semantic_coword_pipeline/processors/document_generator.py',
            'semantic_coword_pipeline/processors/easygraph_interface.py'
        ],
        'analyzers/': [
            'semantic_coword_pipeline/analyzers/network_analyzer.py'
        ],
        'main/': [
            'semantic_coword_pipeline/pipeline.py',
            'semantic_coword_pipeline/cli.py',
            'semantic_coword_pipeline/__init__.py'
        ],
        'tests/': [
            'tests/test_*.py',
            'tests/conftest.py',
            'tests/__init__.py'
        ],
        'docs/': [
            'docs/user_guide.md',
            'docs/api_reference.md',
            'docs/integration_guide.md'
        ],
        'config/': [
            'config/default_config.json',
            'config/pipeline_config.json'
        ],
        'examples/': [
            'demo.py',
            'demo_*.py'
        ],
        'tools/': [
            'generate_final_quality_report.py',
            'run_performance_benchmarks.py',
            'test_batch_processing.py'
        ]
    }
    
    # 创建简化版目录
    simplified_dir = Path('simplified_project')
    if simplified_dir.exists():
        shutil.rmtree(simplified_dir)
    simplified_dir.mkdir()
    
    # 复制文件到新结构
    for new_dir, file_patterns in new_structure.items():
        target_dir = simplified_dir / new_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for pattern in file_patterns:
            if '*' in pattern:
                # 处理通配符
                base_dir = Path(pattern).parent
                pattern_name = Path(pattern).name
                if base_dir.exists():
                    for file_path in base_dir.glob(pattern_name):
                        if file_path.is_file():
                            shutil.copy2(file_path, target_dir / file_path.name)
                            print(f"复制: {file_path} -> {target_dir / file_path.name}")
            else:
                # 处理具体文件
                source_path = Path(pattern)
                if source_path.exists():
                    shutil.copy2(source_path, target_dir / source_path.name)
                    print(f"复制: {source_path} -> {target_dir / source_path.name}")
    
    # 复制重要的根目录文件
    root_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        'pytest.ini',
        'semantic_coword_analyzer.py'  # 新的整合脚本
    ]
    
    for file_name in root_files:
        source_path = Path(file_name)
        if source_path.exists():
            shutil.copy2(source_path, simplified_dir / file_name)
            print(f"复制: {source_path} -> {simplified_dir / file_name}")
    
    # 创建新的README
    create_simplified_readme(simplified_dir)
    
    # 创建快速启动脚本
    create_quick_start_script(simplified_dir)
    
    print(f"\n项目重组完成！")
    print(f"简化版项目位置: {simplified_dir.absolute()}")
    print("\n新的项目结构:")
    print_directory_tree(simplified_dir)

def create_simplified_readme(project_dir: Path):
    """创建简化版README"""
    readme_content = """# 语义共词网络分析器 - 简化版

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
"""
    
    with open(project_dir / 'README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)

def create_quick_start_script(project_dir: Path):
    """创建快速启动脚本"""
    script_content = """#!/usr/bin/env python3
\"\"\"
快速启动脚本

使用完整的语义共词网络分析管线进行分析。
如果你想要更简单的使用方式，请使用 semantic_coword_analyzer.py
\"\"\"

import sys
import os
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

def main():
    if len(sys.argv) < 3:
        print("使用方法: python quick_start.py <input_dir> <output_dir>")
        print("示例: python quick_start.py input_data/ output_results/")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    try:
        # 尝试导入完整管线
        from main.pipeline import SemanticCowordPipeline
        
        print("使用完整管线进行分析...")
        pipeline = SemanticCowordPipeline()
        result = pipeline.run(input_dir, output_dir)
        
        print(f"分析完成！处理了 {result.processed_files} 个文件")
        print(f"输出目录: {output_dir}")
        
    except ImportError as e:
        print(f"完整管线导入失败: {e}")
        print("回退到整合版分析器...")
        
        # 回退到整合版
        os.system(f"python semantic_coword_analyzer.py {input_dir} {output_dir}")

if __name__ == '__main__':
    main()
"""
    
    with open(project_dir / 'quick_start.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(project_dir / 'quick_start.py', 0o755)

def print_directory_tree(directory: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
    """打印目录树"""
    if current_depth > max_depth:
        return
    
    items = sorted(directory.iterdir(), key=lambda x: (x.is_file(), x.name))
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir() and current_depth < max_depth:
            extension = "    " if is_last else "│   "
            print_directory_tree(item, prefix + extension, max_depth, current_depth + 1)

if __name__ == '__main__':
    reorganize_project()