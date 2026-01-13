#!/usr/bin/env python3
"""
检查项目可复现性

这个脚本检查项目是否满足可复现性要求，包括：
1. 依赖完整性
2. 数据可用性
3. 配置文件
4. 文档完整性
5. 示例数据
"""

import os
import sys
import json
from pathlib import Path
import subprocess

def check_dependencies():
    """检查依赖文件"""
    print("🔍 检查依赖文件...")
    
    issues = []
    
    # 检查requirements.txt
    if not os.path.exists('requirements.txt'):
        issues.append("❌ 缺少 requirements.txt")
    else:
        print("✅ requirements.txt 存在")
        
        # 检查关键依赖
        with open('requirements.txt', 'r') as f:
            content = f.read()
            
        required_deps = ['numpy', 'pandas', 'nltk', 'jieba', 'matplotlib', 'networkx', 'tqdm']
        missing_deps = []
        
        for dep in required_deps:
            if dep not in content:
                missing_deps.append(dep)
        
        if missing_deps:
            issues.append(f"❌ requirements.txt 缺少关键依赖: {', '.join(missing_deps)}")
        else:
            print("✅ 关键依赖完整")
    
    # 检查setup.py
    if not os.path.exists('setup.py'):
        issues.append("❌ 缺少 setup.py")
    else:
        print("✅ setup.py 存在")
    
    return issues

def check_data_availability():
    """检查数据可用性"""
    print("\\n📊 检查数据可用性...")
    
    issues = []
    
    # 检查示例数据目录
    if not os.path.exists('data'):
        issues.append("❌ 缺少 data/ 目录")
        os.makedirs('data', exist_ok=True)
        print("✅ 创建了 data/ 目录")
    else:
        print("✅ data/ 目录存在")
    
    # 检查示例数据文件
    sample_files = ['sample_data.json', 'README.md']
    
    for file in sample_files:
        file_path = os.path.join('data', file)
        if not os.path.exists(file_path):
            issues.append(f"❌ 缺少示例数据: data/{file}")
    
    # 检查test_input目录
    if not os.path.exists('test_input'):
        issues.append("❌ 缺少 test_input/ 目录")
    else:
        print("✅ test_input/ 目录存在")
    
    return issues

def check_configuration():
    """检查配置文件"""
    print("\\n⚙️ 检查配置文件...")
    
    issues = []
    
    # 检查config目录
    if not os.path.exists('config'):
        issues.append("❌ 缺少 config/ 目录")
        os.makedirs('config', exist_ok=True)
        print("✅ 创建了 config/ 目录")
    else:
        print("✅ config/ 目录存在")
    
    # 检查默认配置文件
    default_config_path = 'config/default_config.json'
    if not os.path.exists(default_config_path):
        issues.append("❌ 缺少默认配置文件")
    else:
        print("✅ 默认配置文件存在")
    
    return issues

def check_documentation():
    """检查文档完整性"""
    print("\\n📚 检查文档完整性...")
    
    issues = []
    
    # 检查README.md
    if not os.path.exists('README.md'):
        issues.append("❌ 缺少 README.md")
    else:
        print("✅ README.md 存在")
        
        # 检查README内容
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        required_sections = ['安装', '快速开始', '使用方法', '依赖']
        missing_sections = []
        
        for section in required_sections:
            if section not in readme_content:
                missing_sections.append(section)
        
        if missing_sections:
            issues.append(f"❌ README.md 缺少重要章节: {', '.join(missing_sections)}")
        else:
            print("✅ README.md 内容完整")
    
    # 检查docs目录
    if not os.path.exists('docs'):
        issues.append("❌ 缺少 docs/ 目录")
    else:
        print("✅ docs/ 目录存在")
    
    return issues

def check_entry_points():
    """检查程序入口点"""
    print("\\n🚀 检查程序入口点...")
    
    issues = []
    
    # 检查主程序文件
    main_files = ['complete_usage_guide.py', 'demo.py']
    
    for file in main_files:
        if not os.path.exists(file):
            issues.append(f"❌ 缺少主程序文件: {file}")
        else:
            print(f"✅ {file} 存在")
    
    # 检查是否可以导入主模块
    try:
        import complete_usage_guide
        print("✅ complete_usage_guide 可以导入")
    except ImportError as e:
        issues.append(f"❌ complete_usage_guide 导入失败: {e}")
    
    return issues

def check_tests():
    """检查测试文件"""
    print("\\n🧪 检查测试文件...")
    
    issues = []
    
    # 检查tests目录
    if not os.path.exists('tests'):
        issues.append("❌ 缺少 tests/ 目录")
    else:
        print("✅ tests/ 目录存在")
        
        # 检查是否有测试文件
        test_files = [f for f in os.listdir('tests') if f.startswith('test_') and f.endswith('.py')]
        if not test_files:
            issues.append("❌ tests/ 目录中没有测试文件")
        else:
            print(f"✅ 找到 {len(test_files)} 个测试文件")
    
    # 检查根目录的测试文件
    root_test_files = [f for f in os.listdir('.') if f.startswith('test_') and f.endswith('.py')]
    if root_test_files:
        print(f"✅ 根目录有 {len(root_test_files)} 个测试文件")
    
    return issues

def create_missing_files():
    """创建缺失的关键文件"""
    print("\\n🔧 创建缺失的关键文件...")
    
    # 创建示例数据文件
    if not os.path.exists('data/sample_data.json'):
        sample_data = [
            {
                "segment_id": "sample_001",
                "title": "Introduction to Network Analysis",
                "level": 1,
                "order": 1,
                "text": "Network analysis is a powerful method for understanding complex relationships in data. This approach allows researchers to visualize and analyze connections between entities.",
                "state": "CA",
                "language": "en"
            },
            {
                "segment_id": "sample_002",
                "title": "Graph Theory Fundamentals",
                "level": 2,
                "order": 2,
                "text": "Graph theory provides the mathematical foundation for network analysis. Nodes represent entities while edges represent relationships between them.",
                "state": "CA",
                "language": "en"
            },
            {
                "segment_id": "sample_003",
                "title": "网络分析基础",
                "level": 1,
                "order": 3,
                "text": "网络分析是研究复杂系统中实体间关系的重要方法。通过图论的数学工具，我们可以深入理解社会网络、生物网络等复杂系统的结构特征。",
                "state": "NY",
                "language": "zh"
            }
        ]
        
        os.makedirs('data', exist_ok=True)
        with open('data/sample_data.json', 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        print("✅ 创建了 data/sample_data.json")
    
    # 创建数据目录README
    if not os.path.exists('data/README.md'):
        data_readme = """# 数据目录

这个目录包含项目的示例数据和数据说明。

## 文件说明

- `sample_data.json` - 示例TOC格式数据，用于测试和演示
- 其他数据文件根据需要添加

## 数据格式

输入数据应为TOC JSON格式，每个文档包含以下字段：

```json
{
  "segment_id": "唯一标识符",
  "title": "段落标题",
  "level": 1,
  "order": 1,
  "text": "段落文本内容",
  "state": "州标识",
  "language": "语言标识"
}
```

## 使用方法

将您的数据文件放在此目录中，然后在主程序中指定路径即可。
"""
        with open('data/README.md', 'w', encoding='utf-8') as f:
            f.write(data_readme)
        print("✅ 创建了 data/README.md")
    
    # 创建默认配置文件
    if not os.path.exists('config/default_config.json'):
        default_config = {
            "text_processing": {
                "min_phrase_frequency": 2,
                "ngram_size": 2,
                "language_detection": "auto"
            },
            "graph_construction": {
                "edge_weight_method": "npmi",
                "min_cooccurrence_threshold": 1,
                "edge_density_reduction": 0.5
            },
            "visualization": {
                "random_seed": 42,
                "output_dpi": 300,
                "figure_size": [16, 12]
            },
            "output": {
                "base_directory": "output",
                "save_intermediate_results": True
            }
        }
        
        os.makedirs('config', exist_ok=True)
        with open('config/default_config.json', 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        print("✅ 创建了 config/default_config.json")

def update_requirements():
    """更新requirements.txt，确保包含tqdm"""
    print("\\n📦 更新requirements.txt...")
    
    with open('requirements.txt', 'r') as f:
        content = f.read()
    
    # 检查是否包含tqdm
    if 'tqdm' not in content:
        # 添加tqdm到requirements.txt
        lines = content.strip().split('\\n')
        
        # 找到合适的位置插入tqdm
        insert_index = -1
        for i, line in enumerate(lines):
            if line.startswith('# 可视化') or line.startswith('matplotlib'):
                insert_index = i
                break
        
        if insert_index == -1:
            lines.append('tqdm>=4.62.0')
        else:
            lines.insert(insert_index, 'tqdm>=4.62.0')
        
        updated_content = '\\n'.join(lines)
        
        with open('requirements.txt', 'w') as f:
            f.write(updated_content)
        
        print("✅ 添加了tqdm依赖到requirements.txt")
    else:
        print("✅ tqdm依赖已存在")

def main():
    """主函数"""
    print("🔍 项目可复现性检查")
    print("=" * 60)
    
    all_issues = []
    
    # 执行各项检查
    all_issues.extend(check_dependencies())
    all_issues.extend(check_data_availability())
    all_issues.extend(check_configuration())
    all_issues.extend(check_documentation())
    all_issues.extend(check_entry_points())
    all_issues.extend(check_tests())
    
    # 创建缺失文件
    create_missing_files()
    
    # 更新requirements
    update_requirements()
    
    # 总结
    print("\\n" + "=" * 60)
    print("📋 检查总结")
    print("=" * 60)
    
    if all_issues:
        print(f"❌ 发现 {len(all_issues)} 个问题:")
        for issue in all_issues:
            print(f"   {issue}")
        print("\\n🔧 建议修复这些问题以提高可复现性")
        return 1
    else:
        print("✅ 项目可复现性检查通过！")
        print("🎉 用户应该能够从GitHub下载后直接运行")
        return 0

if __name__ == "__main__":
    sys.exit(main())