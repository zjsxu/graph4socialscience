#!/usr/bin/env python3
"""
自动安装脚本

这个脚本自动化项目的安装过程，包括依赖安装、环境配置和初始化设置。
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    
    if sys.version_info < (3, 8):
        print(f"❌ Python版本过低: {sys.version}")
        print("需要Python 3.8或更高版本")
        return False
    
    print(f"✅ Python版本: {sys.version}")
    return True

def install_requirements():
    """安装Python依赖"""
    print("\\n📦 安装Python依赖...")
    
    try:
        # 升级pip
        print("📈 升级pip...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                      check=True, capture_output=True)
        print("✅ pip升级完成")
        
        # 安装requirements.txt中的依赖
        if os.path.exists('requirements.txt'):
            print("📋 安装requirements.txt中的依赖...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                          check=True)
            print("✅ 依赖安装完成")
        else:
            print("⚠️ 未找到requirements.txt，手动安装核心依赖...")
            core_deps = [
                'numpy>=1.20.0',
                'pandas>=1.3.0',
                'nltk>=3.6',
                'jieba>=0.42',
                'matplotlib>=3.3.0',
                'networkx>=2.6.0',
                'tqdm>=4.62.0',
                'pytest>=6.0.0'
            ]
            
            for dep in core_deps:
                print(f"📦 安装 {dep}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', dep], check=True)
            
            print("✅ 核心依赖安装完成")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def install_easygraph():
    """安装EasyGraph"""
    print("\\n🔗 安装EasyGraph...")
    
    easygraph_dir = Path('Easy-Graph')
    
    if easygraph_dir.exists():
        try:
            print("📁 从本地Easy-Graph目录安装...")
            original_dir = os.getcwd()
            os.chdir(easygraph_dir)
            
            # 安装EasyGraph
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], check=True)
            
            os.chdir(original_dir)
            print("✅ EasyGraph安装完成")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ EasyGraph安装失败: {e}")
            os.chdir(original_dir)
            return False
    else:
        print("⚠️ 未找到Easy-Graph目录，跳过EasyGraph安装")
        print("   如需EasyGraph功能，请手动安装或克隆Easy-Graph仓库")
        return True

def download_nltk_data():
    """下载NLTK数据"""
    print("\\n📚 下载NLTK数据...")
    
    try:
        import nltk
        
        # 设置NLTK数据路径
        nltk_data_dir = Path.home() / 'nltk_data'
        if not nltk_data_dir.exists():
            nltk_data_dir.mkdir(parents=True)
        
        # 下载必要的NLTK数据
        nltk_datasets = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
        
        for dataset in nltk_datasets:
            try:
                print(f"📥 下载 {dataset}...")
                nltk.download(dataset, quiet=True)
                print(f"✅ {dataset} 下载完成")
            except Exception as e:
                print(f"⚠️ {dataset} 下载失败: {e}")
        
        print("✅ NLTK数据下载完成")
        return True
        
    except ImportError:
        print("❌ NLTK未安装，无法下载数据")
        return False
    except Exception as e:
        print(f"❌ NLTK数据下载失败: {e}")
        return False

def create_directories():
    """创建必要的目录结构"""
    print("\\n📁 创建目录结构...")
    
    directories = [
        'data',
        'config',
        'output',
        'logs',
        'temp'
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print(f"✅ 创建目录: {directory}")
        else:
            print(f"📁 目录已存在: {directory}")
    
    return True

def create_config_files():
    """创建配置文件"""
    print("\\n⚙️ 创建配置文件...")
    
    # 创建默认配置
    config_dir = Path('config')
    default_config_path = config_dir / 'default_config.json'
    
    if not default_config_path.exists():
        import json
        
        default_config = {
            "text_processing": {
                "min_phrase_frequency": 2,
                "ngram_size": 2,
                "language_detection": "auto",
                "normalize_text": True,
                "remove_punctuation": True,
                "convert_to_lowercase": True
            },
            "graph_construction": {
                "edge_weight_method": "npmi",
                "min_cooccurrence_threshold": 1,
                "edge_density_reduction": 0.5,
                "preserve_isolated_nodes": True,
                "sliding_window_size": 5
            },
            "visualization": {
                "random_seed": 42,
                "output_dpi": 300,
                "figure_size": [16, 12],
                "layout_algorithm": "spring_deterministic"
            },
            "output": {
                "base_directory": "output",
                "save_intermediate_results": True,
                "generate_visualizations": True,
                "export_formats": ["json", "graphml", "csv"]
            },
            "scientific_optimization": {
                "semantic_weighting": "npmi",
                "sparsification_method": "quantile",
                "edge_retention_rate": 0.3,
                "enable_lcc_extraction": False,
                "enable_community_pruning": False,
                "min_community_size": 3
            }
        }
        
        with open(default_config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        print("✅ 创建默认配置文件")
    else:
        print("📄 配置文件已存在")
    
    return True

def verify_installation():
    """验证安装"""
    print("\\n🔍 验证安装...")
    
    try:
        # 测试核心模块导入
        print("📦 测试模块导入...")
        
        import numpy
        print("✅ numpy")
        
        import pandas
        print("✅ pandas")
        
        import nltk
        print("✅ nltk")
        
        import matplotlib
        print("✅ matplotlib")
        
        import networkx
        print("✅ networkx")
        
        import tqdm
        print("✅ tqdm")
        
        # 测试主程序导入
        try:
            import complete_usage_guide
            print("✅ complete_usage_guide")
        except ImportError as e:
            print(f"⚠️ complete_usage_guide导入警告: {e}")
        
        print("✅ 模块导入测试通过")
        return True
        
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False

def show_completion_message():
    """显示完成信息"""
    print("\\n" + "=" * 60)
    print("🎉 安装完成！")
    print("=" * 60)
    
    print("\\n✅ 安装成功完成！现在您可以:")
    
    print("\\n1. 🚀 运行快速开始演示:")
    print("   python quick_start.py")
    
    print("\\n2. 📊 运行主程序:")
    print("   python complete_usage_guide.py")
    
    print("\\n3. 🧪 运行测试:")
    print("   pytest")
    
    print("\\n4. 📚 查看文档:")
    print("   - README.md - 项目概述和使用指南")
    print("   - docs/ - 详细文档")
    
    print("\\n5. 🔧 自定义配置:")
    print("   - 编辑 config/default_config.json")
    
    print("\\n💡 提示:")
    print("   - 如果遇到问题，请查看GitHub Issues")
    print("   - 建议先运行 python quick_start.py 体验功能")

def main():
    """主安装函数"""
    print("🔧 Graph4SocialScience 自动安装")
    print("=" * 60)
    print("正在安装语义增强共词网络分析管线...")
    print("=" * 60)
    
    # 检查Python版本
    if not check_python_version():
        return 1
    
    # 安装Python依赖
    if not install_requirements():
        print("\\n❌ 依赖安装失败，请检查网络连接和权限")
        return 1
    
    # 安装EasyGraph
    install_easygraph()  # 不强制要求成功
    
    # 下载NLTK数据
    download_nltk_data()  # 不强制要求成功
    
    # 创建目录结构
    if not create_directories():
        print("\\n❌ 目录创建失败")
        return 1
    
    # 创建配置文件
    if not create_config_files():
        print("\\n❌ 配置文件创建失败")
        return 1
    
    # 验证安装
    if not verify_installation():
        print("\\n⚠️ 安装验证有警告，但可以继续使用")
    
    # 显示完成信息
    show_completion_message()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())