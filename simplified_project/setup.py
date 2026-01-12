"""
语义增强共词网络分析管线安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "语义增强共词网络分析管线"

# 读取requirements文件
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="semantic-coword-pipeline",
    version="0.1.0",
    description="语义增强共词网络分析管线",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Semantic Coword Enhancement Team",
    author_email="team@example.com",
    url="https://github.com/example/semantic-coword-pipeline",
    
    # 包配置
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    
    # Python版本要求
    python_requires=">=3.8",
    
    # 依赖
    install_requires=read_requirements(),
    
    # 额外依赖
    extras_require={
        'dev': [
            'black>=21.0.0',
            'flake8>=3.9.0',
            'mypy>=0.910',
            'pytest>=6.0.0',
            'pytest-cov>=2.12.0',
            'hypothesis>=6.0.0'
        ],
        'viz': [
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'plotly>=5.0.0'
        ],
        'performance': [
            'memory-profiler>=0.58.0',
            'line-profiler>=3.3.0',
            'cProfile'
        ]
    },
    
    # 分类
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    
    # 关键词
    keywords="nlp, network-analysis, cooccurrence, semantic-enhancement, graph-analysis",
    
    # 项目URL
    project_urls={
        "Bug Reports": "https://github.com/example/semantic-coword-pipeline/issues",
        "Source": "https://github.com/example/semantic-coword-pipeline",
        "Documentation": "https://semantic-coword-pipeline.readthedocs.io/",
    },
    
    # 命令行入口点
    entry_points={
        'console_scripts': [
            'semantic-coword=semantic_coword_pipeline.cli:main',
            'semantic-coword-pipeline=semantic_coword_pipeline.pipeline:main',
        ],
    },
    
    # 包数据
    package_data={
        'semantic_coword_pipeline': [
            'data/*.txt',
            'config/*.json',
            'templates/*.md'
        ],
    },
    
    # 数据文件
    data_files=[
        ('config', ['config/default_config.json']),
    ],
    
    # zip安全
    zip_safe=False,
)