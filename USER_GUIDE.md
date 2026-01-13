# 新用户快速指南

欢迎使用Graph4SocialScience语义增强共词网络分析管线！

## 🚀 三种开始方式

### 方式1: 一键自动安装 (推荐)
```bash
git clone https://github.com/zjsxu/graph4socialscience.git
cd graph4socialscience
python install.py
python quick_start.py
```

### 方式2: 手动安装
```bash
git clone https://github.com/zjsxu/graph4socialscience.git
cd graph4socialscience
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
python complete_usage_guide.py
```

### 方式3: 开发者安装
```bash
git clone https://github.com/zjsxu/graph4socialscience.git
cd graph4socialscience
pip install -e .
pytest  # 运行测试
python demo.py  # 运行演示
```

## 🎯 预期结果

成功运行后，您将看到：
- 📊 网络可视化图像文件
- 📁 完整的分析结果目录
- 📋 详细的处理报告
- 🔍 可追溯的参数配置

## 💡 故障排除

如果遇到问题：
1. 检查Python版本 (需要3.8+)
2. 确保网络连接正常
3. 查看GitHub Issues
4. 运行 `python check_reproducibility.py` 诊断

## 📞 获取帮助

- GitHub Issues: https://github.com/zjsxu/graph4socialscience/issues
- 文档: docs/ 目录
- 示例: data/ 目录
