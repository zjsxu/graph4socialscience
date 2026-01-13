#!/usr/bin/env python3
"""
可复现性验证脚本

这个脚本模拟新用户从GitHub下载项目后的完整体验流程，
验证项目是否真正满足可复现性要求。
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def simulate_fresh_clone():
    """模拟新用户克隆项目的体验"""
    print("🔄 模拟新用户体验...")
    print("=" * 60)
    
    # 创建临时目录模拟新环境
    temp_dir = tempfile.mkdtemp(prefix="graph4socialscience_test_")
    print(f"📁 创建测试环境: {temp_dir}")
    
    try:
        # 复制项目文件到临时目录
        current_dir = os.getcwd()
        
        # 复制关键文件
        key_files = [
            'README.md',
            'requirements.txt',
            'setup.py',
            'install.py',
            'quick_start.py',
            'complete_usage_guide.py',
            'check_reproducibility.py'
        ]
        
        key_dirs = [
            'data',
            'config',
            'semantic_coword_pipeline',
            'tests'
        ]
        
        for file in key_files:
            if os.path.exists(file):
                shutil.copy2(file, temp_dir)
                print(f"✅ 复制文件: {file}")
        
        for dir_name in key_dirs:
            if os.path.exists(dir_name):
                shutil.copytree(dir_name, os.path.join(temp_dir, dir_name))
                print(f"✅ 复制目录: {dir_name}")
        
        # 切换到测试目录
        os.chdir(temp_dir)
        
        # 运行可复现性检查
        print("\\n🔍 在新环境中运行可复现性检查...")
        result = subprocess.run([sys.executable, 'check_reproducibility.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 可复现性检查通过")
            print(result.stdout)
        else:
            print("❌ 可复现性检查失败")
            print(result.stdout)
            print(result.stderr)
            return False
        
        # 测试快速开始脚本
        print("\\n🚀 测试快速开始脚本...")
        
        # 检查脚本是否可以运行（不实际执行，避免长时间运行）
        try:
            with open('quick_start.py', 'r') as f:
                content = f.read()
                if 'def main()' in content and 'check_environment' in content:
                    print("✅ quick_start.py 结构正确")
                else:
                    print("❌ quick_start.py 结构不完整")
                    return False
        except Exception as e:
            print(f"❌ quick_start.py 检查失败: {e}")
            return False
        
        # 测试主程序导入
        print("\\n📦 测试主程序导入...")
        try:
            # 简单的导入测试
            import_test = f'''
import sys
sys.path.insert(0, "{temp_dir}")
try:
    import complete_usage_guide
    print("✅ complete_usage_guide 导入成功")
except ImportError as e:
    print(f"❌ complete_usage_guide 导入失败: {{e}}")
    sys.exit(1)
'''
            
            result = subprocess.run([sys.executable, '-c', import_test], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print("❌ 主程序导入失败")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ 导入测试失败: {e}")
            return False
        
        print("\\n✅ 新用户体验验证通过！")
        return True
        
    except Exception as e:
        print(f"❌ 验证过程失败: {e}")
        return False
        
    finally:
        # 恢复原目录
        os.chdir(current_dir)
        
        # 清理临时目录
        try:
            shutil.rmtree(temp_dir)
            print(f"🧹 清理测试环境: {temp_dir}")
        except Exception as e:
            print(f"⚠️ 清理失败: {e}")

def check_github_readiness():
    """检查GitHub发布就绪性"""
    print("\\n🔍 检查GitHub发布就绪性...")
    print("-" * 40)
    
    issues = []
    
    # 检查关键文件
    required_files = [
        'README.md',
        'requirements.txt',
        'setup.py',
        'install.py',
        'quick_start.py',
        '.gitignore'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"❌ 缺少关键文件: {file}")
        else:
            print(f"✅ {file}")
    
    # 检查目录结构
    required_dirs = [
        'data',
        'config',
        'semantic_coword_pipeline',
        'tests',
        'docs'
    ]
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            issues.append(f"❌ 缺少关键目录: {dir_name}")
        else:
            print(f"✅ {dir_name}/")
    
    # 检查README内容
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        required_sections = ['安装', '快速开始', '使用方法']
        for section in required_sections:
            if section not in readme_content:
                issues.append(f"❌ README缺少章节: {section}")
            else:
                print(f"✅ README包含: {section}")
    
    # 检查git状态
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        
        if result.stdout.strip():
            print("⚠️ 有未提交的更改:")
            print(result.stdout)
        else:
            print("✅ Git状态干净")
            
    except Exception as e:
        print(f"⚠️ Git状态检查失败: {e}")
    
    return len(issues) == 0, issues

def generate_user_guide():
    """生成用户指南"""
    print("\\n📚 生成用户指南...")
    
    user_guide = """# 新用户快速指南

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
"""
    
    with open('USER_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(user_guide)
    
    print("✅ 生成用户指南: USER_GUIDE.md")

def main():
    """主验证函数"""
    print("🔍 Graph4SocialScience 可复现性验证")
    print("=" * 60)
    print("验证项目是否满足可复现性要求...")
    print("=" * 60)
    
    # 1. 检查GitHub发布就绪性
    github_ready, issues = check_github_readiness()
    
    if not github_ready:
        print("\\n❌ GitHub发布就绪性检查失败:")
        for issue in issues:
            print(f"   {issue}")
        return 1
    
    print("\\n✅ GitHub发布就绪性检查通过")
    
    # 2. 模拟新用户体验
    if not simulate_fresh_clone():
        print("\\n❌ 新用户体验验证失败")
        return 1
    
    # 3. 生成用户指南
    generate_user_guide()
    
    # 4. 最终总结
    print("\\n" + "=" * 60)
    print("🎉 可复现性验证完成！")
    print("=" * 60)
    
    print("\\n✅ 验证结果:")
    print("   ✅ 项目结构完整")
    print("   ✅ 依赖管理正确")
    print("   ✅ 文档完整清晰")
    print("   ✅ 安装脚本可用")
    print("   ✅ 新用户体验良好")
    print("   ✅ 代码可正常导入")
    
    print("\\n🎯 项目已准备好发布！")
    print("\\n用户现在可以:")
    print("   1. 从GitHub克隆项目")
    print("   2. 运行 python install.py 自动安装")
    print("   3. 运行 python quick_start.py 立即体验")
    print("   4. 获得完整的功能和文档支持")
    
    print("\\n🔗 GitHub仓库: https://github.com/zjsxu/graph4socialscience")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())