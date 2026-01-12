#!/usr/bin/env python3
"""
快速启动脚本

使用完整的语义共词网络分析管线进行分析。
如果你想要更简单的使用方式，请使用 semantic_coword_analyzer.py
"""

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
