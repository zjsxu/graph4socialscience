#!/usr/bin/env python3
"""
测试Plotly可视化集成到主管线
"""

import os
import sys
from datetime import datetime

def test_plotly_integration():
    """测试Plotly可视化集成"""
    print("🧪 测试Plotly可视化集成到主管线")
    print("=" * 50)
    
    try:
        # 导入主程序
        from complete_usage_guide import ResearchPipelineCLI
        
        # 初始化管线
        print("🔄 初始化管线...")
        app = ResearchPipelineCLI()
        
        # 设置输出目录
        output_dir = "test_output"
        app.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 输出目录: {output_dir}")
        
        # 创建示例数据
        print("📊 创建示例数据...")
        app.create_sample_research_data()
        
        # 运行管线步骤
        print("\n🔄 运行管线步骤...")
        
        # 文本清理
        print("1️⃣ 文本清理...")
        app.clean_and_normalize_text()
        
        # 词组提取
        print("2️⃣ 词组提取...")
        app.extract_tokens_and_phrases()
        
        # 全局图构建
        print("3️⃣ 全局图构建...")
        app.build_global_graph()
        
        # 子图激活
        print("4️⃣ 子图激活...")
        app.activate_state_subgraphs()
        
        # 测试Plotly可视化
        print("5️⃣ 测试Plotly可视化...")
        app.generate_plotly_visualizations()
        
        # 检查生成的文件
        viz_dir = os.path.join(output_dir, "plotly_visualizations")
        if os.path.exists(viz_dir):
            files = [f for f in os.listdir(viz_dir) if f.endswith('.html')]
            print(f"\n✅ 成功生成 {len(files)} 个HTML文件:")
            for file in files[:5]:  # 显示前5个文件
                print(f"   📄 {file}")
            if len(files) > 5:
                print(f"   ... 还有 {len(files) - 5} 个文件")
        
        print("\n🎉 Plotly可视化集成测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_plotly_integration()
    sys.exit(0 if success else 1)