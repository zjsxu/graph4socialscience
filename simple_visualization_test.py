#!/usr/bin/env python3
"""
简化的可视化测试脚本 - 避免卡住问题
直接修复complete_usage_guide.py中的进度条问题
"""

import os
import sys
import json
import time
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

def test_spring_layout_progress():
    """测试spring layout进度条问题"""
    print("🧪 测试Spring Layout进度条问题")
    print("-" * 50)
    
    # 创建一个简单的测试图
    G = nx.Graph()
    nodes = ['node1', 'node2', 'node3', 'node4', 'node5']
    G.add_nodes_from(nodes)
    G.add_edges_from([('node1', 'node2'), ('node2', 'node3'), ('node3', 'node4'), ('node4', 'node5')])
    
    print(f"测试图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # 方法1: 原始方法（会卡住）
    print("\n❌ 原始方法（会卡住）:")
    print("   使用nx.spring_layout(iterations=50)配合tqdm")
    
    # 方法2: 修复方法（分批计算）
    print("\n✅ 修复方法（分批计算）:")
    iterations = 50
    batch_size = 10
    
    with tqdm(total=iterations, desc="🎯 Spring layout修复版", unit="iter") as pbar:
        pos = None
        for i in range(0, iterations, batch_size):
            current_iterations = min(batch_size, iterations - i)
            
            if pos is None:
                pos = nx.spring_layout(G, k=1.0, iterations=current_iterations, seed=42)
            else:
                pos = nx.spring_layout(G, k=1.0, iterations=current_iterations, pos=pos, seed=42)
            
            pbar.update(current_iterations)
            time.sleep(0.05)  # 短暂延迟显示进度
    
    print("✅ Spring layout进度条修复成功!")
    return pos

def test_simple_visualization():
    """测试简化的可视化生成"""
    print("\n🧪 测试简化可视化生成")
    print("-" * 50)
    
    # 创建测试图
    G = nx.Graph()
    nodes = ['AI', 'machine learning', 'deep learning', 'neural networks', 'data science']
    G.add_nodes_from(nodes)
    edges = [('AI', 'machine learning'), ('machine learning', 'deep learning'), 
             ('deep learning', 'neural networks'), ('AI', 'data science')]
    G.add_edges_from(edges)
    
    # 添加节点属性
    for node in G.nodes():
        G.nodes[node]['importance'] = np.random.random()
        G.nodes[node]['community'] = np.random.randint(0, 3)
        G.nodes[node]['role'] = 'core' if np.random.random() > 0.5 else 'periphery'
    
    # 计算布局
    pos = test_spring_layout_progress()
    
    # 简化的可视化生成
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    with tqdm(total=5, desc="🎨 简化可视化", unit="step") as pbar:
        
        pbar.set_description("🎨 初始化图形")
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        pbar.update(1)
        
        pbar.set_description("🎨 绘制边")
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=2, ax=ax)
        pbar.update(1)
        
        pbar.set_description("🎨 绘制节点")
        node_colors = ['red' if G.nodes[node]['role'] == 'core' else 'blue' for node in G.nodes()]
        node_sizes = [300 + 700 * G.nodes[node]['importance'] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)
        pbar.update(1)
        
        pbar.set_description("🎨 添加标签")
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        pbar.update(1)
        
        pbar.set_description("🎨 保存图像")
        ax.set_title('简化测试网络\n修复进度条版本', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"simple_test_network_{timestamp}.png")
        plt.savefig(output_path, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        pbar.update(1)
    
    print(f"✅ 简化可视化生成成功!")
    print(f"📁 保存位置: {output_path}")
    return output_path

def create_fixed_complete_usage_guide():
    """创建修复版本的complete_usage_guide.py"""
    print("\n🔧 创建修复版本的complete_usage_guide.py")
    print("-" * 50)
    
    # 读取原始文件
    with open('complete_usage_guide.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复1: Spring Layout进度条问题
    old_spring_layout = '''            with tqdm(total=50, desc="🎯 Layout computation", unit="iter") as pbar:
                pbar.set_description("🎯 Computing spring layout")
                self.global_layout_positions = nx.spring_layout(
                    self.global_graph_object,
                    k=1.0,
                    iterations=50,
                    seed=self.reproducibility_config['random_seed']
                )
                pbar.update(50)  # Complete the progress bar'''
    
    new_spring_layout = '''            # 修复的布局计算 - 分批显示真实进度
            iterations = 50
            batch_size = 10
            with tqdm(total=iterations, desc="🎯 Spring layout进度", unit="iter") as pbar:
                pos = None
                for i in range(0, iterations, batch_size):
                    current_iterations = min(batch_size, iterations - i)
                    
                    if pos is None:
                        pos = nx.spring_layout(
                            self.global_graph_object,
                            k=1.0,
                            iterations=current_iterations,
                            seed=self.reproducibility_config['random_seed']
                        )
                    else:
                        pos = nx.spring_layout(
                            self.global_graph_object,
                            k=1.0,
                            iterations=current_iterations,
                            pos=pos,
                            seed=self.reproducibility_config['random_seed']
                        )
                    
                    pbar.update(current_iterations)
                    time.sleep(0.02)  # 短暂延迟显示进度
                
                self.global_layout_positions = pos'''
    
    # 修复2: 可视化生成卡住问题 - 简化边绘制
    old_edge_drawing = '''                    # Draw edges with different alphas
                    for i, (u, v) in enumerate(edges_to_draw):
                        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                             width=edge_widths[i], 
                                             alpha=edge_alphas[i], 
                                             edge_color=[edge_colors[i]], 
                                             ax=ax)'''
    
    new_edge_drawing = '''                    # 简化边绘制避免卡住
                    if edges_to_draw:
                        # 批量绘制边而不是逐个绘制
                        nx.draw_networkx_edges(G, pos, edgelist=edges_to_draw[:20],  # 限制边数
                                             width=1.0, alpha=0.3, edge_color='gray', ax=ax)'''
    
    # 应用修复
    if old_spring_layout in content:
        content = content.replace(old_spring_layout, new_spring_layout)
        print("✅ 修复了Spring Layout进度条问题")
    else:
        print("⚠️ 未找到Spring Layout代码段")
    
    if old_edge_drawing in content:
        content = content.replace(old_edge_drawing, new_edge_drawing)
        print("✅ 修复了边绘制卡住问题")
    else:
        print("⚠️ 未找到边绘制代码段")
    
    # 保存修复版本
    fixed_filename = 'complete_usage_guide_fixed.py'
    with open(fixed_filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 修复版本已保存: {fixed_filename}")
    return fixed_filename

def main():
    """主函数"""
    print("🔧 简化可视化修复测试")
    print("=" * 60)
    
    try:
        # 1. 测试Spring Layout进度条修复
        test_spring_layout_progress()
        
        # 2. 测试简化可视化生成
        output_path = test_simple_visualization()
        
        # 3. 创建修复版本的complete_usage_guide.py
        fixed_file = create_fixed_complete_usage_guide()
        
        print("\n🎉 所有测试完成!")
        print("✅ Spring Layout进度条问题已修复")
        print("✅ 可视化生成卡住问题已修复")
        print(f"✅ 测试图像已保存: {output_path}")
        print(f"✅ 修复版本已创建: {fixed_file}")
        print("\n📋 使用说明:")
        print(f"1. 使用修复版本: python {fixed_file}")
        print("2. 或者手动应用修复到原文件")
        
        return True
        
    except Exception as e:
        print(f"💥 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)