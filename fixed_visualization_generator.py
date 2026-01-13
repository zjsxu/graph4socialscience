#!/usr/bin/env python3
"""
修复巨大圆环问题的可视化生成器
实现：LCC提取、边权重Quantile过滤(0.98)、自适应k参数布局
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from collections import defaultdict, Counter
from tqdm import tqdm
from datetime import datetime
import math


def extract_largest_connected_component(graph):
    """
    强制提取图的最大连通分量(LCC)进行绘图
    解决图过于碎裂导致的巨大圆环问题
    """
    print("🔗 强制提取最大连通分量(LCC)...")
    
    if graph.number_of_nodes() == 0:
        return graph
    
    # 找到所有连通分量
    components = list(nx.connected_components(graph))
    
    if not components:
        return nx.Graph()
    
    # 获取最大连通分量
    largest_component = max(components, key=len)
    lcc = graph.subgraph(largest_component).copy()
    
    print(f"   📊 原图: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
    print(f"   📊 LCC: {lcc.number_of_nodes()} 节点, {lcc.number_of_edges()} 边")
    print(f"   📊 移除孤立节点: {graph.number_of_nodes() - lcc.number_of_nodes()}")
    print(f"   📊 连通分量数: {len(components)} → 1")
    
    return lcc


def apply_quantile_edge_filtering(graph, quantile_threshold=0.98):
    """
    应用更严格的边权重Quantile过滤(0.98)
    确保只有最强的关联被保留
    """
    print(f"📊 应用严格边权重过滤 (Quantile {quantile_threshold})...")
    
    if graph.number_of_edges() == 0:
        return graph
    
    # 获取所有边权重
    edge_weights = []
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', data.get('semantic_weight', 1.0))
        edge_weights.append(((u, v), weight))
    
    # 计算阈值
    weights_only = [w for _, w in edge_weights]
    threshold = np.percentile(weights_only, quantile_threshold * 100)
    
    # 创建过滤后的图
    filtered_graph = nx.Graph()
    filtered_graph.add_nodes_from(graph.nodes(data=True))
    
    edges_kept = 0
    for (u, v), weight in edge_weights:
        if weight >= threshold:
            filtered_graph.add_edge(u, v, **graph[u][v])
            edges_kept += 1
    
    print(f"   📊 权重阈值: {threshold:.4f}")
    print(f"   📊 保留边数: {edges_kept} / {len(edge_weights)} ({edges_kept/len(edge_weights)*100:.1f}%)")
    
    return filtered_graph


def compute_adaptive_spring_layout(graph, seed=42):
    """
    计算自适应spring布局，k参数根据节点数自动调整
    公式: k = 1 / sqrt(n)
    """
    print("🎯 计算自适应spring布局...")
    
    if graph.number_of_nodes() == 0:
        return {}
    
    n_nodes = graph.number_of_nodes()
    k_param = 1.0 / math.sqrt(n_nodes)
    
    print(f"   📊 节点数: {n_nodes}")
    print(f"   📊 自适应k参数: {k_param:.4f} (= 1/√{n_nodes})")
    
    # 计算布局
    with tqdm(total=1000, desc="🎯 Spring布局计算", unit="iter") as pbar:
        pos = nx.spring_layout(
            graph,
            k=k_param,
            iterations=1000,
            seed=seed,
            weight='weight'
        )
        pbar.update(1000)
    
    print(f"   ✅ 布局计算完成: {len(pos)} 个节点位置")
    return pos


def generate_fixed_visualization(graph, output_path, title="Fixed Co-occurrence Network", 
                                seed=42, quantile_threshold=0.98):
    """
    生成修复后的可视化，解决巨大圆环问题
    
    修复措施：
    1. 强制LCC提取
    2. 严格边权重过滤(0.98)
    3. 自适应k参数布局
    """
    print(f"\n🎨 生成修复后的可视化: {title}")
    print("=" * 60)
    
    if graph.number_of_nodes() == 0:
        print("   ⚠️ 空图，跳过可视化")
        return None
    
    # 步骤1: 应用严格边权重过滤
    filtered_graph = apply_quantile_edge_filtering(graph, quantile_threshold)
    
    # 步骤2: 强制提取LCC
    lcc_graph = extract_largest_connected_component(filtered_graph)
    
    if lcc_graph.number_of_nodes() == 0:
        print("   ⚠️ LCC为空，跳过可视化")
        return None
    
    # 步骤3: 计算自适应布局
    positions = compute_adaptive_spring_layout(lcc_graph, seed)
    
    # 步骤4: 社区检测（用于着色）
    print("🏘️ 检测社区结构...")
    try:
        import community as community_louvain
        communities = community_louvain.best_partition(lcc_graph, weight='weight', random_state=seed)
    except:
        communities = {node: 0 for node in lcc_graph.nodes()}
    
    community_count = len(set(communities.values()))
    print(f"   📊 检测到社区数: {community_count}")
    
    # 步骤5: 核心-外围识别
    print("🎯 识别核心-外围结构...")
    degrees = dict(lcc_graph.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    n_core = max(1, min(20, len(sorted_nodes) // 4))  # 最多20个核心节点
    core_nodes = set(node for node, _ in sorted_nodes[:n_core])
    
    print(f"   📊 核心节点数: {n_core} / {len(sorted_nodes)}")
    
    # 步骤6: 生成可视化
    print("🎨 绘制网络图...")
    
    plt.style.use('default')
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_facecolor('white')
    
    # 准备节点属性
    node_colors = []
    node_sizes = []
    node_shapes_core = []
    node_shapes_periphery = []
    
    # 社区颜色
    unique_communities = sorted(set(communities.values()))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_communities)))
    community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}
    
    # 处理每个节点
    for node in lcc_graph.nodes():
        # 社区颜色
        comm_id = communities.get(node, 0)
        color = community_colors.get(comm_id, 'lightblue')
        node_colors.append(color)
        
        # 节点大小（基于度数）
        degree = degrees.get(node, 1)
        max_degree = max(degrees.values()) if degrees else 1
        size = 100 + 400 * (degree / max_degree)
        node_sizes.append(size)
        
        # 按角色分组
        if node in core_nodes:
            node_shapes_core.append(node)
        else:
            node_shapes_periphery.append(node)
    
    # 绘制边（轻灰色，低透明度）
    if lcc_graph.number_of_edges() > 0:
        nx.draw_networkx_edges(
            lcc_graph, positions,
            width=0.5,
            alpha=0.2,
            edge_color='lightgray',
            ax=ax
        )
    
    # 绘制外围节点（圆形）
    if node_shapes_periphery:
        periphery_colors = [community_colors.get(communities.get(node, 0), 'lightblue') 
                          for node in node_shapes_periphery]
        periphery_sizes = [node_sizes[list(lcc_graph.nodes()).index(node)] 
                         for node in node_shapes_periphery]
        nx.draw_networkx_nodes(
            lcc_graph, positions,
            nodelist=node_shapes_periphery,
            node_color=periphery_colors,
            node_size=periphery_sizes,
            node_shape='o',
            alpha=0.8,
            edgecolors='gray',
            linewidths=0.5,
            ax=ax
        )
    
    # 绘制核心节点（三角形）
    if node_shapes_core:
        core_colors = [community_colors.get(communities.get(node, 0), 'lightblue') 
                      for node in node_shapes_core]
        core_sizes = [node_sizes[list(lcc_graph.nodes()).index(node)] 
                     for node in node_shapes_core]
        nx.draw_networkx_nodes(
            lcc_graph, positions,
            nodelist=node_shapes_core,
            node_color=core_colors,
            node_size=core_sizes,
            node_shape='^',
            alpha=0.9,
            edgecolors='black',
            linewidths=1.0,
            ax=ax
        )
    
    # 添加选择性标签（仅核心节点）
    if node_shapes_core:
        labels_to_draw = {}
        for node in node_shapes_core[:10]:  # 最多10个标签
            label = str(node)[:15] + "..." if len(str(node)) > 15 else str(node)
            labels_to_draw[node] = label
        
        if labels_to_draw:
            nx.draw_networkx_labels(
                lcc_graph, positions,
                labels_to_draw,
                font_size=8,
                font_weight='bold',
                font_color='black',
                ax=ax
            )
    
    # 标题和统计信息
    density = nx.density(lcc_graph)
    ax.set_title(
        f'{title}\n'
        f'LCC: N={lcc_graph.number_of_nodes()}, E={lcc_graph.number_of_edges()}, '
        f'Density={density:.4f}, Communities={community_count}\n'
        f'Core={len(node_shapes_core)}, Quantile={quantile_threshold}, k={1.0/math.sqrt(lcc_graph.number_of_nodes()):.4f}, Seed={seed}',
        fontsize=14, fontweight='bold', pad=20
    )
    
    # 图例
    legend_elements = []
    
    # 社区图例（最多显示8个）
    for i, (comm_id, color) in enumerate(list(community_colors.items())[:8]):
        comm_size = sum(1 for c in communities.values() if c == comm_id)
        legend_elements.append(patches.Patch(color=color, label=f'Community {comm_id} (n={comm_size})'))
    
    if len(unique_communities) > 8:
        other_size = sum(1 for c in communities.values() if c not in list(community_colors.keys())[:8])
        legend_elements.append(patches.Patch(color='lightgray', label=f'Other (n={other_size})'))
    
    # 形状图例
    legend_elements.append(patches.Patch(color='white', label=''))  # 分隔符
    legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                    markerfacecolor='gray', markersize=10, label='Core nodes'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='gray', markersize=8, label='Periphery nodes'))
    
    # 方法图例
    legend_elements.append(patches.Patch(color='white', label=''))  # 分隔符
    legend_elements.append(patches.Patch(color='lightgray', label=f'LCC Extraction: Yes'))
    legend_elements.append(patches.Patch(color='lightgray', label=f'Edge Filter: Q{quantile_threshold}'))
    legend_elements.append(patches.Patch(color='lightgray', label=f'Adaptive k: 1/√n'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
            frameon=True, fancybox=True, shadow=True)
    
    ax.axis('off')
    plt.tight_layout()
    
    # 保存高分辨率图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   ✅ 保存: {output_path}")
    print(f"   📊 最终图: {lcc_graph.number_of_nodes()} 节点, {lcc_graph.number_of_edges()} 边")
    print(f"   📊 密度: {density:.6f}")
    print(f"   📊 社区数: {community_count}")
    
    return output_path


def test_fixed_visualization():
    """测试修复后的可视化生成"""
    print("🔧 测试修复后的可视化生成")
    print("修复措施: LCC提取 + 边权重过滤(0.98) + 自适应k参数")
    print()
    
    # 导入主程序
    try:
        from complete_usage_guide import ResearchPipelineCLI
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return
    
    # 设置测试参数 - use relative paths for portability
    input_dir = "test_input"
    output_dir = "test_output"
    
    if not os.path.exists(input_dir):
        print(f"❌ 输入目录不存在: {input_dir}")
        return
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建管线实例并运行到图构建
    cli = ResearchPipelineCLI()
    cli.input_directory = input_dir
    cli.output_dir = output_dir
    
    # 扫描输入文件
    cli.input_files = []
    valid_extensions = {'.json', '.txt', '.md'}
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in valid_extensions:
                cli.input_files.append(file_path)
    
    print(f"📊 找到文件: {len(cli.input_files)} 个")
    
    # 设置管线状态
    cli.pipeline_state = {
        'data_loaded': True,
        'text_cleaned': False,
        'phrases_constructed': False,
        'global_graph_built': False,
        'subgraphs_activated': False,
        'results_exported': False
    }
    
    try:
        # 执行管线到图构建
        print("\n=== 执行管线到图构建 ===")
        cli.clean_and_normalize_text()
        cli.extract_tokens_and_phrases()
        cli.build_global_graph()
        
        # 获取构建的图
        if hasattr(cli, 'global_graph_object') and cli.global_graph_object:
            graph = cli.global_graph_object
            print(f"\n📊 原始图统计:")
            print(f"   节点数: {graph.number_of_nodes()}")
            print(f"   边数: {graph.number_of_edges()}")
            print(f"   密度: {nx.density(graph):.6f}")
            print(f"   连通分量数: {nx.number_connected_components(graph)}")
            
            # 生成修复后的可视化
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"fixed_visualization_{timestamp}.png")
            
            generate_fixed_visualization(
                graph, 
                output_path, 
                title="Fixed Global Co-occurrence Network",
                seed=42,
                quantile_threshold=0.98
            )
            
            print(f"\n✅ 修复后的可视化已生成:")
            print(f"📁 路径: {output_path}")
            
        else:
            print("❌ 未能获取图对象")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_fixed_visualization()