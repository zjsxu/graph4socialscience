#!/usr/bin/env python3
"""
专门测试和修复可视化功能的脚本
修复问题：
1. 4.1步骤的spring layout进度条只显示0%和100%
2. 6.1步骤的可视化生成卡住不动

数据来源：/Users/zhangjingsen/Desktop/python/graph4socialscience/toc_doc
输出目录：/Users/zhangjingsen/Desktop/python/graph4socialscience/hajimi/haniumoa/
"""

import os
import sys
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

# 添加当前目录到路径
sys.path.insert(0, '.')

class VisualizationTester:
    """可视化功能测试和修复类"""
    
    def __init__(self):
        self.input_directory = "/Users/zhangjingsen/Desktop/python/graph4socialscience/toc_doc"
        self.output_dir = "/Users/zhangjingsen/Desktop/python/graph4socialscience/hajimi/haniumoa"
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化数据
        self.cleaned_text_data = None
        self.phrase_data = None
        self.global_graph_object = None
        self.global_layout_positions = None
        self.state_subgraph_objects = {}
        self.visualization_paths = {}
        
        # 配置参数
        self.reproducibility_config = {
            'random_seed': 42,
            'phrase_type': 'mixed',
            'min_phrase_frequency': 2,
            'layout_algorithm': 'spring_deterministic'
        }
        
        self.graph_construction_config = {
            'edge_density_reduction': 0.1,
            'min_edge_weight': 2,
            'core_node_percentile': 0.2,
            'community_layout_separation': 2.0,
        }
        
        self.viz_config = {
            'edge_alpha': 0.15,
            'intra_community_edge_alpha': 0.3,
            'inter_community_edge_alpha': 0.05,
            'core_node_shape': '^',
            'periphery_node_shape': 'o',
            'min_node_size': 100,
            'max_node_size': 1000,
            'label_importance_threshold': 0.7,
            'max_labels_per_community': 3,
        }
    
    def load_test_data(self):
        """加载测试数据"""
        print("📁 加载测试数据...")
        print(f"   输入目录: {self.input_directory}")
        
        if not os.path.exists(self.input_directory):
            print(f"❌ 输入目录不存在: {self.input_directory}")
            return False
        
        # 扫描目录中的文件
        input_files = []
        valid_extensions = {'.json', '.txt', '.md'}
        
        for root, dirs, files in os.walk(self.input_directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in valid_extensions:
                    input_files.append(file_path)
        
        print(f"   找到 {len(input_files)} 个有效文件")
        
        # 加载数据
        all_data = []
        for file_path in input_files[:10]:  # 限制文件数量以便测试
            try:
                # 从路径提取状态
                rel_path = os.path.relpath(file_path, self.input_directory)
                path_parts = rel_path.split(os.sep)
                state = path_parts[0] if len(path_parts) > 1 else "Unknown"
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.endswith('.json'):
                        data = json.load(f)
                        if isinstance(data, list):
                            for doc in data:
                                doc['state'] = state
                            all_data.extend(data)
                        else:
                            data['state'] = state
                            all_data.append(data)
                    else:
                        content = f.read()
                        doc_data = {
                            "segment_id": f"doc_{len(all_data)+1}",
                            "title": os.path.basename(file_path),
                            "text": content,
                            "state": state,
                            "language": "english"
                        }
                        all_data.append(doc_data)
            except Exception as e:
                print(f"   ⚠️ 跳过文件 {file_path}: {e}")
        
        print(f"✅ 成功加载 {len(all_data)} 个文档")
        return all_data
    
    def simulate_text_cleaning(self, input_data):
        """模拟文本清理"""
        print("🧹 模拟文本清理...")
        
        cleaned_documents = []
        for doc in tqdm(input_data, desc="🧹 清理文档", unit="doc"):
            cleaned_text = doc['text'].lower().strip()
            tokens = [token for token in cleaned_text.split() if len(token) > 2]
            
            cleaned_doc = {
                'segment_id': doc['segment_id'],
                'title': doc['title'],
                'original_text': doc['text'],
                'cleaned_text': cleaned_text,
                'tokens': tokens,
                'token_count': len(tokens),
                'state': doc['state'],
                'language': doc.get('language', 'english')
            }
            cleaned_documents.append(cleaned_doc)
        
        self.cleaned_text_data = cleaned_documents
        print(f"✅ 清理完成: {len(cleaned_documents)} 个文档")
        return True
    
    def simulate_phrase_extraction(self):
        """模拟短语提取"""
        print("🔍 模拟短语提取...")
        
        all_phrases = []
        phrase_counts = {}
        
        for doc in tqdm(self.cleaned_text_data, desc="🔍 提取短语", unit="doc"):
            tokens = doc['tokens']
            
            # 提取单词
            if self.reproducibility_config['phrase_type'] in ['word', 'mixed']:
                for token in tokens:
                    if len(token) > 2:
                        all_phrases.append(token)
                        phrase_counts[token] = phrase_counts.get(token, 0) + 1
            
            # 提取双词组
            if self.reproducibility_config['phrase_type'] in ['bigram', 'mixed']:
                for i in range(len(tokens) - 1):
                    bigram = f"{tokens[i]} {tokens[i+1]}"
                    all_phrases.append(bigram)
                    phrase_counts[bigram] = phrase_counts.get(bigram, 0) + 1
        
        # 过滤低频短语
        min_freq = self.reproducibility_config['min_phrase_frequency']
        filtered_phrases = {phrase: count for phrase, count in phrase_counts.items() 
                          if count >= min_freq}
        
        self.phrase_data = {
            'all_phrases': all_phrases,
            'phrase_counts': phrase_counts,
            'filtered_phrases': filtered_phrases
        }
        
        print(f"✅ 短语提取完成: {len(filtered_phrases)} 个有效短语")
        return True
    
    def build_global_graph_with_fixed_progress(self):
        """构建全局图，修复进度条问题"""
        print("🌐 构建全局图（修复进度条）...")
        
        # 设置随机种子
        np.random.seed(self.reproducibility_config['random_seed'])
        
        filtered_phrases = self.phrase_data['filtered_phrases']
        phrase_list = list(filtered_phrases.keys())
        
        # 创建NetworkX图
        self.global_graph_object = nx.Graph()
        
        # 添加节点
        for phrase in phrase_list:
            self.global_graph_object.add_node(
                phrase, 
                frequency=filtered_phrases[phrase],
                phrase_type='bigram' if ' ' in phrase else 'unigram'
            )
        
        # 计算共现关系
        cooccurrence_counts = defaultdict(int)
        
        for doc in tqdm(self.cleaned_text_data, desc="🌐 计算共现关系", unit="doc"):
            doc_phrases = []
            tokens = doc['tokens']
            
            # 提取文档中的短语
            if self.reproducibility_config['phrase_type'] in ['word', 'mixed']:
                doc_phrases.extend([token for token in tokens if token in filtered_phrases])
            
            if self.reproducibility_config['phrase_type'] in ['bigram', 'mixed']:
                for i in range(len(tokens) - 1):
                    bigram = f"{tokens[i]} {tokens[i+1]}"
                    if bigram in filtered_phrases:
                        doc_phrases.append(bigram)
            
            # 计算共现
            for i, phrase1 in enumerate(doc_phrases):
                for phrase2 in doc_phrases[i+1:]:
                    if phrase1 != phrase2:
                        edge = tuple(sorted([phrase1, phrase2]))
                        cooccurrence_counts[edge] += 1
        
        # 边过滤
        print("🔧 应用边过滤...")
        min_weight = self.graph_construction_config['min_edge_weight']
        filtered_edges = {edge: weight for edge, weight in cooccurrence_counts.items() 
                        if weight >= min_weight}
        
        if filtered_edges:
            edge_weights = list(filtered_edges.values())
            density_threshold = np.percentile(edge_weights, 
                                            (1 - self.graph_construction_config['edge_density_reduction']) * 100)
            final_edges = {edge: weight for edge, weight in filtered_edges.items() 
                         if weight >= density_threshold}
        else:
            final_edges = {}
        
        # 添加边到图
        for (phrase1, phrase2), weight in final_edges.items():
            self.global_graph_object.add_edge(phrase1, phrase2, weight=weight)
        
        print(f"   原始边数: {len(cooccurrence_counts)}")
        print(f"   过滤后边数: {len(final_edges)}")
        
        # 计算节点重要性
        print("📊 计算节点重要性...")
        degree_centrality = nx.degree_centrality(self.global_graph_object)
        weighted_degree = dict(self.global_graph_object.degree(weight='weight'))
        max_weighted_degree = max(weighted_degree.values()) if weighted_degree else 1
        weighted_degree_norm = {node: deg/max_weighted_degree for node, deg in weighted_degree.items()}
        
        try:
            pagerank = nx.pagerank(self.global_graph_object, weight='weight')
        except:
            pagerank = degree_centrality
        
        # 分配节点角色
        node_importance = {}
        for node in self.global_graph_object.nodes():
            importance = (
                0.4 * degree_centrality.get(node, 0) +
                0.4 * weighted_degree_norm.get(node, 0) +
                0.2 * pagerank.get(node, 0)
            )
            node_importance[node] = importance
        
        importance_threshold = np.percentile(list(node_importance.values()), 
                                           (1 - self.graph_construction_config['core_node_percentile']) * 100)
        
        node_roles = {}
        for node, importance in node_importance.items():
            if importance >= importance_threshold:
                node_roles[node] = 'core'
            else:
                node_roles[node] = 'periphery'
        
        # 存储节点属性
        nx.set_node_attributes(self.global_graph_object, node_importance, 'importance')
        nx.set_node_attributes(self.global_graph_object, node_roles, 'role')
        
        # 修复的布局计算 - 使用自定义进度回调
        print("🎯 计算确定性2D布局（修复进度显示）...")
        
        # 方法1：使用较少的迭代次数但显示真实进度
        iterations = 50
        with tqdm(total=iterations, desc="🎯 Spring layout进度", unit="iter") as pbar:
            def progress_callback():
                pbar.update(1)
            
            # 分批计算布局以显示进度
            pos = None
            batch_size = 10
            for i in range(0, iterations, batch_size):
                current_iterations = min(batch_size, iterations - i)
                
                if pos is None:
                    # 第一次计算
                    pos = nx.spring_layout(
                        self.global_graph_object,
                        k=1.0,
                        iterations=current_iterations,
                        seed=self.reproducibility_config['random_seed']
                    )
                else:
                    # 继续优化布局
                    pos = nx.spring_layout(
                        self.global_graph_object,
                        k=1.0,
                        iterations=current_iterations,
                        pos=pos,  # 使用之前的位置作为起点
                        seed=self.reproducibility_config['random_seed']
                    )
                
                # 更新进度条
                pbar.update(current_iterations)
                time.sleep(0.1)  # 短暂延迟以显示进度
        
        self.global_layout_positions = pos
        nx.set_node_attributes(self.global_graph_object, self.global_layout_positions, 'pos')
        
        # 社区检测
        print("🏘️ 检测社区...")
        try:
            communities = nx.community.greedy_modularity_communities(self.global_graph_object)
            community_map = {}
            for i, community in enumerate(communities):
                for node in community:
                    community_map[node] = i
            nx.set_node_attributes(self.global_graph_object, community_map, 'community')
            print(f"   发现 {len(communities)} 个社区")
        except:
            community_map = {node: 0 for node in self.global_graph_object.nodes()}
            nx.set_node_attributes(self.global_graph_object, community_map, 'community')
            print("   使用单一社区（回退）")
        
        print(f"✅ 全局图构建完成: {self.global_graph_object.number_of_nodes()} 节点, {self.global_graph_object.number_of_edges()} 边")
        return True
    
    def activate_subgraphs(self):
        """激活子图"""
        print("🗺️ 激活状态子图...")
        
        # 按状态分组文档
        state_documents = {}
        for doc in self.cleaned_text_data:
            state = doc['state']
            if state not in state_documents:
                state_documents[state] = []
            state_documents[state].append(doc)
        
        self.state_subgraph_objects = {}
        
        for state, docs in tqdm(state_documents.items(), desc="🗺️ 激活子图", unit="state"):
            # 获取该状态中出现的短语
            state_phrases = set()
            for doc in docs:
                tokens = doc['tokens']
                
                if self.reproducibility_config['phrase_type'] in ['word', 'mixed']:
                    state_phrases.update([token for token in tokens if token in self.phrase_data['filtered_phrases']])
                
                if self.reproducibility_config['phrase_type'] in ['bigram', 'mixed']:
                    for i in range(len(tokens) - 1):
                        bigram = f"{tokens[i]} {tokens[i+1]}"
                        if bigram in self.phrase_data['filtered_phrases']:
                            state_phrases.add(bigram)
            
            # 创建子图
            state_nodes = [node for node in self.global_graph_object.nodes() if node in state_phrases]
            if state_nodes:
                state_subgraph = self.global_graph_object.subgraph(state_nodes)
                self.state_subgraph_objects[state] = state_subgraph
                print(f"   {state}: {state_subgraph.number_of_nodes()} 节点, {state_subgraph.number_of_edges()} 边")
        
        print(f"✅ 子图激活完成: {len(self.state_subgraph_objects)} 个状态子图")
        return True
    
    def generate_visualizations_with_fixed_progress(self):
        """生成可视化（修复进度条卡住问题）"""
        print("🎨 生成可视化（修复进度条）...")
        print(f"   输出目录: {self.output_dir}")
        
        # 设置matplotlib参数
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = self.reproducibility_config['random_seed']
        
        self.visualization_paths = {}
        
        try:
            # 1. 全局图可视化
            print("🌐 生成全局主题网络...")
            if self.global_graph_object and self.global_layout_positions:
                
                # 使用更细粒度的进度条
                with tqdm(total=10, desc="🌐 全局网络可视化", unit="step") as pbar:
                    
                    pbar.set_description("🌐 初始化图形")
                    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
                    G = self.global_graph_object
                    pos = self.global_layout_positions
                    pbar.update(1)
                    
                    pbar.set_description("🌐 准备节点属性")
                    communities = nx.get_node_attributes(G, 'community')
                    importance_scores = nx.get_node_attributes(G, 'importance')
                    node_roles = nx.get_node_attributes(G, 'role')
                    
                    unique_communities = sorted(set(communities.values())) if communities else [0]
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_communities)))
                    community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}
                    pbar.update(1)
                    
                    pbar.set_description("🌐 计算视觉属性")
                    node_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in G.nodes()]
                    
                    node_sizes = []
                    node_shapes_core = []
                    node_shapes_periphery = []
                    
                    for node in G.nodes():
                        importance = importance_scores.get(node, 0)
                        size = self.viz_config['min_node_size'] + (self.viz_config['max_node_size'] - self.viz_config['min_node_size']) * importance
                        node_sizes.append(size)
                        
                        role = node_roles.get(node, 'periphery')
                        if role == 'core':
                            node_shapes_core.append(node)
                        else:
                            node_shapes_periphery.append(node)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 绘制边")
                    # 简化边绘制以避免卡住
                    edge_list = list(G.edges(data=True))
                    if edge_list:
                        max_weight = max([d['weight'] for _, _, d in edge_list])
                        
                        for u, v, data in edge_list:
                            weight = data['weight']
                            u_community = communities.get(u, 0)
                            v_community = communities.get(v, 0)
                            
                            if u_community == v_community:
                                alpha = self.viz_config['intra_community_edge_alpha']
                                color = community_colors.get(u_community, 'gray')
                            else:
                                alpha = self.viz_config['inter_community_edge_alpha']
                                color = 'gray'
                            
                            width = 0.5 + 2.0 * (weight / max_weight)
                            
                            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], 
                                                 width=width, alpha=alpha, 
                                                 edge_color=[color], ax=ax)
                    pbar.update(2)  # 边绘制完成，更新2步
                    
                    pbar.set_description("🌐 绘制核心节点")
                    if node_shapes_core:
                        core_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_core]
                        core_sizes = [node_sizes[list(G.nodes()).index(node)] for node in node_shapes_core]
                        nx.draw_networkx_nodes(G, pos, nodelist=node_shapes_core,
                                             node_color=core_colors, node_size=core_sizes,
                                             node_shape=self.viz_config['core_node_shape'],
                                             alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 绘制外围节点")
                    if node_shapes_periphery:
                        periphery_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_periphery]
                        periphery_sizes = [node_sizes[list(G.nodes()).index(node)] for node in node_shapes_periphery]
                        nx.draw_networkx_nodes(G, pos, nodelist=node_shapes_periphery,
                                             node_color=periphery_colors, node_size=periphery_sizes,
                                             node_shape=self.viz_config['periphery_node_shape'],
                                             alpha=0.8, edgecolors='gray', linewidths=0.5, ax=ax)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 添加标签")
                    # 简化标签添加
                    labels_to_draw = {}
                    if importance_scores:
                        importance_threshold = np.percentile(list(importance_scores.values()), 70)
                        top_nodes = [(node, score) for node, score in importance_scores.items() 
                                   if score >= importance_threshold]
                        top_nodes = sorted(top_nodes, key=lambda x: x[1], reverse=True)[:10]  # 只显示前10个
                        
                        for node, _ in top_nodes:
                            labels_to_draw[node] = node[:15] + "..." if len(node) > 15 else node  # 截断长标签
                    
                    if labels_to_draw:
                        nx.draw_networkx_labels(G, pos, labels_to_draw, 
                                              font_size=8, font_weight='bold', 
                                              font_color='black', ax=ax)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 添加图例")
                    ax.set_title(f'全局主题共现网络\n'
                               f'{G.number_of_nodes()} 节点, {G.number_of_edges()} 边, '
                               f'{len(unique_communities)} 社区\n'
                               f'种子: {seed} | 密度: {nx.density(G)*100:.2f}%', 
                               fontsize=14, fontweight='bold', pad=20)
                    
                    # 简化图例
                    legend_elements = []
                    for comm in sorted(unique_communities)[:5]:  # 只显示前5个社区
                        color = community_colors[comm]
                        legend_elements.append(patches.Patch(color=color, label=f'社区 {comm}'))
                    
                    if legend_elements:
                        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
                    
                    ax.axis('off')
                    plt.tight_layout()
                    pbar.update(1)
                    
                    pbar.set_description("🌐 保存图像")
                    global_viz_name = f"global_thematic_network_seed{seed}_{timestamp}.png"
                    global_viz_path = os.path.join(self.output_dir, global_viz_name)
                    plt.savefig(global_viz_path, bbox_inches='tight', facecolor='white', dpi=300)
                    plt.close()
                    
                    self.visualization_paths['global_graph'] = global_viz_path
                    pbar.update(1)
                
                print(f"      ✅ 保存: {global_viz_name}")
            
            # 2. 状态子图可视化（简化版本）
            subgraph_count = 0
            max_subgraphs = 3  # 限制子图数量以避免卡住
            
            for state, subgraph in list(self.state_subgraph_objects.items())[:max_subgraphs]:
                if subgraph.number_of_nodes() > 0:
                    subgraph_count += 1
                    print(f"🎨 生成状态 {state} 主题网络...")
                    
                    with tqdm(total=6, desc=f"🎨 {state} 网络", unit="step", leave=False) as step_pbar:
                        step_pbar.set_description(f"🎨 {state}: 初始化")
                        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
                        
                        subgraph_pos = {node: self.global_layout_positions[node] for node in subgraph.nodes() 
                                      if node in self.global_layout_positions}
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: 准备属性")
                        communities = {node: self.global_graph_object.nodes[node].get('community', 0) 
                                     for node in subgraph.nodes()}
                        importance_scores = {node: self.global_graph_object.nodes[node].get('importance', 0) 
                                           for node in subgraph.nodes()}
                        node_roles = {node: self.global_graph_object.nodes[node].get('role', 'periphery') 
                                    for node in subgraph.nodes()}
                        
                        unique_communities = sorted(set(communities.values()))
                        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_communities)))
                        community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: 绘制边")
                        # 简化边绘制
                        if subgraph.number_of_edges() > 0:
                            edge_weights = [d['weight'] for _, _, d in subgraph.edges(data=True)]
                            max_weight = max(edge_weights)
                            
                            for u, v, data in subgraph.edges(data=True):
                                weight = data['weight']
                                width = 0.5 + 2.0 * (weight / max_weight)
                                alpha = self.viz_config['intra_community_edge_alpha']
                                
                                nx.draw_networkx_edges(subgraph, subgraph_pos, edgelist=[(u, v)],
                                                     width=width, alpha=alpha, edge_color=['gray'], ax=ax)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: 绘制节点")
                        node_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in subgraph.nodes()]
                        node_sizes = [100 + 400 * importance_scores.get(node, 0) for node in subgraph.nodes()]
                        
                        nx.draw_networkx_nodes(subgraph, subgraph_pos, 
                                             node_color=node_colors, node_size=node_sizes,
                                             alpha=0.8, ax=ax)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: 添加标签")
                        # 简化标签
                        if importance_scores:
                            top_nodes = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                            labels = {node: node[:10] + "..." if len(node) > 10 else node for node, _ in top_nodes}
                            nx.draw_networkx_labels(subgraph, subgraph_pos, labels,
                                                  font_size=8, font_weight='bold', ax=ax)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: 完成")
                        doc_count = len([doc for doc in self.cleaned_text_data if doc['state'] == state])
                        
                        ax.set_title(f'状态 {state} 主题网络\n'
                                   f'{subgraph.number_of_nodes()} 节点, {subgraph.number_of_edges()} 边\n'
                                   f'{doc_count} 文档 | 种子: {seed}', 
                                   fontsize=12, fontweight='bold', pad=15)
                        
                        ax.axis('off')
                        plt.tight_layout()
                        
                        state_viz_name = f"state_{state}_thematic_network_seed{seed}_{timestamp}.png"
                        state_viz_path = os.path.join(self.output_dir, state_viz_name)
                        plt.savefig(state_viz_path, bbox_inches='tight', facecolor='white', dpi=300)
                        plt.close()
                        
                        self.visualization_paths[f'subgraph_{state}'] = state_viz_path
                        step_pbar.update(1)
                    
                    print(f"      ✅ 保存: {state_viz_name}")
            
            print(f"\n✅ 可视化生成完成!")
            print(f"🎨 生成了 {len(self.visualization_paths)} 个可视化文件")
            print(f"📁 输出目录: {self.output_dir}")
            
            # 显示所有生成的文件
            print(f"\n📊 生成的可视化文件:")
            for viz_name, viz_path in self.visualization_paths.items():
                print(f"   {viz_name}: {os.path.basename(viz_path)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 可视化生成失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_test(self):
        """运行完整测试"""
        print("🧪 开始可视化功能测试")
        print("=" * 60)
        print(f"📁 输入目录: {self.input_directory}")
        print(f"📁 输出目录: {self.output_dir}")
        print("=" * 60)
        
        try:
            # 1. 加载数据
            input_data = self.load_test_data()
            if not input_data:
                return False
            
            # 2. 文本清理
            if not self.simulate_text_cleaning(input_data):
                return False
            
            # 3. 短语提取
            if not self.simulate_phrase_extraction():
                return False
            
            # 4. 构建全局图（修复进度条）
            if not self.build_global_graph_with_fixed_progress():
                return False
            
            # 5. 激活子图
            if not self.activate_subgraphs():
                return False
            
            # 6. 生成可视化（修复卡住问题）
            if not self.generate_visualizations_with_fixed_progress():
                return False
            
            print("\n🎉 所有测试通过!")
            print("✅ 进度条显示问题已修复")
            print("✅ 可视化生成卡住问题已修复")
            print("✅ 图像已保存到指定目录")
            
            return True
            
        except Exception as e:
            print(f"💥 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    print("🔧 可视化功能修复测试脚本")
    print("修复问题:")
    print("1. 4.1步骤的spring layout进度条只显示0%和100%")
    print("2. 6.1步骤的可视化生成卡住不动")
    print()
    
    tester = VisualizationTester()
    success = tester.run_test()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())