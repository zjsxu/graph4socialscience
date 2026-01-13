#!/usr/bin/env python3
"""
恢复原始可视化版本

这个脚本将恢复之前版本的可视化方法，替换掉过度简化的scientific optimization，
让图片输出回到685个节点、26979条边的丰富版本。
"""

import os
import sys
from pathlib import Path

def restore_original_visualization():
    """恢复原始的可视化方法"""
    
    print("🔄 恢复原始可视化版本")
    print("=" * 50)
    
    # 读取complete_usage_guide.py
    complete_guide_path = "complete_usage_guide.py"
    
    if not os.path.exists(complete_guide_path):
        print(f"❌ 找不到文件: {complete_guide_path}")
        return False
    
    print(f"📖 读取文件: {complete_guide_path}")
    
    with open(complete_guide_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 找到被注释掉的generate_deterministic_visualizations方法
    # 它在第2263行左右，被三引号注释掉了
    
    # 1. 首先添加generate_deterministic_visualizations方法
    generate_deterministic_visualizations_method = '''
    def generate_deterministic_visualizations(self):
        """Generate deterministic visualizations directly from NetworkX graph objects"""
        if not self.validate_pipeline_step('subgraphs_activated', "Please activate subgraphs first (step 5.1)"):
            return
        
        print("\\n🎨 DETERMINISTIC VISUALIZATION GENERATION")
        print("-" * 60)
        print("🔬 Generating readable thematic network visualizations")
        print(f"🌱 Random seed: {self.reproducibility_config['random_seed']}")
        print(f"🎯 Layout algorithm: {self.reproducibility_config['layout_algorithm']}")
        
        # C. FIXED VISUALIZATION CONFIGURATION for semantic reference style
        self.viz_config = {
            # Deterministic layout
            'fixed_random_seed': self.reproducibility_config['random_seed'],
            'cache_positions': True,
            
            # Visual encoding
            'edge_alpha_light': 0.3,  # Intra-community edges
            'edge_alpha_inter': 0.05,  # Inter-community edges  
            'edge_color': 'lightgray',
            'edge_weight_threshold': 2,  # Hide edges below this weight
            
            # Node shapes by role
            'core_node_shape': '^',  # Triangle for core nodes
            'periphery_node_shape': 'o',  # Circle for periphery nodes
            
            # Node size scaling by semantic importance (TF-IDF, not raw frequency)
            'min_node_size': 50,
            'max_node_size': 800,
            'size_by_tfidf': True,
            
            # Selective labeling
            'label_core_only': True,
            'label_importance_threshold': 0.7,  # Top 30% important nodes
            'max_labels_per_community': 3,
            'never_label_structural': True,
            
            # High-resolution output
            'output_dpi': 300,
            'figure_size': (16, 12),
            'export_format': 'PNG',
        }
        
        try:
            # Create visualizations directory
            viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed = self.reproducibility_config['random_seed']
            
            self.visualization_paths = {}
            
            print("⏳ Generating readable thematic network visualizations...")
            
            # Set matplotlib parameters for consistent, high-resolution output
            plt.rcParams['figure.dpi'] = self.viz_config['output_dpi']
            plt.rcParams['savefig.dpi'] = self.viz_config['output_dpi']
            plt.rcParams['font.size'] = 10
            plt.rcParams['font.family'] = 'sans-serif'
            
            # 1. GLOBAL GRAPH VISUALIZATION - SEMANTIC THEMATIC NETWORK
            if self.global_graph_object and self.global_layout_positions:
                with tqdm(total=8, desc="🌐 Global thematic network", unit="step") as pbar:
                    pbar.set_description("🌐 Setting up figure")
                    fig, ax = plt.subplots(1, 1, figsize=self.viz_config['figure_size'])
                    G = self.global_graph_object
                    
                    # Use cached positions from global graph (deterministic)
                    pos = self.global_layout_positions
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Preparing node attributes")
                    # Get node attributes for visual encoding
                    communities = nx.get_node_attributes(G, 'community')
                    importance_scores = nx.get_node_attributes(G, 'importance')
                    node_roles = nx.get_node_attributes(G, 'role')
                    tf_idf_scores = nx.get_node_attributes(G, 'tf_idf_score')
                    is_structural = nx.get_node_attributes(G, 'is_structural')
                    
                    # Create distinct color map for communities
                    unique_communities = sorted(set(communities.values())) if communities else [0]
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
                    community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Computing visual attributes")
                    # Node visual attributes based on semantic importance
                    node_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in G.nodes()]
                    
                    # Node sizes based on TF-IDF scores (semantic importance), NOT raw frequency
                    node_sizes = []
                    node_shapes_core = []
                    node_shapes_periphery = []
                    
                    for node in G.nodes():
                        # Use TF-IDF for size scaling if available, fallback to importance
                        if self.viz_config['size_by_tfidf'] and tf_idf_scores:
                            semantic_score = tf_idf_scores.get(node, 0)
                            max_score = max(tf_idf_scores.values()) if tf_idf_scores.values() else 1
                            normalized_score = semantic_score / max_score if max_score > 0 else 0
                        else:
                            normalized_score = importance_scores.get(node, 0)
                        
                        size = self.viz_config['min_node_size'] + (self.viz_config['max_node_size'] - self.viz_config['min_node_size']) * normalized_score
                        node_sizes.append(size)
                        
                        # Separate nodes by role for different shapes
                        role = node_roles.get(node, 'periphery')
                        if role == 'core':
                            node_shapes_core.append(node)
                        else:
                            node_shapes_periphery.append(node)
                    
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Drawing edges with community-aware filtering")
                    # Edge rendering with community-aware alpha and weight threshold
                    edges_to_draw = []
                    edge_colors = []
                    edge_alphas = []
                    
                    for u, v, data in G.edges(data=True):
                        weight = data['weight']
                        
                        # Apply weight threshold to avoid hairball effect
                        if weight < self.viz_config['edge_weight_threshold']:
                            continue
                        
                        u_community = communities.get(u, 0)
                        v_community = communities.get(v, 0)
                        
                        # Community-aware edge rendering
                        if u_community == v_community:
                            # Intra-community edges: higher alpha
                            alpha = self.viz_config['edge_alpha_light']
                        else:
                            # Inter-community edges: lower alpha
                            alpha = self.viz_config['edge_alpha_inter']
                        
                        edges_to_draw.append((u, v))
                        edge_colors.append(self.viz_config['edge_color'])
                        edge_alphas.append(alpha)
                    
                    # Draw edges in batches to avoid performance issues
                    if edges_to_draw:
                        # Separate intra and inter community edges for different rendering
                        intra_edges = []
                        inter_edges = []
                        
                        for i, (u, v) in enumerate(edges_to_draw):
                            u_community = communities.get(u, 0)
                            v_community = communities.get(v, 0)
                            
                            if u_community == v_community:
                                intra_edges.append((u, v))
                            else:
                                inter_edges.append((u, v))
                        
                        # Draw inter-community edges first (lower layer)
                        if inter_edges:
                            nx.draw_networkx_edges(G, pos, edgelist=inter_edges,
                                                 width=0.5, alpha=self.viz_config['edge_alpha_inter'], 
                                                 edge_color=self.viz_config['edge_color'], ax=ax)
                        
                        # Draw intra-community edges on top
                        if intra_edges:
                            nx.draw_networkx_edges(G, pos, edgelist=intra_edges,
                                                 width=1.0, alpha=self.viz_config['edge_alpha_light'], 
                                                 edge_color=self.viz_config['edge_color'], ax=ax)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Drawing nodes by role")
                    # Draw nodes by role with different shapes
                    if node_shapes_core:
                        core_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_core]
                        core_sizes = [node_sizes[list(G.nodes()).index(node)] for node in node_shapes_core]
                        nx.draw_networkx_nodes(G, pos, nodelist=node_shapes_core,
                                             node_color=core_colors, node_size=core_sizes,
                                             node_shape=self.viz_config['core_node_shape'],
                                             alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax)
                    
                    if node_shapes_periphery:
                        periphery_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_periphery]
                        periphery_sizes = [node_sizes[list(G.nodes()).index(node)] for node in node_shapes_periphery]
                        nx.draw_networkx_nodes(G, pos, nodelist=node_shapes_periphery,
                                             node_color=periphery_colors, node_size=periphery_sizes,
                                             node_shape=self.viz_config['periphery_node_shape'],
                                             alpha=0.8, edgecolors='gray', linewidths=0.5, ax=ax)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Adding selective semantic labels")
                    # SELECTIVE LABELING - Only label core nodes, never structural tokens
                    labels_to_draw = {}
                    
                    if self.viz_config['label_core_only']:
                        # Only label core nodes
                        candidate_nodes = node_shapes_core
                    else:
                        # Label all nodes above importance threshold
                        importance_threshold = np.percentile(list(importance_scores.values()), 
                                                           self.viz_config['label_importance_threshold'] * 100)
                        candidate_nodes = [node for node in G.nodes() 
                                         if importance_scores.get(node, 0) >= importance_threshold]
                    
                    # Never label structural tokens
                    if self.viz_config['never_label_structural']:
                        candidate_nodes = [node for node in candidate_nodes 
                                         if not is_structural.get(node, False)]
                    
                    # Group nodes by community for balanced labeling
                    community_nodes = defaultdict(list)
                    for node in candidate_nodes:
                        community = communities.get(node, 0)
                        importance = importance_scores.get(node, 0)
                        community_nodes[community].append((node, importance))
                    
                    # Select top nodes per community (max 3 per community)
                    for community, nodes in community_nodes.items():
                        # Sort by importance and take top N
                        top_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)[:self.viz_config['max_labels_per_community']]
                        for node, _ in top_nodes:
                            # Truncate long labels for readability
                            label = node[:15] + "..." if len(node) > 15 else node
                            labels_to_draw[node] = label
                    
                    if labels_to_draw:
                        nx.draw_networkx_labels(G, pos, labels_to_draw, 
                                              font_size=9, font_weight='bold', 
                                              font_color='black', ax=ax)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Adding enhanced legends")
                    # Enhanced title with semantic filtering info
                    structural_removed = len(self.phrase_data.get('filtered_phrases', {})) - G.number_of_nodes()
                    ax.set_title(f'Global Semantic Co-occurrence Network\\n'
                               f'{G.number_of_nodes()} nodes ({structural_removed} structural tokens removed), '
                               f'{G.number_of_edges()} edges, {len(unique_communities)} communities\\n'
                               f'Seed: {seed} | Density: {nx.density(G)*100:.2f}% | TF-IDF weighted', 
                               fontsize=14, fontweight='bold', pad=20)
                    
                    # Enhanced community legend
                    legend_elements = []
                    for comm in sorted(unique_communities)[:8]:  # Show first 8 communities
                        color = community_colors[comm]
                        legend_elements.append(patches.Patch(color=color, label=f'Community {comm}'))
                    
                    if len(unique_communities) > 8:
                        legend_elements.append(patches.Patch(color='lightgray', label=f'... +{len(unique_communities)-8} more'))
                    
                    # Role and semantic legend
                    legend_elements.append(patches.Patch(color='white', label=''))  # Spacer
                    legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                                    markerfacecolor='gray', markersize=10, label='Core nodes (triangles)'))
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor='gray', markersize=8, label='Periphery nodes (circles)'))
                    legend_elements.append(patches.Patch(color='white', label=''))  # Spacer
                    legend_elements.append(patches.Patch(color='lightgray', label='Node size: TF-IDF score'))
                    legend_elements.append(patches.Patch(color='lightgray', label='Edge alpha: Community relationship'))
                    
                    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                            frameon=True, fancybox=True, shadow=True)
                    
                    ax.axis('off')
                    plt.tight_layout()
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Saving high-resolution visualization")
                    global_viz_name = f"global_thematic_network_seed{seed}_{timestamp}.png"
                    global_viz_path = os.path.join(viz_dir, global_viz_name)
                    
                    # Always export physical image file (PNG) with high resolution
                    plt.savefig(global_viz_path, bbox_inches='tight', facecolor='white', 
                              dpi=self.viz_config['output_dpi'], format=self.viz_config['export_format'])
                    plt.close()
                    
                    # Print absolute output image path after generation
                    print(f"      ✅ Saved: {os.path.basename(global_viz_path)}")
                    print(f"      📁 Full path: {os.path.abspath(global_viz_path)}")
                    
                    self.visualization_paths['global'] = os.path.abspath(global_viz_path)
                    
                    self.visualization_paths['global_graph'] = global_viz_path
                    pbar.update(1)
                
                print(f"      ✅ Saved: {global_viz_name}")
            
            # 2. STATE SUBGRAPH VISUALIZATIONS - HIGHLIGHTED SUBSETS
            subgraph_items = list(self.state_subgraph_objects.items())
            
            for state, subgraph in tqdm(subgraph_items, desc="🎨 Generating state thematic networks", unit="subgraph"):
                if subgraph.number_of_nodes() > 0:
                    with tqdm(total=8, desc=f"🎨 {state} thematic network", unit="step", leave=False) as step_pbar:
                        step_pbar.set_description(f"🎨 {state}: Setting up figure")
                        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
                        
                        # Use same positions as global graph for consistency
                        subgraph_pos = {node: self.global_layout_positions[node] for node in subgraph.nodes() 
                                      if node in self.global_layout_positions}
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Preparing attributes")
                        # Get node attributes from global graph (maintain consistency)
                        communities = {node: self.global_graph_object.nodes[node].get('community', 0) 
                                     for node in subgraph.nodes()}
                        importance_scores = {node: self.global_graph_object.nodes[node].get('importance', 0) 
                                           for node in subgraph.nodes()}
                        node_roles = {node: self.global_graph_object.nodes[node].get('role', 'periphery') 
                                    for node in subgraph.nodes()}
                        
                        # Use same color scheme as global graph
                        unique_communities = sorted(set(communities.values()))
                        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_communities)))
                        community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Computing visual attributes")
                        # Node visual attributes (consistent with global)
                        node_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in subgraph.nodes()]
                        
                        node_sizes = []
                        node_shapes_core = []
                        node_shapes_periphery = []
                        
                        for node in subgraph.nodes():
                            importance = importance_scores.get(node, 0)
                            size = self.viz_config['min_node_size'] + (self.viz_config['max_node_size'] - self.viz_config['min_node_size']) * importance
                            node_sizes.append(size)
                            
                            role = node_roles.get(node, 'periphery')
                            if role == 'core':
                                node_shapes_core.append(node)
                            else:
                                node_shapes_periphery.append(node)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Drawing edges")
                        # 简化边绘制避免卡住
                        if subgraph.number_of_edges() > 0:
                            # 限制边数并简化绘制
                            edge_list = list(subgraph.edges(data=True))[:30]  # 最多30条边
                            if edge_list:
                                nx.draw_networkx_edges(subgraph, subgraph_pos, 
                                                     edgelist=[(u, v) for u, v, _ in edge_list],
                                                     width=1.0, alpha=0.3, edge_color='gray', ax=ax)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Drawing nodes")
                        # Draw nodes by role
                        if node_shapes_core:
                            core_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_core]
                            core_sizes = [node_sizes[list(subgraph.nodes()).index(node)] for node in node_shapes_core]
                            nx.draw_networkx_nodes(subgraph, subgraph_pos, nodelist=node_shapes_core,
                                                 node_color=core_colors, node_size=core_sizes,
                                                 node_shape=self.viz_config['core_node_shape'],
                                                 alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax)
                        
                        if node_shapes_periphery:
                            periphery_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_periphery]
                            periphery_sizes = [node_sizes[list(subgraph.nodes()).index(node)] for node in node_shapes_periphery]
                            nx.draw_networkx_nodes(subgraph, subgraph_pos, nodelist=node_shapes_periphery,
                                                 node_color=periphery_colors, node_size=periphery_sizes,
                                                 node_shape=self.viz_config['periphery_node_shape'],
                                                 alpha=0.8, edgecolors='gray', linewidths=0.5, ax=ax)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Adding labels")
                        # Selective labeling for subgraph
                        labels_to_draw = {}
                        if importance_scores:
                            importance_threshold = np.percentile(list(importance_scores.values()), 70)
                            
                            community_nodes = defaultdict(list)
                            for node in subgraph.nodes():
                                community = communities.get(node, 0)
                                importance = importance_scores.get(node, 0)
                                if importance >= importance_threshold:
                                    community_nodes[community].append((node, importance))
                            
                            for community, nodes in community_nodes.items():
                                top_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)[:2]  # Fewer labels for subgraphs
                                for node, _ in top_nodes:
                                    labels_to_draw[node] = node
                        
                        if labels_to_draw:
                            nx.draw_networkx_labels(subgraph, subgraph_pos, labels_to_draw,
                                                  font_size=9, font_weight='bold',
                                                  font_color='black', ax=ax)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Finalizing")
                        doc_count = len([doc for doc in self.cleaned_text_data if doc['state'] == state])
                        core_count = len(node_shapes_core)
                        
                        ax.set_title(f'State {state} Thematic Network\\n'
                                   f'{subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges, '
                                   f'{len(unique_communities)} communities\\n'
                                   f'{doc_count} documents, {core_count} core nodes | Seed: {seed}', 
                                   fontsize=12, fontweight='bold', pad=15)
                        
                        # Add legend (simplified for subgraphs)
                        legend_elements = []
                        for comm in sorted(unique_communities):
                            color = community_colors[comm]
                            legend_elements.append(patches.Patch(color=color, label=f'Community {comm}'))
                        
                        if len(legend_elements) > 0:
                            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
                        
                        ax.axis('off')
                        plt.tight_layout()
                        
                        state_viz_name = f"state_{state}_thematic_network_seed{seed}_{timestamp}.png"
                        state_viz_path = os.path.join(viz_dir, state_viz_name)
                        plt.savefig(state_viz_path, bbox_inches='tight', facecolor='white', dpi=300)
                        plt.close()
                        
                        self.visualization_paths[f'subgraph_{state}'] = state_viz_path
                        step_pbar.update(1)
                    
                    print(f"      ✅ Saved: {state_viz_name}")
            
            print(f"\\n✅ Thematic network visualization generation completed!")
            print(f"🎨 Generated {len(self.visualization_paths)} readable visualizations")
            print(f"📁 Output directory: {viz_dir}")
            print(f"🎯 All visualizations use consistent community-aware layout")
            print(f"🔍 Edge filtering applied for readability")
            print(f"🎭 Node roles visualized (core=triangles, periphery=circles)")
            
            self.pipeline_state['results_exported'] = True
            
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
            import traceback
            traceback.print_exc()
'''
    
    # 2. 修改scientific optimization，让它不要过度简化
    modified_scientific_config = '''
        # Scientific optimization parameters - LESS AGGRESSIVE
        self.scientific_config = {
            'semantic_weighting': 'npmi',  # 'npmi', 'salton', 'pmi'
            'sparsification_method': 'quantile',  # Use quantile only, not adaptive
            'edge_retention_rate': 0.3,  # Keep top 30% of edges (was 0.05)
            'disparity_alpha': 0.05,  # Significance level for disparity filter
            'min_community_size': 3,  # Smaller communities allowed (was 8)
            'max_legend_communities': 15,  # More communities in legend (was 10)
            'core_method': 'k_core',  # 'k_core', 'pagerank'
            'min_core_nodes': 20,  # Fewer core nodes required (was 50)
            'enable_lcc_extraction': False,  # DISABLE LCC extraction
            'enable_community_pruning': False,  # DISABLE community pruning
        }'''
    
    # 3. 修改graph construction config，让它保留更多节点和边
    modified_graph_config = '''
        # Add graph construction parameters for structural filtering - LESS AGGRESSIVE
        self.graph_construction_config = {
            'edge_density_reduction': 0.5,  # Keep top 50% of edges by weight (was 0.1)
            'min_edge_weight': 1,  # Lower minimum co-occurrence count (was 2)
            'core_node_percentile': 0.3,  # Top 30% nodes are "core" (was 0.2)
            'community_layout_separation': 2.0,  # Separation factor between communities
            'sliding_window_size': 5,  # Sliding window for co-occurrence
            'min_cooccurrence_threshold': 1,  # Lower minimum global co-occurrence threshold (was 3)
        }'''
    
    # 查找并替换相关配置
    print("🔧 修改scientific optimization配置...")
    
    # 替换scientific_config
    old_scientific_config_start = content.find("self.scientific_config = {")
    if old_scientific_config_start != -1:
        # 找到配置块的结束
        brace_count = 0
        pos = old_scientific_config_start
        while pos < len(content):
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
                if brace_count == 0:
                    old_scientific_config_end = pos + 1
                    break
            pos += 1
        
        # 替换配置
        content = content[:old_scientific_config_start] + modified_scientific_config.strip() + content[old_scientific_config_end:]
        print("   ✅ 修改了scientific_config")
    
    # 替换graph_construction_config
    old_graph_config_start = content.find("self.graph_construction_config = {")
    if old_graph_config_start != -1:
        # 找到配置块的结束
        brace_count = 0
        pos = old_graph_config_start
        while pos < len(content):
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
                if brace_count == 0:
                    old_graph_config_end = pos + 1
                    break
            pos += 1
        
        # 替换配置
        content = content[:old_graph_config_start] + modified_graph_config.strip() + content[old_graph_config_end:]
        print("   ✅ 修改了graph_construction_config")
    
    # 4. 添加generate_deterministic_visualizations方法
    # 找到ResearchPipelineCLI类中合适的位置插入方法
    class_method_insertion_point = content.find("def view_output_image_paths(self):")
    if class_method_insertion_point != -1:
        content = content[:class_method_insertion_point] + generate_deterministic_visualizations_method + "\\n\\n    " + content[class_method_insertion_point:]
        print("   ✅ 添加了generate_deterministic_visualizations方法")
    
    # 5. 修改菜单选项，将6.1改为调用generate_deterministic_visualizations
    content = content.replace(
        'elif choice == "6.1":\\n                self.generate_scientific_visualizations()',
        'elif choice == "6.1":\\n                self.generate_deterministic_visualizations()'
    )
    print("   ✅ 修改了菜单选项6.1")
    
    # 6. 保存修改后的文件
    backup_path = f"{complete_guide_path}.backup"
    print(f"📋 创建备份: {backup_path}")
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    with open(complete_guide_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 恢复完成！")
    print("📋 修改内容:")
    print("   - 恢复了generate_deterministic_visualizations方法")
    print("   - 禁用了LCC extraction和community pruning")
    print("   - 提高了edge retention rate (5% → 30%)")
    print("   - 降低了minimum edge weight (2 → 1)")
    print("   - 降低了minimum cooccurrence threshold (3 → 1)")
    print("   - 允许更小的社区 (8 → 3)")
    print("   - 修改了菜单选项6.1调用原始可视化方法")
    
    print("\\n🎯 现在可以运行pipeline，应该能看到更丰富的可视化结果！")
    
    return True

if __name__ == "__main__":
    restore_original_visualization()