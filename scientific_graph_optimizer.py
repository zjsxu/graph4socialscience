#!/usr/bin/env python3
"""
Scientific Graph Optimization Module
Implements rigorous network science methods for semantic co-occurrence analysis

Key Features:
- NPMI/Salton's Cosine semantic weighting
- Adaptive graph sparsification (Disparity Filter + Quantile-based)
- LCC extraction and community pruning
- K-Core decomposition for core-periphery identification
- Deterministic layouts with scientific reporting
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from collections import defaultdict, Counter
from tqdm import tqdm
import math
from scipy import sparse
from scipy.stats import entropy
import community as community_louvain  # python-louvain
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class ScientificGraphOptimizer:
    """
    Scientific optimization of co-occurrence networks with rigorous methods
    """
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Scientific parameters
        self.config = {
            # Semantic weighting
            'semantic_weighting': 'npmi',  # 'npmi', 'salton', 'pmi'
            'min_cooccurrence': 3,
            
            # Graph sparsification
            'sparsification_method': 'adaptive',  # 'quantile', 'disparity', 'adaptive'
            'edge_retention_rate': 0.05,  # Keep top 5% of edges
            'disparity_alpha': 0.05,  # Significance level for disparity filter
            
            # Community detection
            'community_algorithm': 'louvain',  # 'louvain', 'leiden'
            'min_community_size': 8,  # Collapse smaller communities
            'max_legend_communities': 10,
            
            # Core-periphery
            'core_method': 'k_core',  # 'k_core', 'pagerank'
            'min_core_nodes': 50,
            'core_percentile': 0.15,  # Top 15% for PageRank method
            
            # Layout and visualization
            'layout_k_factor': 1.0,  # k = k_factor / sqrt(n)
            'layout_iterations': 1000,
            'node_size_log_base': 2,
            'edge_alpha': 0.2,
            'label_centrality_threshold': 0.8,  # Top 20% nodes get labels
        }
    
    def compute_semantic_weights(self, cooccurrence_matrix, phrase_frequencies, total_phrases):
        """
        Compute semantic edge weights using NPMI or Salton's Cosine
        
        Args:
            cooccurrence_matrix: dict of {(phrase1, phrase2): count}
            phrase_frequencies: dict of {phrase: frequency}
            total_phrases: total number of phrase instances
            
        Returns:
            dict of {(phrase1, phrase2): semantic_weight}
        """
        print(f"🧮 Computing semantic weights using {self.config['semantic_weighting'].upper()}...")
        
        semantic_weights = {}
        method = self.config['semantic_weighting']
        
        with tqdm(cooccurrence_matrix.items(), desc="Computing semantic weights", unit="edge") as pbar:
            for (phrase1, phrase2), cooccur_count in pbar:
                if cooccur_count < self.config['min_cooccurrence']:
                    continue
                
                freq1 = phrase_frequencies.get(phrase1, 0)
                freq2 = phrase_frequencies.get(phrase2, 0)
                
                if freq1 == 0 or freq2 == 0:
                    continue
                
                # Probabilities
                p_x = freq1 / total_phrases
                p_y = freq2 / total_phrases
                p_xy = cooccur_count / total_phrases
                
                if method == 'npmi':
                    # Normalized Pointwise Mutual Information
                    if p_xy > 0:
                        pmi = math.log(p_xy / (p_x * p_y))
                        npmi = pmi / (-math.log(p_xy))
                        semantic_weights[(phrase1, phrase2)] = max(0, npmi)  # Ensure non-negative
                
                elif method == 'salton':
                    # Salton's Cosine Coefficient
                    salton = cooccur_count / math.sqrt(freq1 * freq2)
                    semantic_weights[(phrase1, phrase2)] = salton
                
                elif method == 'pmi':
                    # Standard PMI
                    if p_xy > 0:
                        pmi = math.log(p_xy / (p_x * p_y))
                        semantic_weights[(phrase1, phrase2)] = max(0, pmi)
        
        print(f"   ✅ Computed {len(semantic_weights)} semantic edge weights")
        return semantic_weights
    
    def apply_disparity_filter(self, graph, alpha=0.05):
        """
        Apply disparity filter for backbone extraction
        Preserves multi-scale structures by keeping statistically significant edges
        
        Args:
            graph: NetworkX graph
            alpha: significance level
            
        Returns:
            NetworkX graph with filtered edges
        """
        print(f"🔬 Applying disparity filter (α={alpha})...")
        
        filtered_graph = graph.copy()
        edges_to_remove = []
        
        for node in tqdm(graph.nodes(), desc="Disparity filtering", unit="node"):
            neighbors = list(graph.neighbors(node))
            if len(neighbors) <= 1:
                continue
            
            # Get edge weights for this node
            weights = []
            for neighbor in neighbors:
                weight = graph[node][neighbor].get('weight', 1.0)
                weights.append((neighbor, weight))
            
            # Calculate strength (sum of weights)
            strength = sum(w for _, w in weights)
            if strength == 0:
                continue
            
            # Calculate p-values for each edge
            for neighbor, weight in weights:
                p_ij = weight / strength
                k = len(neighbors)
                
                # Disparity measure: probability that a random variable 
                # with uniform distribution gives a value >= p_ij
                if k > 1:
                    disparity = (1 - p_ij) ** (k - 1)
                    
                    # Remove edge if not statistically significant
                    if disparity > alpha:
                        edges_to_remove.append((node, neighbor))
        
        # Remove non-significant edges
        filtered_graph.remove_edges_from(edges_to_remove)
        
        print(f"   📊 Removed {len(edges_to_remove)} non-significant edges")
        print(f"   📊 Retained {filtered_graph.number_of_edges()} edges")
        
        return filtered_graph
    
    def apply_quantile_sparsification(self, graph, retention_rate=0.05):
        """
        Keep only top quantile of edges by weight
        
        Args:
            graph: NetworkX graph
            retention_rate: fraction of edges to retain
            
        Returns:
            NetworkX graph with top edges only
        """
        print(f"📊 Applying quantile sparsification (retain top {retention_rate*100:.1f}%)...")
        
        # Get all edge weights
        edge_weights = []
        for u, v, data in graph.edges(data=True):
            weight = data.get('weight', 1.0)
            edge_weights.append(((u, v), weight))
        
        # Sort by weight and keep top fraction
        edge_weights.sort(key=lambda x: x[1], reverse=True)
        n_keep = max(1, int(len(edge_weights) * retention_rate))
        top_edges = edge_weights[:n_keep]
        
        # Create new graph with only top edges
        filtered_graph = nx.Graph()
        filtered_graph.add_nodes_from(graph.nodes(data=True))
        
        for (u, v), weight in top_edges:
            filtered_graph.add_edge(u, v, **graph[u][v])
        
        print(f"   📊 Retained {filtered_graph.number_of_edges()} / {graph.number_of_edges()} edges")
        
        return filtered_graph
    
    def extract_largest_connected_component(self, graph):
        """
        Extract the Largest Connected Component (LCC) for visualization
        
        Args:
            graph: NetworkX graph
            
        Returns:
            NetworkX graph (LCC only)
        """
        print("🔗 Extracting Largest Connected Component (LCC)...")
        
        if graph.number_of_nodes() == 0:
            return graph
        
        # Find all connected components
        components = list(nx.connected_components(graph))
        
        if not components:
            return nx.Graph()
        
        # Get the largest component
        largest_component = max(components, key=len)
        lcc = graph.subgraph(largest_component).copy()
        
        print(f"   📊 LCC size: {lcc.number_of_nodes()} nodes, {lcc.number_of_edges()} edges")
        print(f"   📊 Removed {graph.number_of_nodes() - lcc.number_of_nodes()} isolated nodes")
        
        return lcc
    
    def detect_and_prune_communities(self, graph):
        """
        Detect communities and prune small ones
        
        Args:
            graph: NetworkX graph
            
        Returns:
            dict: {node: community_id} with pruned communities
        """
        print("🏘️ Detecting and pruning communities...")
        
        if graph.number_of_edges() == 0:
            # No edges, assign all nodes to community 0
            return {node: 0 for node in graph.nodes()}
        
        # Detect communities using Louvain algorithm
        try:
            partition = community_louvain.best_partition(graph, weight='weight', random_state=self.random_seed)
        except:
            # Fallback: assign all nodes to single community
            partition = {node: 0 for node in graph.nodes()}
        
        # Count community sizes
        community_sizes = Counter(partition.values())
        
        # Identify small communities to collapse
        small_communities = {comm_id for comm_id, size in community_sizes.items() 
                           if size < self.config['min_community_size']}
        
        # Reassign small communities to "Other" category
        other_community_id = max(community_sizes.keys()) + 1 if community_sizes else 0
        
        pruned_partition = {}
        for node, comm_id in partition.items():
            if comm_id in small_communities:
                pruned_partition[node] = other_community_id
            else:
                pruned_partition[node] = comm_id
        
        # Renumber communities to be consecutive
        unique_communities = sorted(set(pruned_partition.values()))
        community_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_communities)}
        
        final_partition = {node: community_mapping[comm_id] 
                          for node, comm_id in pruned_partition.items()}
        
        final_community_sizes = Counter(final_partition.values())
        
        print(f"   📊 Original communities: {len(community_sizes)}")
        print(f"   📊 Small communities collapsed: {len(small_communities)}")
        print(f"   📊 Final communities: {len(final_community_sizes)}")
        print(f"   📊 Community sizes: {dict(sorted(final_community_sizes.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        return final_partition
    
    def identify_core_periphery(self, graph):
        """
        Identify core and periphery nodes using K-Core decomposition or PageRank
        
        Args:
            graph: NetworkX graph
            
        Returns:
            dict: {node: 'core' or 'periphery'}
        """
        print(f"🎯 Identifying core-periphery structure using {self.config['core_method']}...")
        
        if graph.number_of_nodes() == 0:
            return {}
        
        node_roles = {}
        
        if self.config['core_method'] == 'k_core':
            # K-Core decomposition
            core_numbers = nx.core_number(graph)
            
            # Find maximum k such that |nodes in k-core| >= min_core_nodes
            max_k = max(core_numbers.values()) if core_numbers else 0
            
            for k in range(max_k, 0, -1):
                k_core_nodes = [node for node, core_num in core_numbers.items() if core_num >= k]
                if len(k_core_nodes) >= self.config['min_core_nodes']:
                    print(f"   🎯 Using {k}-core with {len(k_core_nodes)} nodes as core")
                    
                    for node in graph.nodes():
                        if core_numbers[node] >= k:
                            node_roles[node] = 'core'
                        else:
                            node_roles[node] = 'periphery'
                    break
            else:
                # Fallback: use top nodes by degree
                degrees = dict(graph.degree())
                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                n_core = min(self.config['min_core_nodes'], len(sorted_nodes))
                
                core_nodes = set(node for node, _ in sorted_nodes[:n_core])
                for node in graph.nodes():
                    node_roles[node] = 'core' if node in core_nodes else 'periphery'
                
                print(f"   🎯 Fallback: Using top {n_core} nodes by degree as core")
        
        elif self.config['core_method'] == 'pagerank':
            # PageRank-based core identification
            pagerank = nx.pagerank(graph, weight='weight')
            threshold = np.percentile(list(pagerank.values()), 
                                    (1 - self.config['core_percentile']) * 100)
            
            core_count = 0
            for node, pr_score in pagerank.items():
                if pr_score >= threshold:
                    node_roles[node] = 'core'
                    core_count += 1
                else:
                    node_roles[node] = 'periphery'
            
            print(f"   🎯 PageRank threshold: {threshold:.6f}")
            print(f"   🎯 Core nodes: {core_count}")
        
        core_count = sum(1 for role in node_roles.values() if role == 'core')
        periphery_count = len(node_roles) - core_count
        
        print(f"   📊 Core nodes: {core_count} ({core_count/len(node_roles)*100:.1f}%)")
        print(f"   📊 Periphery nodes: {periphery_count} ({periphery_count/len(node_roles)*100:.1f}%)")
        
        return node_roles
    
    def compute_deterministic_layout(self, graph):
        """
        Compute deterministic force-directed layout with scientific parameters
        
        Args:
            graph: NetworkX graph
            
        Returns:
            dict: {node: (x, y)} positions
        """
        print("🎯 Computing deterministic force-directed layout...")
        
        if graph.number_of_nodes() == 0:
            return {}
        
        # Scientific k parameter: k = k_factor / sqrt(n)
        n_nodes = graph.number_of_nodes()
        k_param = self.config['layout_k_factor'] / math.sqrt(n_nodes)
        
        print(f"   📊 Layout parameters: k={k_param:.4f}, iterations={self.config['layout_iterations']}")
        
        # Compute layout with progress bar
        with tqdm(total=self.config['layout_iterations'], desc="🎯 Force-directed layout", unit="iter") as pbar:
            pos = nx.spring_layout(
                graph,
                k=k_param,
                iterations=self.config['layout_iterations'],
                seed=self.random_seed,
                weight='weight'
            )
            pbar.update(self.config['layout_iterations'])
        
        print(f"   ✅ Layout computed for {len(pos)} nodes")
        return pos
    
    def generate_scientific_visualization(self, graph, communities, node_roles, positions, 
                                        output_path, title="Scientific Co-occurrence Network"):
        """
        Generate publication-quality visualization with scientific styling
        
        Args:
            graph: NetworkX graph
            communities: dict {node: community_id}
            node_roles: dict {node: 'core'/'periphery'}
            positions: dict {node: (x, y)}
            output_path: str, path to save visualization
            title: str, plot title
        """
        print(f"🎨 Generating scientific visualization: {title}")
        
        if graph.number_of_nodes() == 0:
            print("   ⚠️ Empty graph, skipping visualization")
            return
        
        # Set up publication-quality figure
        plt.style.use('default')  # Clean style
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.set_facecolor('white')
        
        # Prepare node attributes
        node_colors = []
        node_sizes = []
        node_shapes_core = []
        node_shapes_periphery = []
        
        # Get TF-IDF scores for node sizing
        tfidf_scores = nx.get_node_attributes(graph, 'tf_idf_score')
        if not tfidf_scores:
            tfidf_scores = {node: 1.0 for node in graph.nodes()}
        
        # Community colors (limit to top communities)
        unique_communities = sorted(set(communities.values()))
        n_communities = min(len(unique_communities), self.config['max_legend_communities'])
        
        # Use scientific color palette
        colors = plt.cm.tab10(np.linspace(0, 1, n_communities))
        community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities[:n_communities])}
        
        # Assign "Other" color for remaining communities
        other_color = 'lightgray'
        for comm in unique_communities[n_communities:]:
            community_colors[comm] = other_color
        
        # Process each node
        for node in graph.nodes():
            # Color by community
            comm_id = communities.get(node, 0)
            color = community_colors.get(comm_id, other_color)
            node_colors.append(color)
            
            # Size by log(TF-IDF + 1)
            tfidf = tfidf_scores.get(node, 1.0)
            size = 100 + 400 * math.log(tfidf + 1, self.config['node_size_log_base']) / math.log(100, self.config['node_size_log_base'])
            size = max(50, min(800, size))  # Clamp size
            node_sizes.append(size)
            
            # Separate by role for different shapes
            role = node_roles.get(node, 'periphery')
            if role == 'core':
                node_shapes_core.append(node)
            else:
                node_shapes_periphery.append(node)
        
        # Draw edges with scientific styling
        if graph.number_of_edges() > 0:
            nx.draw_networkx_edges(
                graph, positions,
                width=0.5,
                alpha=self.config['edge_alpha'],
                edge_color='lightgray',
                ax=ax
            )
        
        # Draw nodes by role with different shapes
        if node_shapes_core:
            core_colors = [community_colors.get(communities.get(node, 0), other_color) for node in node_shapes_core]
            core_sizes = [node_sizes[list(graph.nodes()).index(node)] for node in node_shapes_core]
            nx.draw_networkx_nodes(
                graph, positions,
                nodelist=node_shapes_core,
                node_color=core_colors,
                node_size=core_sizes,
                node_shape='^',  # Triangles for core
                alpha=0.9,
                edgecolors='black',
                linewidths=1.0,
                ax=ax
            )
        
        if node_shapes_periphery:
            periphery_colors = [community_colors.get(communities.get(node, 0), other_color) for node in node_shapes_periphery]
            periphery_sizes = [node_sizes[list(graph.nodes()).index(node)] for node in node_shapes_periphery]
            nx.draw_networkx_nodes(
                graph, positions,
                nodelist=node_shapes_periphery,
                node_color=periphery_colors,
                node_size=periphery_sizes,
                node_shape='o',  # Circles for periphery
                alpha=0.8,
                edgecolors='gray',
                linewidths=0.5,
                ax=ax
            )
        
        # Add selective labels for high-centrality nodes
        centrality = nx.degree_centrality(graph)
        if centrality:
            centrality_threshold = np.percentile(list(centrality.values()), 
                                               self.config['label_centrality_threshold'] * 100)
            
            labels_to_draw = {}
            for node in graph.nodes():
                if centrality.get(node, 0) >= centrality_threshold:
                    # Truncate long labels
                    label = str(node)[:12] + "..." if len(str(node)) > 12 else str(node)
                    labels_to_draw[node] = label
            
            if labels_to_draw:
                nx.draw_networkx_labels(
                    graph, positions,
                    labels_to_draw,
                    font_size=8,
                    font_weight='bold',
                    font_color='black',
                    ax=ax
                )
        
        # Scientific title and statistics
        n_core = len(node_shapes_core)
        n_periphery = len(node_shapes_periphery)
        density = nx.density(graph)
        
        ax.set_title(
            f'{title}\n'
            f'N={graph.number_of_nodes()}, E={graph.number_of_edges()}, '
            f'Density={density:.4f}, Communities={len(set(communities.values()))}\n'
            f'Core={n_core}, Periphery={n_periphery}, Seed={self.random_seed}',
            fontsize=14, fontweight='bold', pad=20
        )
        
        # Scientific legend (limited to top communities)
        legend_elements = []
        
        # Community legend
        community_sizes = Counter(communities.values())
        top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:self.config['max_legend_communities']]
        
        for comm_id, size in top_communities:
            color = community_colors.get(comm_id, other_color)
            legend_elements.append(patches.Patch(color=color, label=f'Community {comm_id} (n={size})'))
        
        # Add "Other" if there are more communities
        if len(unique_communities) > self.config['max_legend_communities']:
            other_size = sum(size for comm_id, size in community_sizes.items() 
                           if comm_id not in [c for c, _ in top_communities])
            legend_elements.append(patches.Patch(color=other_color, label=f'Other (n={other_size})'))
        
        # Shape legend
        legend_elements.append(patches.Patch(color='white', label=''))  # Spacer
        legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                        markerfacecolor='gray', markersize=10, label='Core nodes'))
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor='gray', markersize=8, label='Periphery nodes'))
        
        # Method legend
        legend_elements.append(patches.Patch(color='white', label=''))  # Spacer
        legend_elements.append(patches.Patch(color='lightgray', label=f'Weighting: {self.config["semantic_weighting"].upper()}'))
        legend_elements.append(patches.Patch(color='lightgray', label=f'Sparsification: {self.config["sparsification_method"]}'))
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                frameon=True, fancybox=True, shadow=True)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Save with high resolution
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Saved: {output_path}")
        return output_path
    
    def compute_structural_statistics(self, graph):
        """
        Compute comprehensive structural statistics for scientific reporting
        
        Args:
            graph: NetworkX graph
            
        Returns:
            dict: structural statistics
        """
        print("📊 Computing structural statistics...")
        
        stats = {}
        
        if graph.number_of_nodes() == 0:
            return {'nodes': 0, 'edges': 0, 'density': 0, 'components': 0}
        
        # Basic structure
        stats['nodes'] = graph.number_of_nodes()
        stats['edges'] = graph.number_of_edges()
        stats['density'] = nx.density(graph)
        
        # Connectivity
        stats['components'] = nx.number_connected_components(graph)
        if stats['components'] > 0:
            largest_cc = max(nx.connected_components(graph), key=len)
            stats['largest_component_size'] = len(largest_cc)
            stats['largest_component_fraction'] = len(largest_cc) / stats['nodes']
        
        # Clustering and paths (only for connected graphs)
        if nx.is_connected(graph):
            stats['average_clustering'] = nx.average_clustering(graph)
            stats['transitivity'] = nx.transitivity(graph)
            stats['average_path_length'] = nx.average_shortest_path_length(graph)
            stats['diameter'] = nx.diameter(graph)
        else:
            # For disconnected graphs, compute on LCC
            if stats['components'] > 1:
                lcc = graph.subgraph(max(nx.connected_components(graph), key=len))
                if lcc.number_of_nodes() > 1:
                    stats['average_clustering'] = nx.average_clustering(lcc)
                    stats['transitivity'] = nx.transitivity(lcc)
                    if lcc.number_of_nodes() > 2:
                        stats['average_path_length'] = nx.average_shortest_path_length(lcc)
                        stats['diameter'] = nx.diameter(lcc)
        
        # Degree statistics
        degrees = [d for n, d in graph.degree()]
        if degrees:
            stats['average_degree'] = np.mean(degrees)
            stats['degree_std'] = np.std(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)
        
        # Centrality measures
        if graph.number_of_edges() > 0:
            degree_centrality = nx.degree_centrality(graph)
            stats['max_degree_centrality'] = max(degree_centrality.values())
            stats['centralization'] = (stats['max_degree_centrality'] - np.mean(list(degree_centrality.values()))) / (1 - 1/stats['nodes'])
        
        print(f"   ✅ Computed {len(stats)} structural measures")
        return stats
    
    def get_top_phrases_by_weighted_degree(self, graph, top_k=10):
        """
        Get top phrases by weighted degree for content analysis
        
        Args:
            graph: NetworkX graph
            top_k: number of top phrases to return
            
        Returns:
            list: [(phrase, weighted_degree, tfidf_score)]
        """
        weighted_degrees = dict(graph.degree(weight='weight'))
        tfidf_scores = nx.get_node_attributes(graph, 'tf_idf_score')
        
        phrase_stats = []
        for phrase, w_degree in weighted_degrees.items():
            tfidf = tfidf_scores.get(phrase, 0)
            phrase_stats.append((phrase, w_degree, tfidf))
        
        # Sort by weighted degree
        phrase_stats.sort(key=lambda x: x[1], reverse=True)
        
        return phrase_stats[:top_k]
    
    def optimize_graph(self, raw_graph, phrase_frequencies, total_phrases):
        """
        Main optimization pipeline
        
        Args:
            raw_graph: NetworkX graph with raw co-occurrence counts
            phrase_frequencies: dict of phrase frequencies
            total_phrases: total phrase instances
            
        Returns:
            tuple: (optimized_graph, communities, node_roles, positions, stats)
        """
        print("\n🔬 SCIENTIFIC GRAPH OPTIMIZATION PIPELINE")
        print("=" * 60)
        
        # Step 1: Compute semantic weights
        cooccurrence_matrix = {}
        for u, v, data in raw_graph.edges(data=True):
            weight = data.get('weight', data.get('raw_weight', 1))
            cooccurrence_matrix[(u, v)] = weight
        
        semantic_weights = self.compute_semantic_weights(cooccurrence_matrix, phrase_frequencies, total_phrases)
        
        # Step 2: Create semantically weighted graph
        semantic_graph = nx.Graph()
        semantic_graph.add_nodes_from(raw_graph.nodes(data=True))
        
        for (u, v), weight in semantic_weights.items():
            semantic_graph.add_edge(u, v, weight=weight, semantic_weight=weight)
        
        print(f"📊 Semantic graph: {semantic_graph.number_of_nodes()} nodes, {semantic_graph.number_of_edges()} edges")
        
        # Step 3: Apply sparsification
        if self.config['sparsification_method'] == 'quantile':
            sparsified_graph = self.apply_quantile_sparsification(semantic_graph, self.config['edge_retention_rate'])
        elif self.config['sparsification_method'] == 'disparity':
            sparsified_graph = self.apply_disparity_filter(semantic_graph, self.config['disparity_alpha'])
        elif self.config['sparsification_method'] == 'adaptive':
            # Apply both methods and choose the one with better connectivity
            quantile_graph = self.apply_quantile_sparsification(semantic_graph, self.config['edge_retention_rate'])
            disparity_graph = self.apply_disparity_filter(semantic_graph, self.config['disparity_alpha'])
            
            # Choose based on largest connected component size
            quantile_lcc_size = len(max(nx.connected_components(quantile_graph), key=len)) if quantile_graph.number_of_nodes() > 0 else 0
            disparity_lcc_size = len(max(nx.connected_components(disparity_graph), key=len)) if disparity_graph.number_of_nodes() > 0 else 0
            
            if quantile_lcc_size >= disparity_lcc_size:
                sparsified_graph = quantile_graph
                print("   🎯 Selected quantile-based sparsification")
            else:
                sparsified_graph = disparity_graph
                print("   🎯 Selected disparity filter sparsification")
        
        # Step 4: Extract LCC
        lcc_graph = self.extract_largest_connected_component(sparsified_graph)
        
        # Step 5: Community detection and pruning
        communities = self.detect_and_prune_communities(lcc_graph)
        
        # Step 6: Core-periphery identification
        node_roles = self.identify_core_periphery(lcc_graph)
        
        # Step 7: Deterministic layout
        positions = self.compute_deterministic_layout(lcc_graph)
        
        # Step 8: Compute statistics
        stats = self.compute_structural_statistics(lcc_graph)
        
        # Store optimization metadata
        nx.set_node_attributes(lcc_graph, communities, 'community_id')
        nx.set_node_attributes(lcc_graph, node_roles, 'node_type')
        nx.set_node_attributes(lcc_graph, positions, 'layout_pos')
        
        print(f"\n✅ OPTIMIZATION COMPLETE")
        print(f"📊 Final graph: {lcc_graph.number_of_nodes()} nodes, {lcc_graph.number_of_edges()} edges")
        print(f"📊 Density: {stats['density']:.6f}")
        print(f"📊 Communities: {len(set(communities.values()))}")
        print(f"📊 Core nodes: {sum(1 for role in node_roles.values() if role == 'core')}")
        
        return lcc_graph, communities, node_roles, positions, stats


def integrate_scientific_optimizer():
    """
    Integration function to add scientific optimization to the existing pipeline
    """
    print("🔬 Scientific Graph Optimizer Module Loaded")
    print("Available methods:")
    print("  - NPMI/Salton semantic weighting")
    print("  - Adaptive graph sparsification")
    print("  - LCC extraction")
    print("  - Community pruning")
    print("  - K-Core decomposition")
    print("  - Deterministic layouts")
    print("  - Scientific reporting")


if __name__ == "__main__":
    integrate_scientific_optimizer()