"""
网络分析器（NetworkAnalyzer）

根据需求8.1-8.5实现多维度网络结构对比分析功能。
主要功能包括：
- 创建基础网络统计计算
- 实现高级网络指标分析
- 添加多维度对比分析功能
- 建立社群划分和中心性分析
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Easy-Graph'))

from typing import Dict, List, Set, Tuple, Any, Optional, Union
import numpy as np
import scipy.sparse
from collections import defaultdict, Counter
import logging
from datetime import datetime
import json

try:
    import easygraph as eg
except ImportError:
    print("Warning: EasyGraph not available. Some functionality may be limited.")
    eg = None

from ..core.data_models import GlobalGraph, StateSubgraph
from ..core.logger import setup_logger


class NetworkAnalyzer:
    """
    网络分析器
    
    根据需求8实现多维度网络结构对比分析，包括：
    - 基础网络统计（需求8.4）
    - 高级网络指标（需求8.5）
    - 多维度对比分析（需求8.1, 8.2, 8.3）
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化网络分析器
        
        Args:
            config: 配置参数字典
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # 配置参数
        self.enable_community_detection = config.get('enable_community_detection', True)
        self.centrality_metrics = config.get('centrality_metrics', ['degree', 'betweenness', 'closeness', 'pagerank'])
        self.community_algorithm = config.get('community_algorithm', 'louvain')
        self.comparison_metrics = config.get('comparison_metrics', [
            'node_count', 'edge_count', 'density', 'isolated_nodes_ratio',
            'connected_components', 'clustering_coefficient', 'average_path_length'
        ])
        
        self.logger.info(f"NetworkAnalyzer initialized with config: {config}")
    
    def calculate_basic_statistics(self, graph: Union[GlobalGraph, StateSubgraph]) -> Dict[str, float]:
        """
        计算基础网络统计指标
        
        根据需求8.4计算节点/边规模、密度、孤立点比例等基础指标。
        
        Args:
            graph: 全局图或州级子图对象
            
        Returns:
            Dict[str, float]: 基础统计指标字典
        """
        self.logger.debug("Calculating basic network statistics")
        
        if graph.easygraph_instance is None:
            raise ValueError("EasyGraph instance is required for network analysis")
        
        eg_graph = graph.easygraph_instance
        stats = {}
        
        # 基础规模指标
        stats['node_count'] = float(eg_graph.number_of_nodes())
        stats['edge_count'] = float(eg_graph.number_of_edges())
        
        # 密度计算
        n = stats['node_count']
        if n > 1:
            max_edges = n * (n - 1) / 2  # 无向图的最大边数
            stats['density'] = stats['edge_count'] / max_edges if max_edges > 0 else 0.0
        else:
            stats['density'] = 0.0
        
        # 孤立节点比例
        isolated_nodes = [node for node in eg_graph.nodes() if eg_graph.degree(node) == 0]
        stats['isolated_nodes_count'] = float(len(isolated_nodes))
        stats['isolated_nodes_ratio'] = stats['isolated_nodes_count'] / stats['node_count'] if stats['node_count'] > 0 else 0.0
        
        # 度分布统计
        degrees = [eg_graph.degree(node) for node in eg_graph.nodes()]
        if degrees:
            stats['average_degree'] = float(np.mean(degrees))
            stats['max_degree'] = float(np.max(degrees))
            stats['min_degree'] = float(np.min(degrees))
            stats['degree_std'] = float(np.std(degrees))
        else:
            stats['average_degree'] = 0.0
            stats['max_degree'] = 0.0
            stats['min_degree'] = 0.0
            stats['degree_std'] = 0.0
        
        self.logger.debug(f"Basic statistics calculated: {len(stats)} metrics")
        return stats
    
    def calculate_advanced_metrics(self, graph: Union[GlobalGraph, StateSubgraph]) -> Dict[str, Any]:
        """
        计算高级网络指标
        
        根据需求8.5分析连通分量、聚类系数、平均路径长度、中心性排序稳定性。
        
        Args:
            graph: 全局图或州级子图对象
            
        Returns:
            Dict[str, Any]: 高级网络指标字典
        """
        self.logger.debug("Calculating advanced network metrics")
        
        if graph.easygraph_instance is None:
            raise ValueError("EasyGraph instance is required for advanced analysis")
        
        eg_graph = graph.easygraph_instance
        metrics = {}
        
        # 连通分量分析
        try:
            if eg.is_connected(eg_graph):
                metrics['connected_components_count'] = 1
                metrics['largest_component_size'] = float(eg_graph.number_of_nodes())
                metrics['largest_component_ratio'] = 1.0
            else:
                components = list(eg.connected_components(eg_graph))
                metrics['connected_components_count'] = len(components)
                largest_component_size = max(len(comp) for comp in components) if components else 0
                metrics['largest_component_size'] = float(largest_component_size)
                metrics['largest_component_ratio'] = largest_component_size / eg_graph.number_of_nodes() if eg_graph.number_of_nodes() > 0 else 0.0
        except Exception as e:
            self.logger.warning(f"Error calculating connected components: {e}")
            metrics['connected_components_count'] = 0
            metrics['largest_component_size'] = 0.0
            metrics['largest_component_ratio'] = 0.0
        
        # 聚类系数
        try:
            if eg_graph.number_of_nodes() > 0:
                clustering_coeffs = eg.clustering(eg_graph)
                if clustering_coeffs:
                    metrics['average_clustering_coefficient'] = float(np.mean(list(clustering_coeffs.values())))
                    metrics['global_clustering_coefficient'] = float(eg.transitivity(eg_graph))
                else:
                    metrics['average_clustering_coefficient'] = 0.0
                    metrics['global_clustering_coefficient'] = 0.0
            else:
                metrics['average_clustering_coefficient'] = 0.0
                metrics['global_clustering_coefficient'] = 0.0
        except Exception as e:
            self.logger.warning(f"Error calculating clustering coefficient: {e}")
            metrics['average_clustering_coefficient'] = 0.0
            metrics['global_clustering_coefficient'] = 0.0
        
        # 平均路径长度（仅对连通图计算）
        try:
            if eg.is_connected(eg_graph) and eg_graph.number_of_nodes() > 1:
                metrics['average_path_length'] = float(eg.average_shortest_path_length(eg_graph))
                metrics['diameter'] = float(eg.diameter(eg_graph))
            else:
                metrics['average_path_length'] = float('inf')
                metrics['diameter'] = float('inf')
        except Exception as e:
            self.logger.warning(f"Error calculating path metrics: {e}")
            metrics['average_path_length'] = float('inf')
            metrics['diameter'] = float('inf')
        
        # 中心性指标
        metrics['centrality_metrics'] = self._calculate_centrality_metrics(eg_graph)
        
        self.logger.debug(f"Advanced metrics calculated: {len(metrics)} metrics")
        return metrics
    
    def _calculate_centrality_metrics(self, eg_graph) -> Dict[str, Dict[int, float]]:
        """
        计算各种中心性指标
        
        Args:
            eg_graph: EasyGraph图实例
            
        Returns:
            Dict[str, Dict[int, float]]: 中心性指标字典
        """
        centrality_results = {}
        
        if eg_graph.number_of_nodes() == 0:
            return centrality_results
        
        # 度中心性
        if 'degree' in self.centrality_metrics:
            try:
                degree_centrality = eg.degree_centrality(eg_graph)
                centrality_results['degree_centrality'] = degree_centrality
            except Exception as e:
                self.logger.warning(f"Error calculating degree centrality: {e}")
                centrality_results['degree_centrality'] = {}
        
        # 介数中心性
        if 'betweenness' in self.centrality_metrics:
            try:
                betweenness_centrality = eg.betweenness_centrality(eg_graph)
                centrality_results['betweenness_centrality'] = betweenness_centrality
            except Exception as e:
                self.logger.warning(f"Error calculating betweenness centrality: {e}")
                centrality_results['betweenness_centrality'] = {}
        
        # 接近中心性（仅对连通图计算）
        if 'closeness' in self.centrality_metrics:
            try:
                if eg.is_connected(eg_graph):
                    closeness_centrality = eg.closeness_centrality(eg_graph)
                    centrality_results['closeness_centrality'] = closeness_centrality
                else:
                    centrality_results['closeness_centrality'] = {}
            except Exception as e:
                self.logger.warning(f"Error calculating closeness centrality: {e}")
                centrality_results['closeness_centrality'] = {}
        
        # PageRank中心性
        if 'pagerank' in self.centrality_metrics:
            try:
                pagerank_centrality = eg.pagerank(eg_graph)
                centrality_results['pagerank_centrality'] = pagerank_centrality
            except Exception as e:
                self.logger.warning(f"Error calculating PageRank centrality: {e}")
                centrality_results['pagerank_centrality'] = {}
        
        return centrality_results
    
    def detect_communities(self, graph: Union[GlobalGraph, StateSubgraph]) -> Dict[str, Any]:
        """
        社群划分分析
        
        根据需求8.6评估社群划分可解释性。
        
        Args:
            graph: 全局图或州级子图对象
            
        Returns:
            Dict[str, Any]: 社群划分结果
        """
        self.logger.debug("Detecting communities")
        
        if not self.enable_community_detection:
            return {'communities': [], 'modularity': 0.0, 'community_count': 0}
        
        if graph.easygraph_instance is None:
            raise ValueError("EasyGraph instance is required for community detection")
        
        eg_graph = graph.easygraph_instance
        
        if eg_graph.number_of_nodes() < 2:
            return {'communities': [], 'modularity': 0.0, 'community_count': 0}
        
        try:
            # 使用Louvain算法进行社群检测
            if self.community_algorithm == 'louvain':
                communities = eg.louvain_communities(eg_graph)
            else:
                # 默认使用Louvain
                communities = eg.louvain_communities(eg_graph)
            
            # 计算模块度
            modularity = eg.modularity(eg_graph, communities)
            
            # 社群统计
            community_sizes = [len(community) for community in communities]
            
            result = {
                'communities': [list(community) for community in communities],
                'modularity': float(modularity),
                'community_count': len(communities),
                'community_sizes': community_sizes,
                'average_community_size': float(np.mean(community_sizes)) if community_sizes else 0.0,
                'largest_community_size': max(community_sizes) if community_sizes else 0,
                'smallest_community_size': min(community_sizes) if community_sizes else 0
            }
            
            self.logger.debug(f"Detected {len(communities)} communities with modularity {modularity:.3f}")
            return result
            
        except Exception as e:
            self.logger.warning(f"Error in community detection: {e}")
            return {'communities': [], 'modularity': 0.0, 'community_count': 0}
    
    def compare_network_structures(self, graphs: Dict[str, Union[GlobalGraph, StateSubgraph]], 
                                 comparison_name: str = "network_comparison") -> Dict[str, Any]:
        """
        多维度网络结构对比分析
        
        根据需求8.1、8.2、8.3对比不同设置对网络结构的影响。
        
        Args:
            graphs: 待对比的图字典，键为图名称，值为图对象
            comparison_name: 对比分析名称
            
        Returns:
            Dict[str, Any]: 对比分析结果
        """
        self.logger.info(f"Performing network structure comparison: {comparison_name}")
        
        if not graphs:
            raise ValueError("At least one graph is required for comparison")
        
        comparison_results = {
            'comparison_name': comparison_name,
            'timestamp': datetime.now().isoformat(),
            'graph_names': list(graphs.keys()),
            'basic_statistics': {},
            'advanced_metrics': {},
            'community_analysis': {},
            'summary': {}
        }
        
        # 计算每个图的统计指标
        for graph_name, graph in graphs.items():
            self.logger.debug(f"Analyzing graph: {graph_name}")
            
            # 基础统计
            basic_stats = self.calculate_basic_statistics(graph)
            comparison_results['basic_statistics'][graph_name] = basic_stats
            
            # 高级指标
            advanced_metrics = self.calculate_advanced_metrics(graph)
            comparison_results['advanced_metrics'][graph_name] = advanced_metrics
            
            # 社群分析
            community_results = self.detect_communities(graph)
            comparison_results['community_analysis'][graph_name] = community_results
        
        # 生成对比摘要
        comparison_results['summary'] = self._generate_comparison_summary(comparison_results)
        
        self.logger.info(f"Network comparison completed for {len(graphs)} graphs")
        return comparison_results
    
    def _generate_comparison_summary(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成对比分析摘要
        
        Args:
            comparison_results: 对比分析结果
            
        Returns:
            Dict[str, Any]: 对比摘要
        """
        summary = {}
        graph_names = comparison_results['graph_names']
        
        if len(graph_names) < 2:
            return summary
        
        # 基础统计对比
        basic_stats = comparison_results['basic_statistics']
        
        # 节点数对比
        node_counts = {name: stats['node_count'] for name, stats in basic_stats.items()}
        summary['node_count_comparison'] = {
            'values': node_counts,
            'max_graph': max(node_counts, key=node_counts.get),
            'min_graph': min(node_counts, key=node_counts.get),
            'ratio_max_to_min': node_counts[max(node_counts, key=node_counts.get)] / max(1, node_counts[min(node_counts, key=node_counts.get)])
        }
        
        # 边数对比
        edge_counts = {name: stats['edge_count'] for name, stats in basic_stats.items()}
        summary['edge_count_comparison'] = {
            'values': edge_counts,
            'max_graph': max(edge_counts, key=edge_counts.get),
            'min_graph': min(edge_counts, key=edge_counts.get),
            'ratio_max_to_min': edge_counts[max(edge_counts, key=edge_counts.get)] / max(1, edge_counts[min(edge_counts, key=edge_counts.get)])
        }
        
        # 密度对比
        densities = {name: stats['density'] for name, stats in basic_stats.items()}
        summary['density_comparison'] = {
            'values': densities,
            'max_graph': max(densities, key=densities.get),
            'min_graph': min(densities, key=densities.get)
        }
        
        # 孤立节点比例对比
        isolated_ratios = {name: stats['isolated_nodes_ratio'] for name, stats in basic_stats.items()}
        summary['isolated_nodes_ratio_comparison'] = {
            'values': isolated_ratios,
            'max_graph': max(isolated_ratios, key=isolated_ratios.get),
            'min_graph': min(isolated_ratios, key=isolated_ratios.get)
        }
        
        # 社群分析对比
        community_analysis = comparison_results['community_analysis']
        modularities = {name: analysis['modularity'] for name, analysis in community_analysis.items()}
        summary['modularity_comparison'] = {
            'values': modularities,
            'max_graph': max(modularities, key=modularities.get),
            'min_graph': min(modularities, key=modularities.get)
        }
        
        return summary
    
    def analyze_cross_state_differences(self, state_subgraphs: Dict[str, StateSubgraph]) -> Dict[str, Any]:
        """
        跨州关键主题短语差异分析
        
        根据需求8.6评估跨州关键主题短语差异。
        
        Args:
            state_subgraphs: 州级子图字典
            
        Returns:
            Dict[str, Any]: 跨州差异分析结果
        """
        self.logger.info(f"Analyzing cross-state differences for {len(state_subgraphs)} states")
        
        if len(state_subgraphs) < 2:
            return {'error': 'At least 2 state subgraphs required for cross-state analysis'}
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'states_analyzed': list(state_subgraphs.keys()),
            'state_statistics': {},
            'phrase_frequency_analysis': {},
            'centrality_ranking_stability': {},
            'unique_phrases_by_state': {},
            'common_phrases': set(),
            'state_specific_phrases': {}
        }
        
        # 分析每个州的统计信息
        all_phrases = set()
        state_phrases = {}
        
        for state_name, subgraph in state_subgraphs.items():
            # 基础统计
            basic_stats = self.calculate_basic_statistics(subgraph)
            analysis_results['state_statistics'][state_name] = basic_stats
            
            # 收集该州的所有短语
            if subgraph.parent_global_graph and subgraph.parent_global_graph.reverse_vocabulary:
                active_nodes = subgraph.get_active_nodes()
                state_phrase_set = set()
                for node_id in active_nodes:
                    phrase = subgraph.parent_global_graph.get_phrase(node_id)
                    if phrase:
                        state_phrase_set.add(phrase)
                        all_phrases.add(phrase)
                
                state_phrases[state_name] = state_phrase_set
                analysis_results['unique_phrases_by_state'][state_name] = len(state_phrase_set)
        
        # 找出共同短语和州特有短语
        if len(state_phrases) >= 2:
            # 共同短语（所有州都有的短语）
            common_phrases = set.intersection(*state_phrases.values()) if state_phrases else set()
            analysis_results['common_phrases'] = list(common_phrases)
            
            # 每个州特有的短语
            for state_name, phrases in state_phrases.items():
                other_states_phrases = set()
                for other_state, other_phrases in state_phrases.items():
                    if other_state != state_name:
                        other_states_phrases.update(other_phrases)
                
                unique_to_state = phrases - other_states_phrases
                analysis_results['state_specific_phrases'][state_name] = list(unique_to_state)
        
        # 中心性排序稳定性分析
        centrality_rankings = {}
        for state_name, subgraph in state_subgraphs.items():
            if subgraph.easygraph_instance and subgraph.easygraph_instance.number_of_nodes() > 0:
                try:
                    # 计算度中心性排序
                    degree_centrality = eg.degree_centrality(subgraph.easygraph_instance)
                    sorted_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
                    
                    # 转换为短语排序
                    phrase_ranking = []
                    for node_id, centrality_score in sorted_nodes[:20]:  # 取前20个
                        phrase = subgraph.parent_global_graph.get_phrase(node_id)
                        if phrase:
                            phrase_ranking.append((phrase, centrality_score))
                    
                    centrality_rankings[state_name] = phrase_ranking
                except Exception as e:
                    self.logger.warning(f"Error calculating centrality ranking for {state_name}: {e}")
        
        analysis_results['centrality_ranking_stability'] = centrality_rankings
        
        self.logger.info("Cross-state difference analysis completed")
        return analysis_results
    
    def generate_analysis_report(self, analysis_results: Dict[str, Any], 
                               output_path: Optional[str] = None) -> str:
        """
        生成分析报告
        
        Args:
            analysis_results: 分析结果字典
            output_path: 输出路径（可选）
            
        Returns:
            str: 报告内容
        """
        self.logger.info("Generating network analysis report")
        
        report_lines = []
        report_lines.append("# 网络分析报告")
        report_lines.append(f"生成时间: {datetime.now().isoformat()}")
        report_lines.append("")
        
        # 基础统计摘要
        if 'basic_statistics' in analysis_results:
            report_lines.append("## 基础网络统计")
            for graph_name, stats in analysis_results['basic_statistics'].items():
                report_lines.append(f"### {graph_name}")
                report_lines.append(f"- 节点数: {stats.get('node_count', 0):.0f}")
                report_lines.append(f"- 边数: {stats.get('edge_count', 0):.0f}")
                report_lines.append(f"- 密度: {stats.get('density', 0):.4f}")
                report_lines.append(f"- 孤立节点比例: {stats.get('isolated_nodes_ratio', 0):.4f}")
                report_lines.append(f"- 平均度: {stats.get('average_degree', 0):.2f}")
                report_lines.append("")
        
        # 高级指标摘要
        if 'advanced_metrics' in analysis_results:
            report_lines.append("## 高级网络指标")
            for graph_name, metrics in analysis_results['advanced_metrics'].items():
                report_lines.append(f"### {graph_name}")
                report_lines.append(f"- 连通分量数: {metrics.get('connected_components_count', 0)}")
                report_lines.append(f"- 最大连通分量比例: {metrics.get('largest_component_ratio', 0):.4f}")
                report_lines.append(f"- 平均聚类系数: {metrics.get('average_clustering_coefficient', 0):.4f}")
                report_lines.append(f"- 全局聚类系数: {metrics.get('global_clustering_coefficient', 0):.4f}")
                
                avg_path = metrics.get('average_path_length', float('inf'))
                if avg_path != float('inf'):
                    report_lines.append(f"- 平均路径长度: {avg_path:.2f}")
                else:
                    report_lines.append("- 平均路径长度: 无穷大（图不连通）")
                report_lines.append("")
        
        # 社群分析摘要
        if 'community_analysis' in analysis_results:
            report_lines.append("## 社群分析")
            for graph_name, community_info in analysis_results['community_analysis'].items():
                report_lines.append(f"### {graph_name}")
                report_lines.append(f"- 社群数量: {community_info.get('community_count', 0)}")
                report_lines.append(f"- 模块度: {community_info.get('modularity', 0):.4f}")
                report_lines.append(f"- 平均社群大小: {community_info.get('average_community_size', 0):.1f}")
                report_lines.append("")
        
        # 对比分析摘要
        if 'summary' in analysis_results:
            report_lines.append("## 对比分析摘要")
            summary = analysis_results['summary']
            
            if 'node_count_comparison' in summary:
                node_comp = summary['node_count_comparison']
                report_lines.append(f"- 节点数最多: {node_comp['max_graph']} ({node_comp['values'][node_comp['max_graph']]:.0f})")
                report_lines.append(f"- 节点数最少: {node_comp['min_graph']} ({node_comp['values'][node_comp['min_graph']]:.0f})")
                report_lines.append(f"- 节点数比例: {node_comp['ratio_max_to_min']:.2f}")
                report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                self.logger.info(f"Analysis report saved to: {output_path}")
            except Exception as e:
                self.logger.error(f"Error saving report to {output_path}: {e}")
        
        return report_content