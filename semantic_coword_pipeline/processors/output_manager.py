"""
输出管理器

实现结构化输出管理和多格式数据导出功能。
根据需求7.2, 7.3, 7.4提供规范化的输出管理。
"""

import os
import json
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import scipy.sparse
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from ..core.data_models import GlobalGraph, StateSubgraph, ProcessedDocument, TOCDocument
from ..core.logger import PipelineLogger
from ..core.error_handler import ErrorHandler, OutputError


class OutputManager:
    """
    输出管理器
    
    提供结构化的输出管理和多格式数据导出功能。
    根据需求7.2, 7.3, 7.4实现规范化的中间结果和最终结果输出。
    """
    
    def __init__(self, config: Dict[str, Any], logger: PipelineLogger):
        """
        初始化输出管理器
        
        Args:
            config: 配置字典
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.error_handler = ErrorHandler()
        
        # 输出配置
        self.output_config = config.get('output', {})
        self.base_path = self.output_config.get('base_path', 'output/')
        self.export_formats = self.output_config.get('export_formats', ['json', 'csv'])
        self.generate_visualizations = self.output_config.get('generate_visualizations', True)
        self.save_intermediate = self.output_config.get('save_intermediate_results', True)
        self.compression_enabled = self.output_config.get('compression_enabled', False)
        
        # 创建输出目录结构
        self.output_structure = {
            'data': 'data',
            'graphs': 'graphs',
            'visualizations': 'visualizations',
            'reports': 'reports',
            'intermediate': 'intermediate',
            'logs': 'logs'
        }
    
    def generate_all_outputs(self, global_graph: GlobalGraph, 
                           state_subgraphs: Dict[str, StateSubgraph],
                           processed_docs: List[ProcessedDocument],
                           output_dir: str,
                           error_report: Dict[str, Any],
                           process_history: List[Dict[str, Any]]) -> List[str]:
        """
        生成所有输出文件
        
        Args:
            global_graph: 全局图
            state_subgraphs: 州级子图字典
            processed_docs: 处理后的文档列表
            output_dir: 输出目录
            error_report: 错误报告
            process_history: 处理历史
            
        Returns:
            生成的输出文件路径列表
        """
        output_files = []
        
        try:
            # 创建输出目录结构
            self._create_output_directories(output_dir)
            
            # 生成数据文件
            data_files = self._generate_data_outputs(
                global_graph, state_subgraphs, processed_docs, output_dir
            )
            output_files.extend(data_files)
            
            # 生成图文件
            graph_files = self._generate_graph_outputs(
                global_graph, state_subgraphs, output_dir
            )
            output_files.extend(graph_files)
            
            # 生成可视化文件
            if self.generate_visualizations:
                viz_files = self._generate_visualization_outputs(
                    global_graph, state_subgraphs, output_dir
                )
                output_files.extend(viz_files)
            
            # 生成报告文件
            report_files = self._generate_report_outputs(
                global_graph, state_subgraphs, processed_docs, 
                error_report, process_history, output_dir
            )
            output_files.extend(report_files)
            
            # 生成中间结果文件
            if self.save_intermediate:
                intermediate_files = self._generate_intermediate_outputs(
                    processed_docs, output_dir
                )
                output_files.extend(intermediate_files)
            
            # 生成日志文件
            log_files = self._generate_log_outputs(
                error_report, process_history, output_dir
            )
            output_files.extend(log_files)
            
            # 生成输出清单
            manifest_file = self._generate_output_manifest(output_files, output_dir)
            output_files.append(manifest_file)
            
            self.logger.info(f"Generated {len(output_files)} output files")
            return output_files
            
        except Exception as e:
            self.error_handler.handle_output_error(e, "generate_all_outputs")
    
    def _create_output_directories(self, output_dir: str) -> None:
        """创建输出目录结构"""
        base_path = Path(output_dir)
        
        for dir_name in self.output_structure.values():
            dir_path = base_path / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.debug(f"Created output directory structure in {output_dir}")
    
    def _generate_data_outputs(self, global_graph: GlobalGraph, 
                              state_subgraphs: Dict[str, StateSubgraph],
                              processed_docs: List[ProcessedDocument],
                              output_dir: str) -> List[str]:
        """生成数据输出文件"""
        output_files = []
        data_dir = Path(output_dir) / self.output_structure['data']
        
        try:
            # 生成词表文件
            vocab_files = self._export_vocabulary(global_graph, data_dir)
            output_files.extend(vocab_files)
            
            # 生成短语统计文件
            phrase_stats_files = self._export_phrase_statistics(processed_docs, data_dir)
            output_files.extend(phrase_stats_files)
            
            # 生成共现矩阵文件
            cooccurrence_files = self._export_cooccurrence_matrix(global_graph, data_dir)
            output_files.extend(cooccurrence_files)
            
            # 生成边表文件
            edge_files = self._export_edge_lists(global_graph, state_subgraphs, data_dir)
            output_files.extend(edge_files)
            
            # 生成文档统计文件
            doc_stats_files = self._export_document_statistics(processed_docs, data_dir)
            output_files.extend(doc_stats_files)
            
            self.logger.info(f"Generated {len(output_files)} data output files")
            
        except Exception as e:
            self.error_handler.handle_output_error(e, "data_outputs", str(data_dir))
        
        return output_files
    
    def _export_vocabulary(self, global_graph: GlobalGraph, data_dir: Path) -> List[str]:
        """导出词表"""
        output_files = []
        
        # 准备词表数据
        vocab_data = []
        for phrase, node_id in global_graph.vocabulary.items():
            vocab_data.append({
                'phrase': phrase,
                'node_id': node_id,
                'phrase_length': len(phrase.split())
            })
        
        # 按节点ID排序
        vocab_data.sort(key=lambda x: x['node_id'])
        
        # 导出为不同格式
        if 'json' in self.export_formats:
            json_file = data_dir / 'vocabulary.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(vocab_data, f, indent=2, ensure_ascii=False)
            output_files.append(str(json_file))
        
        if 'csv' in self.export_formats:
            csv_file = data_dir / 'vocabulary.csv'
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['phrase', 'node_id', 'phrase_length'])
                writer.writeheader()
                writer.writerows(vocab_data)
            output_files.append(str(csv_file))
        
        return output_files
    
    def _export_phrase_statistics(self, processed_docs: List[ProcessedDocument], 
                                 data_dir: Path) -> List[str]:
        """导出短语统计"""
        output_files = []
        
        # 计算短语频率
        phrase_counts = {}
        document_counts = {}
        
        for doc in processed_docs:
            doc_phrases = set(doc.phrases)
            for phrase in doc.phrases:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            
            for phrase in doc_phrases:
                document_counts[phrase] = document_counts.get(phrase, 0) + 1
        
        # 计算TF-IDF
        total_docs = len(processed_docs)
        phrase_stats = []
        
        for phrase, count in phrase_counts.items():
            doc_freq = document_counts.get(phrase, 0)
            tf_idf = count * np.log(total_docs / (doc_freq + 1))  # 加1避免除零
            
            phrase_stats.append({
                'phrase': phrase,
                'frequency': count,
                'document_frequency': doc_freq,
                'tf_idf_score': tf_idf,
                'document_ratio': doc_freq / total_docs
            })
        
        # 按频率排序
        phrase_stats.sort(key=lambda x: x['frequency'], reverse=True)
        
        # 导出为不同格式
        if 'json' in self.export_formats:
            json_file = data_dir / 'phrase_statistics.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(phrase_stats, f, indent=2, ensure_ascii=False)
            output_files.append(str(json_file))
        
        if 'csv' in self.export_formats:
            csv_file = data_dir / 'phrase_statistics.csv'
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'phrase', 'frequency', 'document_frequency', 'tf_idf_score', 'document_ratio'
                ])
                writer.writeheader()
                writer.writerows(phrase_stats)
            output_files.append(str(csv_file))
        
        return output_files
    
    def _export_cooccurrence_matrix(self, global_graph: GlobalGraph, 
                                   data_dir: Path) -> List[str]:
        """导出共现矩阵"""
        output_files = []
        
        if global_graph.cooccurrence_matrix is not None:
            # 导出稀疏矩阵
            if 'npz' in self.export_formats or 'numpy' in self.export_formats:
                npz_file = data_dir / 'cooccurrence_matrix.npz'
                scipy.sparse.save_npz(npz_file, global_graph.cooccurrence_matrix)
                output_files.append(str(npz_file))
            
            # 导出为CSV（仅非零元素）
            if 'csv' in self.export_formats:
                csv_file = data_dir / 'cooccurrence_edges.csv'
                
                # 获取非零元素
                coo_matrix = global_graph.cooccurrence_matrix.tocoo()
                
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['source_node', 'target_node', 'weight'])
                    
                    for i, j, weight in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
                        if i <= j:  # 只保存上三角矩阵（无向图）
                            writer.writerow([i, j, weight])
                
                output_files.append(str(csv_file))
        
        return output_files
    
    def _export_edge_lists(self, global_graph: GlobalGraph, 
                          state_subgraphs: Dict[str, StateSubgraph],
                          data_dir: Path) -> List[str]:
        """导出边表"""
        output_files = []
        
        # 导出全局图边表
        if global_graph.easygraph_instance:
            global_edges = self._extract_edge_list(global_graph.easygraph_instance, global_graph)
            
            if 'json' in self.export_formats:
                json_file = data_dir / 'global_graph_edges.json'
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(global_edges, f, indent=2, ensure_ascii=False)
                output_files.append(str(json_file))
            
            if 'csv' in self.export_formats:
                csv_file = data_dir / 'global_graph_edges.csv'
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'source_phrase', 'target_phrase', 'source_id', 'target_id', 'weight'
                    ])
                    writer.writeheader()
                    writer.writerows(global_edges)
                output_files.append(str(csv_file))
        
        # 导出州级子图边表
        for state, subgraph in state_subgraphs.items():
            if subgraph.easygraph_instance:
                state_edges = self._extract_edge_list(
                    subgraph.easygraph_instance, 
                    subgraph.parent_global_graph
                )
                
                if 'json' in self.export_formats:
                    json_file = data_dir / f'state_{state}_edges.json'
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(state_edges, f, indent=2, ensure_ascii=False)
                    output_files.append(str(json_file))
                
                if 'csv' in self.export_formats:
                    csv_file = data_dir / f'state_{state}_edges.csv'
                    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=[
                            'source_phrase', 'target_phrase', 'source_id', 'target_id', 'weight'
                        ])
                        writer.writeheader()
                        writer.writerows(state_edges)
                    output_files.append(str(csv_file))
        
        return output_files
    
    def _extract_edge_list(self, graph, global_graph: GlobalGraph) -> List[Dict[str, Any]]:
        """从图中提取边表"""
        edges = []
        
        try:
            for edge in graph.edges(data=True):
                source_id, target_id, edge_data = edge
                weight = edge_data.get('weight', 1.0)
                
                source_phrase = global_graph.get_phrase(source_id)
                target_phrase = global_graph.get_phrase(target_id)
                
                edges.append({
                    'source_phrase': source_phrase,
                    'target_phrase': target_phrase,
                    'source_id': source_id,
                    'target_id': target_id,
                    'weight': weight
                })
        except Exception as e:
            self.logger.warning(f"Failed to extract edge list: {e}")
        
        return edges
    
    def _export_document_statistics(self, processed_docs: List[ProcessedDocument], 
                                   data_dir: Path) -> List[str]:
        """导出文档统计"""
        output_files = []
        
        # 计算文档统计
        doc_stats = []
        for doc in processed_docs:
            stats = {
                'segment_id': doc.original_doc.segment_id,
                'state': doc.original_doc.state,
                'language': doc.original_doc.language,
                'title': doc.original_doc.title,
                'level': doc.original_doc.level,
                'order': doc.original_doc.order,
                'original_text_length': len(doc.original_doc.text),
                'cleaned_text_length': len(doc.cleaned_text),
                'token_count': len(doc.tokens),
                'phrase_count': len(doc.phrases),
                'window_count': len(doc.windows),
                'processing_metadata': doc.processing_metadata
            }
            doc_stats.append(stats)
        
        # 导出为不同格式
        if 'json' in self.export_formats:
            json_file = data_dir / 'document_statistics.json'
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(doc_stats, f, indent=2, ensure_ascii=False)
            output_files.append(str(json_file))
        
        if 'csv' in self.export_formats:
            csv_file = data_dir / 'document_statistics.csv'
            
            # 展平处理元数据
            flattened_stats = []
            for stats in doc_stats:
                flat_stats = stats.copy()
                metadata = flat_stats.pop('processing_metadata', {})
                flat_stats.update({f'metadata_{k}': v for k, v in metadata.items()})
                flattened_stats.append(flat_stats)
            
            if flattened_stats:
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=flattened_stats[0].keys())
                    writer.writeheader()
                    writer.writerows(flattened_stats)
                output_files.append(str(csv_file))
        
        return output_files
    
    def _generate_graph_outputs(self, global_graph: GlobalGraph, 
                               state_subgraphs: Dict[str, StateSubgraph],
                               output_dir: str) -> List[str]:
        """生成图输出文件"""
        output_files = []
        graphs_dir = Path(output_dir) / self.output_structure['graphs']
        
        try:
            # 导出全局图
            if global_graph.easygraph_instance:
                global_files = self._export_graph(
                    global_graph.easygraph_instance, 
                    'global_graph', 
                    graphs_dir,
                    global_graph
                )
                output_files.extend(global_files)
            
            # 导出州级子图
            for state, subgraph in state_subgraphs.items():
                if subgraph.easygraph_instance:
                    state_files = self._export_graph(
                        subgraph.easygraph_instance, 
                        f'state_{state}', 
                        graphs_dir,
                        subgraph.parent_global_graph,
                        subgraph.node_positions
                    )
                    output_files.extend(state_files)
            
            self.logger.info(f"Generated {len(output_files)} graph output files")
            
        except Exception as e:
            self.error_handler.handle_output_error(e, "graph_outputs", str(graphs_dir))
        
        return output_files
    
    def _export_graph(self, graph, graph_name: str, graphs_dir: Path,
                     global_graph: GlobalGraph, 
                     positions: Optional[Dict[int, tuple]] = None) -> List[str]:
        """导出单个图"""
        output_files = []
        
        try:
            # 导出为GraphML格式
            if 'graphml' in self.export_formats:
                graphml_file = graphs_dir / f'{graph_name}.graphml'
                # 这里需要实现GraphML导出逻辑
                # graph.write_graphml(str(graphml_file))
                # output_files.append(str(graphml_file))
            
            # 导出为GML格式
            if 'gml' in self.export_formats:
                gml_file = graphs_dir / f'{graph_name}.gml'
                # 这里需要实现GML导出逻辑
                # graph.write_gml(str(gml_file))
                # output_files.append(str(gml_file))
            
            # 导出为Pickle格式（包含所有信息）
            if 'pickle' in self.export_formats:
                pickle_file = graphs_dir / f'{graph_name}.pkl'
                graph_data = {
                    'graph': graph,
                    'vocabulary': global_graph.vocabulary,
                    'reverse_vocabulary': global_graph.reverse_vocabulary,
                    'positions': positions,
                    'metadata': {
                        'graph_name': graph_name,
                        'node_count': graph.number_of_nodes() if hasattr(graph, 'number_of_nodes') else 0,
                        'edge_count': graph.number_of_edges() if hasattr(graph, 'number_of_edges') else 0,
                        'exported_at': datetime.now().isoformat()
                    }
                }
                
                with open(pickle_file, 'wb') as f:
                    pickle.dump(graph_data, f)
                output_files.append(str(pickle_file))
            
        except Exception as e:
            self.logger.warning(f"Failed to export graph {graph_name}: {e}")
        
        return output_files
    
    def _generate_visualization_outputs(self, global_graph: GlobalGraph, 
                                       state_subgraphs: Dict[str, StateSubgraph],
                                       output_dir: str) -> List[str]:
        """生成可视化输出文件"""
        output_files = []
        viz_dir = Path(output_dir) / self.output_structure['visualizations']
        
        try:
            # 生成全局图可视化
            if global_graph.easygraph_instance:
                global_viz_files = self._create_graph_visualization(
                    global_graph.easygraph_instance,
                    'global_graph',
                    viz_dir,
                    global_graph
                )
                output_files.extend(global_viz_files)
            
            # 生成州级子图可视化
            for state, subgraph in state_subgraphs.items():
                if subgraph.easygraph_instance:
                    state_viz_files = self._create_graph_visualization(
                        subgraph.easygraph_instance,
                        f'state_{state}',
                        viz_dir,
                        subgraph.parent_global_graph,
                        subgraph.node_positions
                    )
                    output_files.extend(state_viz_files)
            
            # 生成对比可视化
            comparison_files = self._create_comparison_visualizations(
                global_graph, state_subgraphs, viz_dir
            )
            output_files.extend(comparison_files)
            
            self.logger.info(f"Generated {len(output_files)} visualization files")
            
        except Exception as e:
            self.error_handler.handle_output_error(e, "visualization_outputs", str(viz_dir))
        
        return output_files
    
    def _create_graph_visualization(self, graph, graph_name: str, viz_dir: Path,
                                   global_graph: GlobalGraph,
                                   positions: Optional[Dict[int, tuple]] = None) -> List[str]:
        """创建图可视化"""
        output_files = []
        
        try:
            # 创建matplotlib图形
            plt.figure(figsize=(12, 8))
            
            # 如果没有位置信息，使用简单布局
            if positions is None:
                positions = {}
                nodes = list(graph.nodes()) if hasattr(graph, 'nodes') else []
                for i, node in enumerate(nodes):
                    angle = 2 * np.pi * i / len(nodes)
                    positions[node] = (np.cos(angle), np.sin(angle))
            
            # 绘制节点
            if positions:
                x_coords = [pos[0] for pos in positions.values()]
                y_coords = [pos[1] for pos in positions.values()]
                plt.scatter(x_coords, y_coords, alpha=0.6, s=50)
            
            # 绘制边（简化版本）
            if hasattr(graph, 'edges') and positions:
                for edge in graph.edges():
                    if len(edge) >= 2:
                        source, target = edge[0], edge[1]
                        if source in positions and target in positions:
                            x1, y1 = positions[source]
                            x2, y2 = positions[target]
                            plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)
            
            plt.title(f'Graph Visualization: {graph_name}')
            plt.axis('equal')
            plt.axis('off')
            
            # 保存图像
            png_file = viz_dir / f'{graph_name}.png'
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_files.append(str(png_file))
            
        except Exception as e:
            self.logger.warning(f"Failed to create visualization for {graph_name}: {e}")
        
        return output_files
    
    def _create_comparison_visualizations(self, global_graph: GlobalGraph, 
                                         state_subgraphs: Dict[str, StateSubgraph],
                                         viz_dir: Path) -> List[str]:
        """创建对比可视化"""
        output_files = []
        
        try:
            # 创建网络统计对比图
            stats_file = self._create_network_statistics_plot(
                global_graph, state_subgraphs, viz_dir
            )
            if stats_file:
                output_files.append(stats_file)
            
            # 创建节点度分布对比图
            degree_file = self._create_degree_distribution_plot(
                global_graph, state_subgraphs, viz_dir
            )
            if degree_file:
                output_files.append(degree_file)
            
        except Exception as e:
            self.logger.warning(f"Failed to create comparison visualizations: {e}")
        
        return output_files
    
    def _create_network_statistics_plot(self, global_graph: GlobalGraph, 
                                       state_subgraphs: Dict[str, StateSubgraph],
                                       viz_dir: Path) -> Optional[str]:
        """创建网络统计对比图"""
        try:
            # 收集统计数据
            stats_data = []
            
            # 全局图统计
            if global_graph.easygraph_instance:
                global_stats = self._calculate_graph_statistics(global_graph.easygraph_instance)
                global_stats['name'] = 'Global Graph'
                stats_data.append(global_stats)
            
            # 州级子图统计
            for state, subgraph in state_subgraphs.items():
                if subgraph.easygraph_instance:
                    state_stats = self._calculate_graph_statistics(subgraph.easygraph_instance)
                    state_stats['name'] = f'State: {state}'
                    stats_data.append(state_stats)
            
            if not stats_data:
                return None
            
            # 创建对比图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            names = [data['name'] for data in stats_data]
            node_counts = [data['node_count'] for data in stats_data]
            edge_counts = [data['edge_count'] for data in stats_data]
            densities = [data['density'] for data in stats_data]
            isolated_ratios = [data['isolated_node_ratio'] for data in stats_data]
            
            # 节点数量对比
            axes[0, 0].bar(names, node_counts)
            axes[0, 0].set_title('Node Count Comparison')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 边数量对比
            axes[0, 1].bar(names, edge_counts)
            axes[0, 1].set_title('Edge Count Comparison')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 密度对比
            axes[1, 0].bar(names, densities)
            axes[1, 0].set_title('Graph Density Comparison')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 孤立节点比例对比
            axes[1, 1].bar(names, isolated_ratios)
            axes[1, 1].set_title('Isolated Node Ratio Comparison')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图像
            stats_file = viz_dir / 'network_statistics_comparison.png'
            plt.savefig(stats_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(stats_file)
            
        except Exception as e:
            self.logger.warning(f"Failed to create network statistics plot: {e}")
            return None
    
    def _create_degree_distribution_plot(self, global_graph: GlobalGraph, 
                                        state_subgraphs: Dict[str, StateSubgraph],
                                        viz_dir: Path) -> Optional[str]:
        """创建度分布对比图"""
        try:
            plt.figure(figsize=(12, 8))
            
            # 全局图度分布
            if global_graph.easygraph_instance:
                global_degrees = self._get_degree_sequence(global_graph.easygraph_instance)
                if global_degrees:
                    plt.hist(global_degrees, bins=20, alpha=0.7, label='Global Graph', density=True)
            
            # 州级子图度分布（选择前几个）
            colors = ['red', 'green', 'blue', 'orange', 'purple']
            for i, (state, subgraph) in enumerate(list(state_subgraphs.items())[:5]):
                if subgraph.easygraph_instance:
                    state_degrees = self._get_degree_sequence(subgraph.easygraph_instance)
                    if state_degrees:
                        plt.hist(state_degrees, bins=20, alpha=0.5, 
                               label=f'State: {state}', color=colors[i % len(colors)], density=True)
            
            plt.xlabel('Node Degree')
            plt.ylabel('Density')
            plt.title('Degree Distribution Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图像
            degree_file = viz_dir / 'degree_distribution_comparison.png'
            plt.savefig(degree_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(degree_file)
            
        except Exception as e:
            self.logger.warning(f"Failed to create degree distribution plot: {e}")
            return None
    
    def _calculate_graph_statistics(self, graph) -> Dict[str, float]:
        """计算图统计指标"""
        stats = {
            'node_count': 0,
            'edge_count': 0,
            'density': 0.0,
            'isolated_node_ratio': 0.0
        }
        
        try:
            if hasattr(graph, 'number_of_nodes'):
                stats['node_count'] = graph.number_of_nodes()
            
            if hasattr(graph, 'number_of_edges'):
                stats['edge_count'] = graph.number_of_edges()
            
            # 计算密度
            n = stats['node_count']
            if n > 1:
                max_edges = n * (n - 1) / 2  # 无向图
                stats['density'] = stats['edge_count'] / max_edges if max_edges > 0 else 0
            
            # 计算孤立节点比例
            if hasattr(graph, 'degree') and n > 0:
                isolated_count = sum(1 for node, degree in graph.degree() if degree == 0)
                stats['isolated_node_ratio'] = isolated_count / n
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate graph statistics: {e}")
        
        return stats
    
    def _get_degree_sequence(self, graph) -> List[int]:
        """获取度序列"""
        try:
            if hasattr(graph, 'degree'):
                return [degree for node, degree in graph.degree()]
        except Exception as e:
            self.logger.warning(f"Failed to get degree sequence: {e}")
        
        return []
    
    def _generate_report_outputs(self, global_graph: GlobalGraph, 
                                state_subgraphs: Dict[str, StateSubgraph],
                                processed_docs: List[ProcessedDocument],
                                error_report: Dict[str, Any],
                                process_history: List[Dict[str, Any]],
                                output_dir: str) -> List[str]:
        """生成报告输出文件"""
        output_files = []
        reports_dir = Path(output_dir) / self.output_structure['reports']
        
        try:
            # 生成处理摘要报告
            summary_file = self._generate_processing_summary(
                global_graph, state_subgraphs, processed_docs, reports_dir
            )
            output_files.append(summary_file)
            
            # 生成网络分析报告
            analysis_file = self._generate_network_analysis_report(
                global_graph, state_subgraphs, reports_dir
            )
            output_files.append(analysis_file)
            
            # 生成质量报告
            quality_file = self._generate_quality_report(
                error_report, process_history, reports_dir
            )
            output_files.append(quality_file)
            
            self.logger.info(f"Generated {len(output_files)} report files")
            
        except Exception as e:
            self.error_handler.handle_output_error(e, "report_outputs", str(reports_dir))
        
        return output_files
    
    def _generate_processing_summary(self, global_graph: GlobalGraph, 
                                    state_subgraphs: Dict[str, StateSubgraph],
                                    processed_docs: List[ProcessedDocument],
                                    reports_dir: Path) -> str:
        """生成处理摘要报告"""
        summary = {
            'processing_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_documents': len(processed_docs),
                'states_processed': len(state_subgraphs),
                'global_graph': {
                    'node_count': global_graph.get_node_count(),
                    'vocabulary_size': len(global_graph.vocabulary)
                },
                'state_subgraphs': {}
            }
        }
        
        # 添加州级统计
        for state, subgraph in state_subgraphs.items():
            summary['processing_summary']['state_subgraphs'][state] = {
                'active_nodes': len(subgraph.get_active_nodes()),
                'statistics': subgraph.statistics
            }
        
        # 添加文档统计
        doc_stats = {
            'by_state': {},
            'by_language': {},
            'total_tokens': sum(len(doc.tokens) for doc in processed_docs),
            'total_phrases': sum(len(doc.phrases) for doc in processed_docs)
        }
        
        for doc in processed_docs:
            state = doc.original_doc.state or 'unknown'
            language = doc.original_doc.language or 'unknown'
            
            if state not in doc_stats['by_state']:
                doc_stats['by_state'][state] = 0
            doc_stats['by_state'][state] += 1
            
            if language not in doc_stats['by_language']:
                doc_stats['by_language'][language] = 0
            doc_stats['by_language'][language] += 1
        
        summary['document_statistics'] = doc_stats
        
        # 保存报告
        summary_file = reports_dir / 'processing_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return str(summary_file)
    
    def _generate_network_analysis_report(self, global_graph: GlobalGraph, 
                                         state_subgraphs: Dict[str, StateSubgraph],
                                         reports_dir: Path) -> str:
        """生成网络分析报告"""
        analysis = {
            'network_analysis': {
                'timestamp': datetime.now().isoformat(),
                'global_graph_analysis': {},
                'state_subgraph_analysis': {},
                'comparative_analysis': {}
            }
        }
        
        # 全局图分析
        if global_graph.easygraph_instance:
            analysis['network_analysis']['global_graph_analysis'] = \
                self._calculate_graph_statistics(global_graph.easygraph_instance)
        
        # 州级子图分析
        for state, subgraph in state_subgraphs.items():
            if subgraph.easygraph_instance:
                analysis['network_analysis']['state_subgraph_analysis'][state] = \
                    self._calculate_graph_statistics(subgraph.easygraph_instance)
        
        # 对比分析
        if state_subgraphs:
            node_counts = [len(sg.get_active_nodes()) for sg in state_subgraphs.values()]
            analysis['network_analysis']['comparative_analysis'] = {
                'max_state_nodes': max(node_counts) if node_counts else 0,
                'min_state_nodes': min(node_counts) if node_counts else 0,
                'avg_state_nodes': sum(node_counts) / len(node_counts) if node_counts else 0,
                'state_count': len(state_subgraphs)
            }
        
        # 保存报告
        analysis_file = reports_dir / 'network_analysis.json'
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        return str(analysis_file)
    
    def _generate_quality_report(self, error_report: Dict[str, Any],
                                process_history: List[Dict[str, Any]],
                                reports_dir: Path) -> str:
        """生成质量报告"""
        quality_report = {
            'quality_report': {
                'timestamp': datetime.now().isoformat(),
                'error_summary': error_report.get('summary', {}),
                'process_summary': {
                    'total_processes': len(process_history),
                    'successful_processes': sum(1 for p in process_history if p.get('status') == 'completed'),
                    'failed_processes': sum(1 for p in process_history if p.get('status') == 'failed')
                },
                'recommendations': []
            }
        }
        
        # 添加建议
        error_count = error_report.get('summary', {}).get('total_errors', 0)
        if error_count > 0:
            quality_report['quality_report']['recommendations'].append(
                f"Review {error_count} errors in the error log for potential improvements"
            )
        
        if quality_report['quality_report']['process_summary']['failed_processes'] > 0:
            quality_report['quality_report']['recommendations'].append(
                "Some processes failed - check process history for details"
            )
        
        # 保存报告
        quality_file = reports_dir / 'quality_report.json'
        with open(quality_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False)
        
        return str(quality_file)
    
    def _generate_intermediate_outputs(self, processed_docs: List[ProcessedDocument],
                                      output_dir: str) -> List[str]:
        """生成中间结果输出文件"""
        output_files = []
        intermediate_dir = Path(output_dir) / self.output_structure['intermediate']
        
        try:
            # 保存处理后的文档
            docs_file = intermediate_dir / 'processed_documents.json'
            
            # 序列化处理后的文档
            serialized_docs = []
            for doc in processed_docs:
                doc_dict = {
                    'original_doc': doc.original_doc.to_dict(),
                    'cleaned_text': doc.cleaned_text,
                    'tokens': doc.tokens,
                    'phrases': doc.phrases,
                    'windows': [
                        {
                            'window_id': w.window_id,
                            'phrases': w.phrases,
                            'source_doc': w.source_doc,
                            'state': w.state,
                            'segment_id': w.segment_id
                        }
                        for w in doc.windows
                    ],
                    'processing_metadata': doc.processing_metadata
                }
                serialized_docs.append(doc_dict)
            
            with open(docs_file, 'w', encoding='utf-8') as f:
                json.dump(serialized_docs, f, indent=2, ensure_ascii=False)
            
            output_files.append(str(docs_file))
            
            self.logger.info(f"Generated {len(output_files)} intermediate output files")
            
        except Exception as e:
            self.error_handler.handle_output_error(e, "intermediate_outputs", str(intermediate_dir))
        
        return output_files
    
    def _generate_log_outputs(self, error_report: Dict[str, Any],
                             process_history: List[Dict[str, Any]],
                             output_dir: str) -> List[str]:
        """生成日志输出文件"""
        output_files = []
        logs_dir = Path(output_dir) / self.output_structure['logs']
        
        try:
            # 保存错误报告
            error_file = logs_dir / 'error_report.json'
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(error_report, f, indent=2, ensure_ascii=False)
            output_files.append(str(error_file))
            
            # 保存处理历史
            history_file = logs_dir / 'process_history.json'
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(process_history, f, indent=2, ensure_ascii=False)
            output_files.append(str(history_file))
            
            self.logger.info(f"Generated {len(output_files)} log output files")
            
        except Exception as e:
            self.error_handler.handle_output_error(e, "log_outputs", str(logs_dir))
        
        return output_files
    
    def _generate_output_manifest(self, output_files: List[str], output_dir: str) -> str:
        """生成输出清单"""
        manifest = {
            'output_manifest': {
                'timestamp': datetime.now().isoformat(),
                'output_directory': output_dir,
                'total_files': len(output_files),
                'files_by_category': {},
                'file_list': output_files
            }
        }
        
        # 按类别分组文件
        for file_path in output_files:
            path_obj = Path(file_path)
            category = path_obj.parent.name
            
            if category not in manifest['output_manifest']['files_by_category']:
                manifest['output_manifest']['files_by_category'][category] = []
            
            manifest['output_manifest']['files_by_category'][category].append(file_path)
        
        # 保存清单
        manifest_file = Path(output_dir) / 'output_manifest.json'
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        return str(manifest_file)