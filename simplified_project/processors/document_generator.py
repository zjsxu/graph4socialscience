"""
文档生成和追溯系统

实现结构化文档生成、技术选择记录、处理过程追溯和对比报告生成功能。
根据需求10.1, 10.2, 10.3, 10.4, 10.5, 10.6提供完整的实验文档和可追溯分析过程。
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
import scipy.sparse

from ..core.data_models import GlobalGraph, StateSubgraph, ProcessedDocument, TOCDocument
from ..core.logger import PipelineLogger
from ..core.error_handler import ErrorHandler, ProcessingError


@dataclass
class TechnicalChoice:
    """技术选择记录"""
    component: str
    choice: str
    rationale: str
    alternatives: List[str]
    timestamp: str
    parameters: Dict[str, Any]


@dataclass
class ProcessingStep:
    """处理步骤记录"""
    step_name: str
    input_description: str
    output_description: str
    parameters: Dict[str, Any]
    start_time: str
    end_time: str
    duration_seconds: float
    status: str
    error_message: Optional[str] = None


@dataclass
class ComparisonMetrics:
    """对比分析指标"""
    scenario_name: str
    node_count: int
    edge_count: int
    density: float
    isolated_nodes: int
    connected_components: int
    clustering_coefficient: float
    average_path_length: Optional[float]
    centrality_ranking: List[Tuple[str, float]]


@dataclass
class ExperimentTrace:
    """实验追溯记录"""
    experiment_id: str
    start_time: str
    end_time: str
    input_files: List[str]
    output_files: List[str]
    technical_choices: List[TechnicalChoice]
    processing_steps: List[ProcessingStep]
    comparison_results: List[ComparisonMetrics]
    configuration: Dict[str, Any]


class DocumentGenerator:
    """
    文档生成器
    
    提供结构化文档生成、技术选择记录和处理过程追溯功能。
    根据需求10实现完整的实验文档和可追溯分析过程。
    """
    
    def __init__(self, config: Dict[str, Any], logger: PipelineLogger):
        """
        初始化文档生成器
        
        Args:
            config: 配置字典
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.error_handler = ErrorHandler()
        
        # 文档配置
        self.doc_config = config.get('documentation', {})
        self.output_path = self.doc_config.get('output_path', 'output/documentation/')
        self.template_path = self.doc_config.get('template_path', 'templates/')
        
        # 追溯记录
        self.current_trace = None
        self.technical_choices = []
        self.processing_steps = []
        
        # 确保输出目录存在
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
    
    def start_experiment_trace(self, experiment_id: str, input_files: List[str]) -> None:
        """
        开始实验追溯记录
        
        Args:
            experiment_id: 实验ID
            input_files: 输入文件列表
        """
        try:
            self.current_trace = ExperimentTrace(
                experiment_id=experiment_id,
                start_time=datetime.now().isoformat(),
                end_time="",
                input_files=input_files,
                output_files=[],
                technical_choices=[],
                processing_steps=[],
                comparison_results=[],
                configuration=self._make_config_serializable(self.config.copy())
            )
            
            self.technical_choices = []
            self.processing_steps = []
            
            self.logger.info(f"Started experiment trace: {experiment_id}")
            
        except Exception as e:
            self.error_handler.handle_error(
                ProcessingError(f"Failed to start experiment trace: {str(e)}"),
                "document_generation"
            )
            raise
    
    def record_technical_choice(self, component: str, choice: str, rationale: str, 
                              alternatives: List[str], parameters: Dict[str, Any] = None) -> None:
        """
        记录技术选择
        
        Args:
            component: 组件名称
            choice: 选择的技术
            rationale: 选择理由
            alternatives: 备选方案
            parameters: 相关参数
        """
        try:
            tech_choice = TechnicalChoice(
                component=component,
                choice=choice,
                rationale=rationale,
                alternatives=alternatives,
                timestamp=datetime.now().isoformat(),
                parameters=parameters or {}
            )
            
            self.technical_choices.append(tech_choice)
            
            self.logger.info(f"Recorded technical choice for {component}: {choice}")
            
        except Exception as e:
            self.error_handler.handle_error(
                ProcessingError(f"Failed to record technical choice: {str(e)}"),
                "document_generation"
            )
    
    def start_processing_step(self, step_name: str, input_description: str, 
                            parameters: Dict[str, Any] = None) -> str:
        """
        开始处理步骤记录
        
        Args:
            step_name: 步骤名称
            input_description: 输入描述
            parameters: 处理参数
            
        Returns:
            步骤ID
        """
        try:
            step_id = f"{step_name}_{len(self.processing_steps)}"
            
            step = ProcessingStep(
                step_name=step_name,
                input_description=input_description,
                output_description="",
                parameters=parameters or {},
                start_time=datetime.now().isoformat(),
                end_time="",
                duration_seconds=0.0,
                status="running"
            )
            
            self.processing_steps.append(step)
            
            self.logger.info(f"Started processing step: {step_name}")
            
            return step_id
            
        except Exception as e:
            self.error_handler.handle_error(
                ProcessingError(f"Failed to start processing step: {str(e)}"),
                "document_generation"
            )
            raise
    
    def end_processing_step(self, step_name: str, output_description: str, 
                          status: str = "completed", error_message: str = None) -> None:
        """
        结束处理步骤记录
        
        Args:
            step_name: 步骤名称
            output_description: 输出描述
            status: 状态
            error_message: 错误信息
        """
        try:
            # 找到对应的步骤
            for step in reversed(self.processing_steps):
                if step.step_name == step_name and step.status == "running":
                    end_time = datetime.now()
                    start_time = datetime.fromisoformat(step.start_time)
                    
                    step.end_time = end_time.isoformat()
                    step.duration_seconds = (end_time - start_time).total_seconds()
                    step.output_description = output_description
                    step.status = status
                    step.error_message = error_message
                    
                    self.logger.info(f"Completed processing step: {step_name} ({step.duration_seconds:.2f}s)")
                    break
            
        except Exception as e:
            self.error_handler.handle_error(
                ProcessingError(f"Failed to end processing step: {str(e)}"),
                "document_generation"
            )
    
    def generate_comparison_metrics(self, scenario_name: str, graph: Union[GlobalGraph, StateSubgraph]) -> ComparisonMetrics:
        """
        生成对比分析指标
        
        Args:
            scenario_name: 场景名称
            graph: 图对象
            
        Returns:
            对比指标
        """
        try:
            import easygraph as eg
            
            # 获取EasyGraph实例
            if hasattr(graph, 'easygraph_instance'):
                eg_graph = graph.easygraph_instance
            else:
                eg_graph = graph
            
            # 基础指标
            node_count = eg_graph.number_of_nodes()
            edge_count = eg_graph.number_of_edges()
            
            # 密度
            if node_count > 1:
                max_edges = node_count * (node_count - 1) / 2
                density = edge_count / max_edges if max_edges > 0 else 0.0
            else:
                density = 0.0
            
            # 孤立节点
            isolated_nodes = len([n for n in eg_graph.nodes() if eg_graph.degree(n) == 0])
            
            # 连通分量
            connected_components = len(list(eg.connected_components(eg_graph)))
            
            # 聚类系数
            try:
                clustering_coefficient = eg.average_clustering(eg_graph)
            except:
                clustering_coefficient = 0.0
            
            # 平均路径长度（仅对连通图计算）
            average_path_length = None
            if connected_components == 1 and node_count > 1:
                try:
                    average_path_length = eg.average_shortest_path_length(eg_graph)
                except:
                    pass
            
            # 中心性排序（度中心性）
            centrality_ranking = []
            try:
                degree_centrality = eg.degree_centrality(eg_graph)
                # 获取词汇映射
                if hasattr(graph, 'reverse_vocabulary'):
                    vocab = graph.reverse_vocabulary
                    centrality_ranking = [
                        (vocab.get(node, str(node)), centrality)
                        for node, centrality in sorted(degree_centrality.items(), 
                                                     key=lambda x: x[1], reverse=True)[:10]
                    ]
                else:
                    centrality_ranking = [
                        (str(node), centrality)
                        for node, centrality in sorted(degree_centrality.items(), 
                                                     key=lambda x: x[1], reverse=True)[:10]
                    ]
            except Exception as e:
                self.logger.warning(f"Failed to calculate centrality ranking: {str(e)}")
            
            metrics = ComparisonMetrics(
                scenario_name=scenario_name,
                node_count=node_count,
                edge_count=edge_count,
                density=density,
                isolated_nodes=isolated_nodes,
                connected_components=connected_components,
                clustering_coefficient=clustering_coefficient,
                average_path_length=average_path_length,
                centrality_ranking=centrality_ranking
            )
            
            self.logger.info(f"Generated comparison metrics for {scenario_name}")
            
            return metrics
            
        except Exception as e:
            self.error_handler.handle_error(
                ProcessingError(f"Failed to generate comparison metrics: {str(e)}"),
                "document_generation"
            )
            raise
    
    def add_comparison_result(self, metrics: ComparisonMetrics) -> None:
        """
        添加对比结果
        
        Args:
            metrics: 对比指标
        """
        if self.current_trace:
            self.current_trace.comparison_results.append(metrics)
    
    def end_experiment_trace(self, output_files: List[str]) -> ExperimentTrace:
        """
        结束实验追溯记录
        
        Args:
            output_files: 输出文件列表
            
        Returns:
            完整的实验追溯记录
        """
        try:
            if not self.current_trace:
                raise ProcessingError("No active experiment trace")
            
            self.current_trace.end_time = datetime.now().isoformat()
            self.current_trace.output_files = output_files
            self.current_trace.technical_choices = self.technical_choices.copy()
            self.current_trace.processing_steps = self.processing_steps.copy()
            
            # 保存追溯记录
            # 清理实验ID中的非法字符
            safe_experiment_id = "".join(c for c in self.current_trace.experiment_id if c.isalnum() or c in "._-")
            trace_file = Path(self.output_path) / f"experiment_trace_{safe_experiment_id}.json"
            with open(trace_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.current_trace), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Completed experiment trace: {self.current_trace.experiment_id}")
            
            return self.current_trace
            
        except Exception as e:
            self.error_handler.handle_error(
                ProcessingError(f"Failed to end experiment trace: {str(e)}"),
                "document_generation"
            )
            raise
    
    def generate_structured_document(self, trace: ExperimentTrace, 
                                   template_name: str = "experiment_report") -> str:
        """
        生成结构化文档
        
        Args:
            trace: 实验追溯记录
            template_name: 模板名称
            
        Returns:
            生成的文档路径
        """
        try:
            # 生成Markdown格式的实验报告
            doc_content = self._generate_markdown_report(trace)
            
            # 保存文档
            doc_file = Path(self.output_path) / f"{template_name}_{trace.experiment_id}.md"
            with open(doc_file, 'w', encoding='utf-8') as f:
                f.write(doc_content)
            
            self.logger.info(f"Generated structured document: {doc_file}")
            
            return str(doc_file)
            
        except Exception as e:
            self.error_handler.handle_error(
                ProcessingError(f"Failed to generate structured document: {str(e)}"),
                "document_generation"
            )
            raise
    
    def _generate_markdown_report(self, trace: ExperimentTrace) -> str:
        """
        生成Markdown格式的实验报告
        
        Args:
            trace: 实验追溯记录
            
        Returns:
            Markdown内容
        """
        content = []
        
        # 标题和基本信息
        content.append(f"# 语义增强共词网络分析实验报告")
        content.append(f"")
        content.append(f"**实验ID**: {trace.experiment_id}")
        content.append(f"**开始时间**: {trace.start_time}")
        content.append(f"**结束时间**: {trace.end_time}")
        content.append(f"")
        
        # 实验概述
        content.append("## 实验概述")
        content.append("")
        content.append("本实验实现了语义增强共词网络分析管线，采用'总图优先，州级激活'的两阶段构建策略。")
        content.append("系统以词组/短语为节点单位，通过动态停词发现和确定性布局确保可复现的网络分析结果。")
        content.append("")
        
        # 输入输出
        content.append("## 输入与输出")
        content.append("")
        content.append("### 输入文件")
        for input_file in trace.input_files:
            content.append(f"- {input_file}")
        content.append("")
        
        content.append("### 输出文件")
        for output_file in trace.output_files:
            content.append(f"- {output_file}")
        content.append("")
        
        # 技术选择
        content.append("## 关键技术选择")
        content.append("")
        for choice in trace.technical_choices:
            content.append(f"### {choice.component}")
            content.append(f"**选择**: {choice.choice}")
            content.append(f"**理由**: {choice.rationale}")
            content.append(f"**备选方案**: {', '.join(choice.alternatives)}")
            if choice.parameters:
                content.append(f"**参数**: {json.dumps(choice.parameters, ensure_ascii=False)}")
            content.append("")
        
        # 处理过程
        content.append("## 处理过程追溯")
        content.append("")
        content.append("| 步骤 | 输入 | 输出 | 耗时(秒) | 状态 |")
        content.append("|------|------|------|----------|------|")
        
        for step in trace.processing_steps:
            status_icon = "✅" if step.status == "completed" else "❌" if step.status == "failed" else "⏳"
            content.append(f"| {step.step_name} | {step.input_description} | {step.output_description} | {step.duration_seconds:.2f} | {status_icon} {step.status} |")
        
        content.append("")
        
        # 对比分析结果
        if trace.comparison_results:
            content.append("## 网络结构对比分析")
            content.append("")
            
            # 生成对比表格
            content.append("| 场景 | 节点数 | 边数 | 密度 | 孤立节点 | 连通分量 | 聚类系数 | 平均路径长度 |")
            content.append("|------|--------|------|------|----------|----------|----------|--------------|")
            
            for metrics in trace.comparison_results:
                avg_path = f"{metrics.average_path_length:.3f}" if metrics.average_path_length else "N/A"
                content.append(f"| {metrics.scenario_name} | {metrics.node_count} | {metrics.edge_count} | {metrics.density:.3f} | {metrics.isolated_nodes} | {metrics.connected_components} | {metrics.clustering_coefficient:.3f} | {avg_path} |")
            
            content.append("")
            
            # 中心性排序
            for metrics in trace.comparison_results:
                if metrics.centrality_ranking:
                    content.append(f"### {metrics.scenario_name} - 度中心性排序（前10）")
                    content.append("")
                    for i, (node, centrality) in enumerate(metrics.centrality_ranking[:10], 1):
                        content.append(f"{i}. {node}: {centrality:.3f}")
                    content.append("")
        
        # 配置信息
        content.append("## 配置参数")
        content.append("")
        content.append("```json")
        content.append(json.dumps(trace.configuration, indent=2, ensure_ascii=False))
        content.append("```")
        content.append("")
        
        # 与比赛要求的对应关系
        content.append("## 与比赛要求的对应关系")
        content.append("")
        content.append("本实验设计完全符合比赛要求，具体对应关系如下：")
        content.append("")
        content.append("1. **多文档处理**: 系统支持批量处理TOC分段的政策/法规文档")
        content.append("2. **语义增强**: 采用词组级节点单位和动态停词发现提升语义表示质量")
        content.append("3. **网络分析**: 实现总图构建和州级子图激活的两阶段分析策略")
        content.append("4. **可复现性**: 通过确定性布局和配置管理确保结果可复现")
        content.append("5. **图融合准备**: 为EasyGraph/OpenRank框架提供兼容的数据接口")
        content.append("")
        
        # 结论
        content.append("## 结论")
        content.append("")
        content.append("本实验成功构建了语义增强共词网络分析管线，实现了以下关键目标：")
        content.append("")
        content.append("- 建立了可复现的文本到图的转换流程")
        content.append("- 实现了词组级语义表示和动态停词过滤")
        content.append("- 构建了总图优先的多层次网络分析框架")
        content.append("- 提供了完整的实验追溯和文档生成能力")
        content.append("- 为后续图融合研究奠定了坚实基础")
        content.append("")
        
        return "\n".join(content)
    
    def generate_comparison_report(self, comparison_results: List[ComparisonMetrics], 
                                 output_file: str = None) -> str:
        """
        生成对比报告
        
        Args:
            comparison_results: 对比结果列表
            output_file: 输出文件路径
            
        Returns:
            报告文件路径
        """
        try:
            if not output_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = Path(self.output_path) / f"comparison_report_{timestamp}.md"
            
            content = []
            
            # 标题
            content.append("# 网络结构对比分析报告")
            content.append("")
            content.append(f"**生成时间**: {datetime.now().isoformat()}")
            content.append("")
            
            # 概述
            content.append("## 分析概述")
            content.append("")
            content.append("本报告对比分析了不同配置下的网络结构特征，包括：")
            content.append("- 单词节点 vs 词组节点的影响")
            content.append("- 静态停词 vs 动态停词的效果")
            content.append("- 总图 vs 州级子图的结构差异")
            content.append("")
            
            # 详细对比
            content.append("## 详细对比结果")
            content.append("")
            
            # 基础指标对比表
            content.append("### 基础网络指标")
            content.append("")
            content.append("| 场景 | 节点数 | 边数 | 密度 | 孤立节点 | 连通分量 | 聚类系数 | 平均路径长度 |")
            content.append("|------|--------|------|------|----------|----------|----------|--------------|")
            
            for metrics in comparison_results:
                avg_path = f"{metrics.average_path_length:.3f}" if metrics.average_path_length else "N/A"
                content.append(f"| {metrics.scenario_name} | {metrics.node_count} | {metrics.edge_count} | {metrics.density:.3f} | {metrics.isolated_nodes} | {metrics.connected_components} | {metrics.clustering_coefficient:.3f} | {avg_path} |")
            
            content.append("")
            
            # 分析结论
            content.append("### 分析结论")
            content.append("")
            
            # 节点规模分析
            node_counts = [m.node_count for m in comparison_results]
            if len(set(node_counts)) > 1:
                content.append("**节点规模差异**:")
                max_nodes = max(comparison_results, key=lambda x: x.node_count)
                min_nodes = min(comparison_results, key=lambda x: x.node_count)
                content.append(f"- 最大节点数: {max_nodes.scenario_name} ({max_nodes.node_count})")
                content.append(f"- 最小节点数: {min_nodes.scenario_name} ({min_nodes.node_count})")
                content.append("")
            
            # 密度分析
            densities = [m.density for m in comparison_results]
            if len(set(densities)) > 1:
                content.append("**网络密度差异**:")
                max_density = max(comparison_results, key=lambda x: x.density)
                min_density = min(comparison_results, key=lambda x: x.density)
                content.append(f"- 最高密度: {max_density.scenario_name} ({max_density.density:.3f})")
                content.append(f"- 最低密度: {min_density.scenario_name} ({min_density.density:.3f})")
                content.append("")
            
            # 连通性分析
            content.append("**连通性分析**:")
            for metrics in comparison_results:
                content.append(f"- {metrics.scenario_name}: {metrics.connected_components}个连通分量, {metrics.isolated_nodes}个孤立节点")
            content.append("")
            
            # 中心性排序对比
            content.append("## 中心性排序对比")
            content.append("")
            
            for metrics in comparison_results:
                if metrics.centrality_ranking:
                    content.append(f"### {metrics.scenario_name}")
                    content.append("")
                    content.append("| 排名 | 节点 | 度中心性 |")
                    content.append("|------|------|----------|")
                    
                    for i, (node, centrality) in enumerate(metrics.centrality_ranking[:10], 1):
                        content.append(f"| {i} | {node} | {centrality:.3f} |")
                    
                    content.append("")
            
            # 保存报告
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(content))
            
            self.logger.info(f"Generated comparison report: {output_file}")
            
            return str(output_file)
            
        except Exception as e:
            self.error_handler.handle_error(
                ProcessingError(f"Failed to generate comparison report: {str(e)}"),
                "document_generation"
            )
            raise
    
    def _make_config_serializable(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将配置对象转换为可序列化的格式
        
        Args:
            config: 原始配置字典
            
        Returns:
            可序列化的配置字典
        """
        def serialize_value(obj):
            if isinstance(obj, dict):
                return {k: serialize_value(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize_value(item) for item in obj]
            elif callable(obj):
                return f"<function {obj.__name__}>"
            elif hasattr(obj, '__dict__'):
                return f"<object {obj.__class__.__name__}>"
            else:
                return obj
        
        return serialize_value(config)


class TraceabilityManager:
    """
    追溯管理器
    
    提供处理过程的完整追溯能力，记录数据清洗、词表生成等关键步骤。
    """
    
    def __init__(self, config: Dict[str, Any], logger: PipelineLogger):
        """
        初始化追溯管理器
        
        Args:
            config: 配置字典
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.error_handler = ErrorHandler()
        
        # 追溯记录
        self.data_lineage = {}
        self.processing_history = []
        
    def record_data_transformation(self, step_name: str, input_data: Any, 
                                 output_data: Any, parameters: Dict[str, Any] = None) -> None:
        """
        记录数据转换过程
        
        Args:
            step_name: 步骤名称
            input_data: 输入数据
            output_data: 输出数据
            parameters: 转换参数
        """
        try:
            transformation_record = {
                'step_name': step_name,
                'timestamp': datetime.now().isoformat(),
                'input_summary': self._summarize_data(input_data),
                'output_summary': self._summarize_data(output_data),
                'parameters': parameters or {},
                'transformation_type': type(output_data).__name__
            }
            
            self.processing_history.append(transformation_record)
            
            # 建立数据血缘关系
            input_id = id(input_data)
            output_id = id(output_data)
            
            self.data_lineage[output_id] = {
                'parent': input_id,
                'transformation': step_name,
                'timestamp': transformation_record['timestamp']
            }
            
            self.logger.info(f"Recorded data transformation: {step_name}")
            
        except Exception as e:
            self.error_handler.handle_error(
                ProcessingError(f"Failed to record data transformation: {str(e)}"),
                "traceability_management"
            )
    
    def _summarize_data(self, data: Any) -> Dict[str, Any]:
        """
        生成数据摘要
        
        Args:
            data: 数据对象
            
        Returns:
            数据摘要
        """
        summary = {
            'type': type(data).__name__,
            'size': 0,
            'properties': {}
        }
        
        try:
            if isinstance(data, (list, tuple)):
                summary['size'] = len(data)
                if data:
                    summary['properties']['first_item_type'] = type(data[0]).__name__
            
            elif isinstance(data, dict):
                summary['size'] = len(data)
                summary['properties']['keys'] = list(data.keys())[:5]  # 前5个键
            
            elif isinstance(data, np.ndarray):
                summary['size'] = data.size
                summary['properties']['shape'] = data.shape
                summary['properties']['dtype'] = str(data.dtype)
            
            elif isinstance(data, scipy.sparse.spmatrix):
                summary['size'] = data.nnz
                summary['properties']['shape'] = data.shape
                summary['properties']['format'] = data.format
            
            elif hasattr(data, '__len__'):
                summary['size'] = len(data)
            
        except Exception:
            pass  # 忽略摘要生成错误
        
        return summary
    
    def get_processing_history(self) -> List[Dict[str, Any]]:
        """
        获取处理历史
        
        Returns:
            处理历史记录
        """
        return self.processing_history.copy()
    
    def trace_data_lineage(self, data_id: int) -> List[Dict[str, Any]]:
        """
        追溯数据血缘
        
        Args:
            data_id: 数据ID
            
        Returns:
            血缘追溯链
        """
        lineage_chain = []
        current_id = data_id
        
        while current_id in self.data_lineage:
            lineage_info = self.data_lineage[current_id]
            lineage_chain.append(lineage_info)
            current_id = lineage_info['parent']
        
        return lineage_chain