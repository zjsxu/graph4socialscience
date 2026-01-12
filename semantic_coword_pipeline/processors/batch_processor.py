"""
批处理器

实现自动批处理功能，支持对所有toc_doc文件夹的自动批处理。
根据需求7.1提供完整的批处理能力。
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from dataclasses import dataclass

from ..core.data_models import TOCDocument, ProcessedDocument, GlobalGraph, StateSubgraph
from ..core.config import Config
from ..core.logger import PipelineLogger, ProcessTracker
from ..core.error_handler import ErrorHandler, ProcessingError
from .text_processor import TextProcessor
from .phrase_extractor import PhraseExtractor
from .dynamic_stopword_discoverer import DynamicStopwordDiscoverer
from .global_graph_builder import GlobalGraphBuilder
from .state_subgraph_activator import StateSubgraphActivator
from .deterministic_layout_engine import DeterministicLayoutEngine
from .output_manager import OutputManager


@dataclass
class BatchProcessingResult:
    """批处理结果"""
    total_files: int
    processed_files: int
    failed_files: int
    processing_time: float
    global_graph: Optional[GlobalGraph]
    state_subgraphs: Dict[str, StateSubgraph]
    error_summary: Dict[str, Any]
    output_files: List[str]


class BatchProcessor:
    """
    批处理器
    
    提供自动批处理功能，处理指定文件夹中的所有TOC文档。
    根据需求7.1实现对所有toc_doc文件夹的自动批处理。
    """
    
    def __init__(self, config: Config):
        """
        初始化批处理器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = PipelineLogger("BatchProcessor", config.get_section('logging'))
        self.error_handler = ErrorHandler(config.get_section('error_handling'))
        self.process_tracker = ProcessTracker(self.logger)
        
        # 初始化处理组件
        self._initialize_processors()
        
        # 批处理配置
        self.batch_size = config.get('performance.batch_size', 1000)
        self.max_workers = config.get('performance.max_workers', 4)
        self.enable_parallel = config.get('performance.enable_parallel_processing', True)
        self.memory_limit_mb = config.get('performance.memory_limit_mb', 4096)
    
    def _initialize_processors(self) -> None:
        """初始化处理组件"""
        try:
            self.text_processor = TextProcessor(self.config)
            self.phrase_extractor = PhraseExtractor(self.config)
            self.stopword_discoverer = DynamicStopwordDiscoverer(self.config)
            self.graph_builder = GlobalGraphBuilder(self.config)
            self.subgraph_activator = StateSubgraphActivator(self.config)
            self.layout_engine = DeterministicLayoutEngine(self.config)
            
            self.logger.info("All processors initialized successfully")
            
        except Exception as e:
            self.error_handler.handle_error(e, "processor_initialization")
    
    def process_directory(self, input_dir: str, output_dir: str) -> BatchProcessingResult:
        """
        处理指定目录中的所有TOC文档
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
            
        Returns:
            批处理结果
        """
        start_time = time.time()
        
        # 开始处理过程追踪
        self.process_tracker.start_process("batch_processing", {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'config': self.config.get_all()
        })
        
        try:
            # 发现输入文件
            input_files = self._discover_input_files(input_dir)
            self.logger.info(f"Discovered {len(input_files)} input files")
            
            if not input_files:
                raise ProcessingError("No valid input files found", context={'input_dir': input_dir})
            
            # 处理文档
            processed_docs = self._process_documents(input_files)
            self.logger.info(f"Successfully processed {len(processed_docs)} documents")
            
            # 构建全局图
            global_graph = self._build_global_graph(processed_docs)
            self.logger.info(f"Built global graph with {global_graph.get_node_count()} nodes")
            
            # 生成州级子图
            state_subgraphs = self._generate_state_subgraphs(global_graph, processed_docs)
            self.logger.info(f"Generated {len(state_subgraphs)} state subgraphs")
            
            # 计算布局
            self._compute_layouts(global_graph, state_subgraphs)
            self.logger.info("Computed layouts for all graphs")
            
            # 生成输出文件
            output_files = self._generate_outputs(global_graph, state_subgraphs, processed_docs, output_dir)
            self.logger.info(f"Generated {len(output_files)} output files")
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 创建结果对象
            result = BatchProcessingResult(
                total_files=len(input_files),
                processed_files=len(processed_docs),
                failed_files=len(input_files) - len(processed_docs),
                processing_time=processing_time,
                global_graph=global_graph,
                state_subgraphs=state_subgraphs,
                error_summary=self.error_handler.generate_error_report(),
                output_files=output_files
            )
            
            # 结束处理过程追踪
            self.process_tracker.end_process({
                'result': {
                    'total_files': result.total_files,
                    'processed_files': result.processed_files,
                    'failed_files': result.failed_files,
                    'processing_time': result.processing_time,
                    'output_files_count': len(result.output_files)
                }
            })
            
            self.logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
            return result
            
        except Exception as e:
            # 记录错误并结束追踪
            self.process_tracker.end_process({
                'error': str(e)
            }, status='failed')
            
            self.error_handler.handle_error(e, "batch_processing")
    
    def _discover_input_files(self, input_dir: str) -> List[str]:
        """
        发现输入文件
        
        Args:
            input_dir: 输入目录
            
        Returns:
            输入文件路径列表
        """
        input_files = []
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise ProcessingError(f"Input directory does not exist: {input_dir}")
        
        # 递归查找JSON文件
        for file_path in input_path.rglob("*.json"):
            if file_path.is_file():
                input_files.append(str(file_path))
        
        # 验证文件格式
        valid_files = []
        for file_path in input_files:
            if self._validate_input_file(file_path):
                valid_files.append(file_path)
            else:
                self.logger.warning(f"Invalid input file format: {file_path}")
        
        self.process_tracker.add_step("file_discovery", {
            'input_dir': input_dir,
            'total_files_found': len(input_files)
        }, {
            'valid_files': len(valid_files),
            'invalid_files': len(input_files) - len(valid_files)
        })
        
        return valid_files
    
    def _validate_input_file(self, file_path: str) -> bool:
        """
        验证输入文件格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否为有效的TOC文档格式
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否为列表格式
            if isinstance(data, list):
                # 验证第一个元素
                if data and isinstance(data[0], dict):
                    return self._validate_toc_document(data[0])
            elif isinstance(data, dict):
                return self._validate_toc_document(data)
            
            return False
            
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            return False
    
    def _validate_toc_document(self, doc_data: Dict[str, Any]) -> bool:
        """验证TOC文档格式"""
        required_fields = ['segment_id', 'title', 'level', 'order', 'text']
        return all(field in doc_data for field in required_fields)
    
    def _process_documents(self, input_files: List[str]) -> List[ProcessedDocument]:
        """
        处理文档
        
        Args:
            input_files: 输入文件列表
            
        Returns:
            处理后的文档列表
        """
        processed_docs = []
        
        if self.enable_parallel and len(input_files) > 1:
            # 并行处理
            processed_docs = self._process_documents_parallel(input_files)
        else:
            # 串行处理
            processed_docs = self._process_documents_sequential(input_files)
        
        self.process_tracker.add_step("document_processing", {
            'input_files_count': len(input_files),
            'parallel_processing': self.enable_parallel
        }, {
            'processed_documents': len(processed_docs),
            'success_rate': len(processed_docs) / len(input_files) if input_files else 0
        })
        
        return processed_docs
    
    def _process_documents_sequential(self, input_files: List[str]) -> List[ProcessedDocument]:
        """串行处理文档"""
        processed_docs = []
        
        for i, file_path in enumerate(input_files):
            try:
                self.logger.info(f"Processing file {i+1}/{len(input_files)}: {file_path}")
                
                # 加载文档
                toc_docs = self._load_toc_documents(file_path)
                
                # 处理每个文档
                for toc_doc in toc_docs:
                    try:
                        processed_doc = self._process_single_document(toc_doc)
                        processed_docs.append(processed_doc)
                    except Exception as e:
                        self.error_handler.log_warning(
                            f"Failed to process document {toc_doc.segment_id}: {e}",
                            context={'file_path': file_path, 'segment_id': toc_doc.segment_id}
                        )
                
            except Exception as e:
                self.error_handler.log_warning(
                    f"Failed to load file {file_path}: {e}",
                    context={'file_path': file_path}
                )
        
        return processed_docs
    
    def _process_documents_parallel(self, input_files: List[str]) -> List[ProcessedDocument]:
        """并行处理文档"""
        processed_docs = []
        
        # 限制并发数量以控制内存使用
        max_workers = min(self.max_workers, len(input_files))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            future_to_file = {
                executor.submit(self._process_file, file_path): file_path 
                for file_path in input_files
            }
            
            # 收集结果
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_processed_docs = future.result()
                    processed_docs.extend(file_processed_docs)
                    self.logger.debug(f"Completed processing file: {file_path}")
                except Exception as e:
                    self.error_handler.log_warning(
                        f"Failed to process file {file_path}: {e}",
                        context={'file_path': file_path}
                    )
        
        return processed_docs
    
    def _process_file(self, file_path: str) -> List[ProcessedDocument]:
        """处理单个文件"""
        processed_docs = []
        
        try:
            # 加载文档
            toc_docs = self._load_toc_documents(file_path)
            
            # 处理每个文档
            for toc_doc in toc_docs:
                try:
                    processed_doc = self._process_single_document(toc_doc)
                    processed_docs.append(processed_doc)
                except Exception as e:
                    self.error_handler.log_warning(
                        f"Failed to process document {toc_doc.segment_id}: {e}",
                        context={'file_path': file_path, 'segment_id': toc_doc.segment_id}
                    )
        
        except Exception as e:
            self.error_handler.log_warning(
                f"Failed to load file {file_path}: {e}",
                context={'file_path': file_path}
            )
        
        return processed_docs
    
    def _load_toc_documents(self, file_path: str) -> List[TOCDocument]:
        """加载TOC文档"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            toc_docs = []
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        toc_doc = TOCDocument.from_json(item)
                        # 从文件路径推断州信息
                        if not toc_doc.state:
                            toc_doc.state = self._extract_state_from_path(file_path)
                        toc_docs.append(toc_doc)
            elif isinstance(data, dict):
                toc_doc = TOCDocument.from_json(data)
                if not toc_doc.state:
                    toc_doc.state = self._extract_state_from_path(file_path)
                toc_docs.append(toc_doc)
            
            return toc_docs
            
        except Exception as e:
            raise ProcessingError(f"Failed to load TOC documents from {file_path}: {e}")
    
    def _extract_state_from_path(self, file_path: str) -> str:
        """从文件路径提取州信息"""
        path_parts = Path(file_path).parts
        
        # 尝试从路径中找到州名
        for part in path_parts:
            if len(part) == 2 and part.isupper():  # 可能是州缩写
                return part
            elif 'state' in part.lower():
                return part
        
        # 如果找不到，使用文件名（不含扩展名）
        return Path(file_path).stem
    
    def _process_single_document(self, toc_doc: TOCDocument) -> ProcessedDocument:
        """处理单个文档"""
        try:
            # 文本处理
            processed_doc = self.text_processor.process_document(toc_doc)
            
            # 词组抽取
            if processed_doc.tokens:
                processed_doc = self.phrase_extractor.extract_phrases_from_document(processed_doc)
            
            return processed_doc
            
        except Exception as e:
            raise ProcessingError(f"Failed to process document {toc_doc.segment_id}: {e}")
    
    def _build_global_graph(self, processed_docs: List[ProcessedDocument]) -> GlobalGraph:
        """构建全局图"""
        try:
            # 发现动态停词
            all_phrases = [doc.phrases for doc in processed_docs if doc.phrases]
            if all_phrases:
                stopwords = self.stopword_discoverer.discover_stopwords(all_phrases)
                self.logger.info(f"Discovered {len(stopwords)} dynamic stopwords")
            else:
                stopwords = set()
            
            # 构建全局图
            global_graph = self.graph_builder.build_global_graph(processed_docs)
            
            self.process_tracker.add_step("global_graph_construction", {
                'input_documents': len(processed_docs),
                'total_phrases': sum(len(doc.phrases) for doc in processed_docs),
                'dynamic_stopwords': len(stopwords)
            }, {
                'graph_nodes': global_graph.get_node_count(),
                'vocabulary_size': len(global_graph.vocabulary)
            })
            
            return global_graph
            
        except Exception as e:
            self.error_handler.handle_graph_construction_error(e, "global_graph")
    
    def _generate_state_subgraphs(self, global_graph: GlobalGraph, 
                                processed_docs: List[ProcessedDocument]) -> Dict[str, StateSubgraph]:
        """生成州级子图"""
        state_subgraphs = {}
        
        # 按州分组文档
        docs_by_state = {}
        for doc in processed_docs:
            state = doc.original_doc.state or 'unknown'
            if state not in docs_by_state:
                docs_by_state[state] = []
            docs_by_state[state].append(doc)
        
        # 为每个州生成子图
        for state, state_docs in docs_by_state.items():
            try:
                subgraph = self.subgraph_activator.activate_state_subgraph(global_graph, state_docs)
                subgraph.state_name = state
                state_subgraphs[state] = subgraph
                
                self.logger.info(f"Generated subgraph for state {state} with {len(state_docs)} documents")
                
            except Exception as e:
                self.error_handler.log_warning(
                    f"Failed to generate subgraph for state {state}: {e}",
                    context={'state': state, 'document_count': len(state_docs)}
                )
        
        self.process_tracker.add_step("state_subgraph_generation", {
            'states_count': len(docs_by_state),
            'total_documents': len(processed_docs)
        }, {
            'generated_subgraphs': len(state_subgraphs),
            'states': list(state_subgraphs.keys())
        })
        
        return state_subgraphs
    
    def _compute_layouts(self, global_graph: GlobalGraph, 
                        state_subgraphs: Dict[str, StateSubgraph]) -> None:
        """计算布局"""
        try:
            # 计算全局图布局
            if global_graph.easygraph_instance:
                global_positions = self.layout_engine.compute_layout(
                    global_graph.easygraph_instance, 
                    "global_graph"
                )
                self.logger.info("Computed global graph layout")
            
            # 计算州级子图布局（使用全局位置）
            for state, subgraph in state_subgraphs.items():
                if subgraph.easygraph_instance:
                    positions = self.layout_engine.compute_layout(
                        subgraph.easygraph_instance, 
                        f"state_{state}",
                        base_positions=global_positions if 'global_positions' in locals() else None
                    )
                    subgraph.node_positions = positions
                    self.logger.debug(f"Computed layout for state {state}")
            
            self.process_tracker.add_step("layout_computation", {
                'global_graph_nodes': global_graph.get_node_count(),
                'state_subgraphs': len(state_subgraphs)
            }, {
                'layouts_computed': 1 + len(state_subgraphs)
            })
            
        except Exception as e:
            self.error_handler.handle_error(e, "layout_computation")
    
    def _generate_outputs(self, global_graph: GlobalGraph, 
                         state_subgraphs: Dict[str, StateSubgraph],
                         processed_docs: List[ProcessedDocument],
                         output_dir: str) -> List[str]:
        """生成输出文件"""
        try:
            # 创建输出管理器
            output_manager = OutputManager(self.config.get_all(), self.logger)
            
            # 生成所有输出文件
            output_files = output_manager.generate_all_outputs(
                global_graph=global_graph,
                state_subgraphs=state_subgraphs,
                processed_docs=processed_docs,
                output_dir=output_dir,
                error_report=self.error_handler.generate_error_report(),
                process_history=self.process_tracker.get_process_history()
            )
            
            self.process_tracker.add_step("output_generation", {
                'output_dir': output_dir,
                'global_graph_nodes': global_graph.get_node_count(),
                'state_subgraphs': len(state_subgraphs)
            }, {
                'output_files_generated': len(output_files),
                'output_files': output_files
            })
            
            return output_files
            
        except Exception as e:
            self.error_handler.handle_output_error(e, "batch_output_generation")
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            'error_summary': self.error_handler.generate_error_report(),
            'process_history': self.process_tracker.get_process_history(),
            'configuration': self.config.get_all()
        }