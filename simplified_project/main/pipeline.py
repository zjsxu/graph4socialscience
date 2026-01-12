"""
主管线模块

提供统一的管线入口和协调功能，整合所有处理组件。
根据需求7.1-7.6实现完整的批处理和输出管理功能。
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

from .core.config import Config
from .core.logger import PipelineLogger, ProcessTracker
from .core.error_handler import ErrorHandler
from .core.performance import PerformanceMonitor, MemoryProfiler, OptimizationAnalyzer
from .processors.batch_processor import BatchProcessor, BatchProcessingResult
from .processors.document_generator import DocumentGenerator, TraceabilityManager


class SemanticCowordPipeline:
    """
    语义增强共词网络分析主管线
    
    提供统一的入口点和协调功能，整合所有处理组件。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化主管线
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = Config(config_path)
        
        # 初始化日志系统
        self.logger = PipelineLogger("SemanticCowordPipeline", self.config.get_section('logging'))
        
        # 初始化错误处理器
        self.error_handler = ErrorHandler(self.config.get_section('error_handling'))
        
        # 初始化性能监控器
        self.performance_monitor = PerformanceMonitor(self.config.get_section('performance'))
        self.memory_profiler = MemoryProfiler()
        self.optimization_analyzer = OptimizationAnalyzer(self.performance_monitor)
        
        # 初始化处理追踪器
        self.process_tracker = ProcessTracker(self.logger)
        
        # 初始化批处理器
        self.batch_processor = BatchProcessor(self.config)
        
        # 初始化文档生成器和追溯管理器
        self.document_generator = DocumentGenerator(self.config, self.logger)
        self.traceability_manager = TraceabilityManager(self.config, self.logger)
        
        self.logger.info("Semantic Coword Pipeline initialized successfully")
    
    def run(self, input_dir: str, output_dir: str) -> BatchProcessingResult:
        """
        运行完整的管线处理
        
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
            
        Returns:
            批处理结果
        """
        self.logger.info(f"Starting pipeline processing: {input_dir} -> {output_dir}")
        
        # 启动性能监控
        if self.config.get('performance.enable_profiling', False):
            self.performance_monitor.start_monitoring()
            self.memory_profiler.start_profiling()
            self.memory_profiler.take_snapshot('pipeline_start')
        
        # 开始主要处理操作监控
        main_operation_id = self.performance_monitor.start_operation('semantic_coword_pipeline', {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'config_sections': list(self.config.get_all().keys())
        })
        
        # 生成实验ID
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 获取输入文件列表
        input_files = self._get_input_files(input_dir)
        
        # 开始实验追溯
        self.document_generator.start_experiment_trace(experiment_id, input_files)
        
        # 记录关键技术选择
        self._record_technical_choices()
        
        # 开始处理过程追踪
        self.process_tracker.start_process("semantic_coword_pipeline", {
            'input_dir': input_dir,
            'output_dir': output_dir,
            'config_path': self.config._config_path,
            'pipeline_version': self._get_pipeline_version(),
            'experiment_id': experiment_id
        })
        
        try:
            # 验证输入和输出路径
            self._validate_paths(input_dir, output_dir)
            
            # 开始批处理步骤
            step_id = self.document_generator.start_processing_step(
                "batch_processing",
                f"TOC documents from {input_dir}",
                {'input_dir': input_dir, 'output_dir': output_dir}
            )
            
            # 运行批处理
            result = self.batch_processor.process_directory(input_dir, output_dir)
            
            # 结束批处理步骤
            self.document_generator.end_processing_step(
                "batch_processing",
                f"Processed {result.processed_files} files, generated {len(result.output_files)} outputs",
                "completed" if result.failed_files == 0 else "completed_with_errors"
            )
            
            # 生成对比分析
            comparison_results = self._generate_comparison_analysis(result)
            
            # 生成最终报告
            final_report = self._generate_final_report(result, input_dir, output_dir)
            
            # 结束性能监控
            if self.config.get('performance.enable_profiling', False):
                self.performance_monitor.end_operation(main_operation_id)
                self.memory_profiler.take_snapshot('pipeline_end')
                
                # 生成性能报告
                performance_report = self._generate_performance_report(input_dir, output_dir)
                if performance_report:
                    output_files.append(performance_report)
                
                self.performance_monitor.stop_monitoring()
                self.memory_profiler.stop_profiling()
            
            # 结束实验追溯并生成文档
            output_files = result.output_files + [final_report]
            trace = self.document_generator.end_experiment_trace(output_files)
            
            # 生成结构化实验文档
            experiment_doc = self.document_generator.generate_structured_document(trace)
            
            # 生成对比报告
            if comparison_results:
                comparison_report = self.document_generator.generate_comparison_report(comparison_results)
                output_files.append(comparison_report)
            
            # 结束处理过程追踪
            self.process_tracker.end_process({
                'result_summary': {
                    'total_files': result.total_files,
                    'processed_files': result.processed_files,
                    'failed_files': result.failed_files,
                    'processing_time': result.processing_time,
                    'output_files_count': len(output_files)
                },
                'final_report_path': final_report,
                'experiment_document': experiment_doc,
                'experiment_id': experiment_id
            })
            
            self.logger.info(f"Pipeline processing completed successfully in {result.processing_time:.2f} seconds")
            self.logger.info(f"Generated experiment document: {experiment_doc}")
            
            return result
            
        except Exception as e:
            # 记录错误并结束追踪
            self.document_generator.end_processing_step(
                "batch_processing",
                f"Processing failed: {str(e)}",
                "failed",
                str(e)
            )
            
            self.process_tracker.end_process({
                'error': str(e),
                'error_type': type(e).__name__,
                'experiment_id': experiment_id
            }, status='failed')
            
            # 仍然尝试生成错误报告文档
            try:
                trace = self.document_generator.end_experiment_trace([])
                error_doc = self.document_generator.generate_structured_document(trace)
                self.logger.info(f"Generated error report document: {error_doc}")
            except:
                pass  # 忽略文档生成错误
            
            self.error_handler.handle_error(e, "pipeline_execution")
    
    def _validate_paths(self, input_dir: str, output_dir: str) -> None:
        """验证输入和输出路径"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 验证输入目录
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")
        
        # 创建输出目录
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create output directory {output_dir}: {e}")
        
        self.logger.info(f"Validated paths - Input: {input_dir}, Output: {output_dir}")
    
    def _generate_final_report(self, result: BatchProcessingResult, 
                              input_dir: str, output_dir: str) -> str:
        """生成最终报告"""
        try:
            report = {
                'pipeline_execution_report': {
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_version': self._get_pipeline_version(),
                    'input_directory': input_dir,
                    'output_directory': output_dir,
                    'configuration': self.config.get_all(),
                    'processing_results': {
                        'total_files': result.total_files,
                        'processed_files': result.processed_files,
                        'failed_files': result.failed_files,
                        'success_rate': result.processed_files / result.total_files if result.total_files > 0 else 0,
                        'processing_time_seconds': result.processing_time,
                        'output_files_generated': len(result.output_files)
                    },
                    'global_graph_summary': {
                        'node_count': result.global_graph.get_node_count() if result.global_graph else 0,
                        'vocabulary_size': len(result.global_graph.vocabulary) if result.global_graph else 0
                    },
                    'state_subgraphs_summary': {
                        'states_processed': len(result.state_subgraphs),
                        'states': list(result.state_subgraphs.keys())
                    },
                    'error_summary': result.error_summary,
                    'output_files': result.output_files,
                    'process_history': self.process_tracker.get_process_history()
                }
            }
            
            # 保存最终报告
            report_file = Path(output_dir) / 'pipeline_execution_report.json'
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Generated final report: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"Failed to generate final report: {e}")
            return ""
    
    def _get_pipeline_version(self) -> str:
        """获取管线版本信息"""
        return "1.0.0"  # 可以从版本文件或包信息中获取
    
    def _get_input_files(self, input_dir: str) -> list:
        """获取输入文件列表"""
        input_path = Path(input_dir)
        input_files = []
        
        for file_path in input_path.rglob("*.json"):
            input_files.append(str(file_path.relative_to(input_path)))
        
        return input_files
    
    def _record_technical_choices(self) -> None:
        """记录关键技术选择"""
        # 文本处理技术选择
        self.document_generator.record_technical_choice(
            component="text_processor",
            choice="NLTK + jieba",
            rationale="NLTK提供成熟的英文处理能力，jieba提供高质量的中文分词",
            alternatives=["spaCy", "Stanford CoreNLP", "自定义分词器"],
            parameters=self.config.get_section('text_processing')
        )
        
        # 词组抽取技术选择
        self.document_generator.record_technical_choice(
            component="phrase_extractor",
            choice="2-gram + 统计筛选",
            rationale="2-gram提供基础的词组单位，统计筛选确保质量",
            alternatives=["单词级别", "3-gram", "基于依存句法的短语"],
            parameters=self.config.get_section('phrase_extraction')
        )
        
        # 停词发现技术选择
        self.document_generator.record_technical_choice(
            component="stopword_discovery",
            choice="TF-IDF动态发现",
            rationale="自动识别高频低区分度词组，提升网络质量",
            alternatives=["仅静态停词表", "基于词频的过滤", "基于互信息的过滤"],
            parameters=self.config.get_section('stopword_discovery')
        )
        
        # 图构建技术选择
        self.document_generator.record_technical_choice(
            component="graph_builder",
            choice="总图优先 + 州级激活",
            rationale="确保跨州对比的一致性和可比性",
            alternatives=["独立构建各州图", "层次化图构建", "增量图构建"],
            parameters=self.config.get_section('graph_construction')
        )
        
        # 布局技术选择
        self.document_generator.record_technical_choice(
            component="layout_engine",
            choice="确定性力导向布局",
            rationale="固定种子确保可复现性，力导向布局提供直观的视觉效果",
            alternatives=["随机布局", "层次布局", "圆形布局"],
            parameters=self.config.get_section('layout')
        )
    
    def _generate_comparison_analysis(self, result: BatchProcessingResult) -> list:
        """生成对比分析结果"""
        comparison_results = []
        
        try:
            # 全局图指标
            if result.global_graph:
                global_metrics = self.document_generator.generate_comparison_metrics(
                    "全局共现网络", result.global_graph
                )
                comparison_results.append(global_metrics)
                self.document_generator.add_comparison_result(global_metrics)
            
            # 州级子图指标
            for state, subgraph in result.state_subgraphs.items():
                state_metrics = self.document_generator.generate_comparison_metrics(
                    f"{state}州子图", subgraph
                )
                comparison_results.append(state_metrics)
                self.document_generator.add_comparison_result(state_metrics)
            
            self.logger.info(f"Generated comparison analysis for {len(comparison_results)} scenarios")
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison analysis: {e}")
        
        return comparison_results
    
    def _generate_performance_report(self, input_dir: str, output_dir: str) -> Optional[str]:
        """生成性能分析报告"""
        try:
            # 生成性能报告
            performance_report = self.performance_monitor.generate_performance_report()
            
            # 添加内存分析
            if self.memory_profiler.enabled:
                memory_analysis = self.memory_profiler.compare_snapshots('pipeline_start', 'pipeline_end')
                performance_report['memory_analysis'] = memory_analysis
                performance_report['current_memory_usage'] = self.memory_profiler.get_current_memory_usage()
            
            # 添加优化建议
            bottlenecks = self.optimization_analyzer.analyze_bottlenecks()
            optimizations = self.optimization_analyzer.suggest_optimizations()
            
            performance_report['bottleneck_analysis'] = bottlenecks
            performance_report['optimization_suggestions'] = optimizations
            
            # 保存性能报告
            report_file = Path(output_dir) / 'performance_analysis_report.json'
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(performance_report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Generated performance report: {report_file}")
            return str(report_file)
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return None
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            'performance_monitoring_enabled': self.performance_monitor.enabled,
            'memory_profiling_enabled': self.memory_profiler.enabled,
            'operation_summary': self.performance_monitor.get_operation_summary(),
            'system_summary': self.performance_monitor.get_system_summary(),
            'optimization_recommendations': self.optimization_analyzer.suggest_optimizations()
        }
    
    def get_configuration(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.get_all()
    
    def update_configuration(self, config_updates: Dict[str, Any]) -> None:
        """更新配置"""
        self.config.update(config_updates)
        self.logger.info("Configuration updated")
    
    def validate_configuration(self) -> Dict[str, list]:
        """验证配置"""
        return self.config.validate()
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        return {
            'error_summary': self.error_handler.generate_error_report(),
            'process_history': self.process_tracker.get_process_history(),
            'configuration': self.config.get_all(),
            'batch_processor_stats': self.batch_processor.get_processing_statistics(),
            'performance_stats': self.get_performance_statistics()
        }


def create_default_config() -> str:
    """创建默认配置文件"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "pipeline_config.json"
    
    config = Config()
    config.save_to_file(str(config_file))
    
    return str(config_file)


def main():
    """命令行入口点"""
    parser = argparse.ArgumentParser(
        description="Semantic Coword Network Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基本用法
  python -m semantic_coword_pipeline.pipeline input_data/ output_results/
  
  # 使用自定义配置
  python -m semantic_coword_pipeline.pipeline input_data/ output_results/ --config config/custom.json
  
  # 创建默认配置文件
  python -m semantic_coword_pipeline.pipeline --create-config
  
  # 验证配置文件
  python -m semantic_coword_pipeline.pipeline --validate-config config/pipeline.json
        """
    )
    
    parser.add_argument(
        'input_dir', 
        nargs='?',
        help='Input directory containing TOC JSON files'
    )
    
    parser.add_argument(
        'output_dir', 
        nargs='?',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create default configuration file'
    )
    
    parser.add_argument(
        '--validate-config',
        help='Validate configuration file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Semantic Coword Pipeline 1.0.0'
    )
    
    args = parser.parse_args()
    
    # 处理特殊命令
    if args.create_config:
        config_file = create_default_config()
        print(f"Created default configuration file: {config_file}")
        return 0
    
    if args.validate_config:
        try:
            config = Config(args.validate_config)
            validation_result = config.validate()
            
            if validation_result['errors']:
                print("Configuration validation failed:")
                for error in validation_result['errors']:
                    print(f"  ERROR: {error}")
                return 1
            
            if validation_result['warnings']:
                print("Configuration validation warnings:")
                for warning in validation_result['warnings']:
                    print(f"  WARNING: {warning}")
            
            print("Configuration validation passed!")
            return 0
            
        except Exception as e:
            print(f"Failed to validate configuration: {e}")
            return 1
    
    # 验证必需参数
    if not args.input_dir or not args.output_dir:
        parser.error("input_dir and output_dir are required unless using --create-config or --validate-config")
    
    try:
        # 创建并运行管线
        pipeline = SemanticCowordPipeline(args.config)
        
        # 如果启用详细日志，更新配置
        if args.verbose:
            pipeline.update_configuration({'logging.level': 'DEBUG'})
        
        # 运行处理
        result = pipeline.run(args.input_dir, args.output_dir)
        
        # 输出结果摘要
        print(f"\nProcessing completed successfully!")
        print(f"  Total files: {result.total_files}")
        print(f"  Processed files: {result.processed_files}")
        print(f"  Failed files: {result.failed_files}")
        print(f"  Processing time: {result.processing_time:.2f} seconds")
        print(f"  Output files generated: {len(result.output_files)}")
        
        if result.global_graph:
            print(f"  Global graph nodes: {result.global_graph.get_node_count()}")
        
        print(f"  State subgraphs: {len(result.state_subgraphs)}")
        
        if result.error_summary.get('summary', {}).get('total_errors', 0) > 0:
            print(f"  Errors encountered: {result.error_summary['summary']['total_errors']}")
            print("  Check error report in output directory for details")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        return 1
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())