"""
命令行接口模块

提供统一的命令行入口点，支持多种操作模式和配置选项。
根据需求7.1-7.6和所有需求的集成验证。
"""

import sys
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .pipeline import SemanticCowordPipeline, create_default_config
from .core.config import Config
from .core.logger import PipelineLogger
from .core.error_handler import ErrorHandler


class CLIInterface:
    """
    命令行接口类
    
    提供完整的命令行功能，包括配置管理、性能监控和批处理。
    """
    
    def __init__(self):
        """初始化CLI接口"""
        self.logger = None
        self.error_handler = None
    
    def create_parser(self) -> argparse.ArgumentParser:
        """创建命令行参数解析器"""
        parser = argparse.ArgumentParser(
            prog='semantic-coword',
            description='语义增强共词网络分析管线',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  # 基本处理
  semantic-coword process input_data/ output_results/
  
  # 使用自定义配置
  semantic-coword process input_data/ output_results/ --config config/custom.json
  
  # 创建默认配置文件
  semantic-coword config create --output config/default.json
  
  # 验证配置文件
  semantic-coword config validate config/pipeline.json
  
  # 查看配置信息
  semantic-coword config show config/pipeline.json
  
  # 性能分析模式
  semantic-coword process input_data/ output_results/ --profile --memory-monitor
  
  # 批处理模式
  semantic-coword batch --input-pattern "data/*/toc_docs/" --output-base results/
            """
        )
        
        # 全局选项
        parser.add_argument(
            '--version',
            action='version',
            version='Semantic Coword Pipeline 1.0.0'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='count',
            default=0,
            help='增加详细程度 (可重复使用: -v, -vv, -vvv)'
        )
        
        parser.add_argument(
            '--quiet', '-q',
            action='store_true',
            help='静默模式，只输出错误信息'
        )
        
        parser.add_argument(
            '--config', '-c',
            help='配置文件路径'
        )
        
        # 子命令
        subparsers = parser.add_subparsers(
            dest='command',
            help='可用命令',
            metavar='COMMAND'
        )
        
        # process 子命令
        self._add_process_command(subparsers)
        
        # config 子命令
        self._add_config_command(subparsers)
        
        # batch 子命令
        self._add_batch_command(subparsers)
        
        # analyze 子命令
        self._add_analyze_command(subparsers)
        
        # performance 子命令
        self._add_performance_command(subparsers)
        
        return parser
    
    def _add_process_command(self, subparsers):
        """添加process子命令"""
        process_parser = subparsers.add_parser(
            'process',
            help='处理TOC文档生成共词网络',
            description='处理指定目录中的TOC JSON文档，生成语义增强的共词网络分析结果'
        )
        
        process_parser.add_argument(
            'input_dir',
            help='输入目录路径（包含TOC JSON文件）'
        )
        
        process_parser.add_argument(
            'output_dir',
            help='输出目录路径'
        )
        
        process_parser.add_argument(
            '--verbose', '-v',
            action='count',
            default=0,
            help='增加详细程度 (可重复使用: -v, -vv, -vvv)'
        )
        
        process_parser.add_argument(
            '--dry-run',
            action='store_true',
            help='试运行模式，不执行实际处理'
        )
        
        process_parser.add_argument(
            '--profile',
            action='store_true',
            help='启用性能分析'
        )
        
        process_parser.add_argument(
            '--memory-monitor',
            action='store_true',
            help='启用内存监控'
        )
        
        process_parser.add_argument(
            '--parallel',
            type=int,
            metavar='N',
            help='并行处理线程数'
        )
        
        process_parser.add_argument(
            '--resume',
            help='从指定检查点恢复处理'
        )
    
    def _add_config_command(self, subparsers):
        """添加config子命令"""
        config_parser = subparsers.add_parser(
            'config',
            help='配置管理操作',
            description='管理管线配置文件'
        )
        
        config_subparsers = config_parser.add_subparsers(
            dest='config_action',
            help='配置操作',
            metavar='ACTION'
        )
        
        # create 配置
        create_parser = config_subparsers.add_parser(
            'create',
            help='创建默认配置文件'
        )
        create_parser.add_argument(
            '--output', '-o',
            default='config/pipeline_config.json',
            help='输出配置文件路径'
        )
        
        # validate 配置
        validate_parser = config_subparsers.add_parser(
            'validate',
            help='验证配置文件'
        )
        validate_parser.add_argument(
            'config_file',
            help='要验证的配置文件路径'
        )
        
        # show 配置
        show_parser = config_subparsers.add_parser(
            'show',
            help='显示配置内容'
        )
        show_parser.add_argument(
            'config_file',
            nargs='?',
            help='配置文件路径（可选）'
        )
        show_parser.add_argument(
            '--section',
            help='只显示指定配置节'
        )
        
        # update 配置
        update_parser = config_subparsers.add_parser(
            'update',
            help='更新配置值'
        )
        update_parser.add_argument(
            'config_file',
            help='配置文件路径'
        )
        update_parser.add_argument(
            'key',
            help='配置键（支持点分隔）'
        )
        update_parser.add_argument(
            'value',
            help='配置值（JSON格式）'
        )
    
    def _add_batch_command(self, subparsers):
        """添加batch子命令"""
        batch_parser = subparsers.add_parser(
            'batch',
            help='批处理多个数据集',
            description='批量处理多个数据集目录'
        )
        
        batch_parser.add_argument(
            '--input-pattern',
            required=True,
            help='输入目录模式（支持通配符）'
        )
        
        batch_parser.add_argument(
            '--output-base',
            required=True,
            help='输出基础目录'
        )
        
        batch_parser.add_argument(
            '--max-parallel',
            type=int,
            default=2,
            help='最大并行处理数'
        )
        
        batch_parser.add_argument(
            '--continue-on-error',
            action='store_true',
            help='遇到错误时继续处理其他数据集'
        )
    
    def _add_analyze_command(self, subparsers):
        """添加analyze子命令"""
        analyze_parser = subparsers.add_parser(
            'analyze',
            help='分析已生成的结果',
            description='对已生成的网络分析结果进行进一步分析'
        )
        
        analyze_parser.add_argument(
            'result_dir',
            help='结果目录路径'
        )
        
        analyze_parser.add_argument(
            '--comparison',
            action='store_true',
            help='生成对比分析报告'
        )
        
        analyze_parser.add_argument(
            '--metrics',
            nargs='+',
            help='指定要计算的网络指标'
        )
        
        analyze_parser.add_argument(
            '--export-format',
            choices=['json', 'csv', 'xlsx'],
            default='json',
            help='导出格式'
        )
    
    def _add_performance_command(self, subparsers):
        """添加performance子命令"""
        perf_parser = subparsers.add_parser(
            'performance',
            help='性能分析和优化',
            description='分析管线性能并提供优化建议'
        )
        
        perf_parser.add_argument(
            'input_dir',
            help='测试数据目录'
        )
        
        perf_parser.add_argument(
            '--benchmark',
            action='store_true',
            help='运行性能基准测试'
        )
        
        perf_parser.add_argument(
            '--profile-components',
            nargs='+',
            help='指定要分析的组件'
        )
        
        perf_parser.add_argument(
            '--memory-profile',
            action='store_true',
            help='启用内存使用分析'
        )
        
        perf_parser.add_argument(
            '--optimization-report',
            action='store_true',
            help='生成优化建议报告'
        )
    
    def run(self, args=None) -> int:
        """运行CLI"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        # 设置日志级别
        log_level = self._determine_log_level(parsed_args)
        
        try:
            # 初始化基础组件
            self._initialize_components(parsed_args, log_level)
            
            # 执行命令
            if parsed_args.command == 'process':
                return self._handle_process_command(parsed_args)
            elif parsed_args.command == 'config':
                return self._handle_config_command(parsed_args)
            elif parsed_args.command == 'batch':
                return self._handle_batch_command(parsed_args)
            elif parsed_args.command == 'analyze':
                return self._handle_analyze_command(parsed_args)
            elif parsed_args.command == 'performance':
                return self._handle_performance_command(parsed_args)
            else:
                parser.print_help()
                return 1
                
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info("操作被用户中断")
            print("\n操作被用户中断")
            return 1
            
        except Exception as e:
            if self.error_handler:
                self.error_handler.handle_error(e, "cli_execution")
            print(f"错误: {e}")
            return 1
    
    def _determine_log_level(self, args) -> str:
        """确定日志级别"""
        if getattr(args, 'quiet', False):
            return 'ERROR'
        
        verbose_level = getattr(args, 'verbose', 0)
        if verbose_level >= 3:
            return 'DEBUG'
        elif verbose_level >= 2:
            return 'INFO'
        elif verbose_level >= 1:
            return 'INFO'
        else:
            return 'WARNING'
    
    def _initialize_components(self, args, log_level: str) -> None:
        """初始化基础组件"""
        # 创建临时配置用于日志初始化
        temp_config = Config()
        temp_config.set('logging.level', log_level)
        
        # 初始化日志器
        self.logger = PipelineLogger("CLI", temp_config.get_section('logging'))
        
        # 初始化错误处理器
        self.error_handler = ErrorHandler(temp_config.get_section('error_handling'))
        
        self.logger.info("CLI组件初始化完成")
    
    def _handle_process_command(self, args) -> int:
        """处理process命令"""
        if not self.logger:
            # 如果logger未初始化，创建临时logger
            from .core.logger import PipelineLogger
            from .core.config import Config
            temp_config = Config()
            self.logger = PipelineLogger("CLI", temp_config.get_section('logging'))
        
        self.logger.info(f"开始处理: {args.input_dir} -> {args.output_dir}")
        
        if args.dry_run:
            self.logger.info("试运行模式 - 不执行实际处理")
            return self._dry_run_process(args)
        
        try:
            # 创建管线实例
            pipeline = SemanticCowordPipeline(args.config)
            
            # 应用命令行参数覆盖
            self._apply_process_overrides(pipeline, args)
            
            # 运行处理
            result = pipeline.run(args.input_dir, args.output_dir)
            
            # 输出结果摘要
            self._print_process_summary(result)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"处理失败: {e}")
            return 1
    
    def _handle_config_command(self, args) -> int:
        """处理config命令"""
        if args.config_action == 'create':
            return self._create_config(args)
        elif args.config_action == 'validate':
            return self._validate_config(args)
        elif args.config_action == 'show':
            return self._show_config(args)
        elif args.config_action == 'update':
            return self._update_config(args)
        else:
            print("请指定配置操作: create, validate, show, update")
            return 1
    
    def _handle_batch_command(self, args) -> int:
        """处理batch命令"""
        self.logger.info(f"开始批处理: {args.input_pattern}")
        
        try:
            from glob import glob
            import concurrent.futures
            
            # 查找匹配的输入目录
            input_dirs = glob(args.input_pattern)
            if not input_dirs:
                print(f"未找到匹配的输入目录: {args.input_pattern}")
                return 1
            
            self.logger.info(f"找到 {len(input_dirs)} 个输入目录")
            
            # 批处理结果
            results = []
            failed_dirs = []
            
            # 并行处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
                future_to_dir = {}
                
                for input_dir in input_dirs:
                    # 生成输出目录名
                    dir_name = Path(input_dir).name
                    output_dir = Path(args.output_base) / dir_name
                    
                    # 提交处理任务
                    future = executor.submit(self._process_single_directory, input_dir, str(output_dir))
                    future_to_dir[future] = input_dir
                
                # 收集结果
                for future in concurrent.futures.as_completed(future_to_dir):
                    input_dir = future_to_dir[future]
                    try:
                        result = future.result()
                        results.append((input_dir, result))
                        self.logger.info(f"完成处理: {input_dir}")
                    except Exception as e:
                        failed_dirs.append((input_dir, str(e)))
                        self.logger.error(f"处理失败 {input_dir}: {e}")
                        
                        if not args.continue_on_error:
                            break
            
            # 输出批处理摘要
            self._print_batch_summary(results, failed_dirs)
            
            return 0 if not failed_dirs else 1
            
        except Exception as e:
            self.logger.error(f"批处理失败: {e}")
            return 1
    
    def _handle_analyze_command(self, args) -> int:
        """处理analyze命令"""
        self.logger.info(f"分析结果目录: {args.result_dir}")
        
        try:
            # 这里可以实现结果分析功能
            # 暂时返回成功状态
            print(f"分析功能开发中 - 结果目录: {args.result_dir}")
            return 0
            
        except Exception as e:
            self.logger.error(f"分析失败: {e}")
            return 1
    
    def _handle_performance_command(self, args) -> int:
        """处理performance命令"""
        self.logger.info(f"性能分析: {args.input_dir}")
        
        try:
            # 这里可以实现性能分析功能
            # 暂时返回成功状态
            print(f"性能分析功能开发中 - 输入目录: {args.input_dir}")
            return 0
            
        except Exception as e:
            self.logger.error(f"性能分析失败: {e}")
            return 1
    
    def _dry_run_process(self, args) -> int:
        """试运行处理"""
        print(f"试运行模式:")
        print(f"  输入目录: {args.input_dir}")
        print(f"  输出目录: {args.output_dir}")
        print(f"  配置文件: {args.config or '默认配置'}")
        
        # 验证输入目录
        if not Path(args.input_dir).exists():
            print(f"  错误: 输入目录不存在")
            return 1
        
        # 检查输入文件
        input_files = list(Path(args.input_dir).rglob("*.json"))
        print(f"  找到 {len(input_files)} 个JSON文件")
        
        # 检查输出目录
        output_path = Path(args.output_dir)
        if output_path.exists():
            print(f"  输出目录已存在")
        else:
            print(f"  将创建输出目录")
        
        print("试运行完成 - 未执行实际处理")
        return 0
    
    def _apply_process_overrides(self, pipeline, args) -> None:
        """应用命令行参数覆盖"""
        config_updates = {}
        
        if args.parallel:
            config_updates['performance.max_workers'] = args.parallel
        
        if args.profile:
            config_updates['performance.enable_profiling'] = True
        
        if args.memory_monitor:
            config_updates['performance.enable_memory_monitoring'] = True
        
        if config_updates:
            pipeline.update_configuration(config_updates)
    
    def _create_config(self, args) -> int:
        """创建配置文件"""
        try:
            # 确保输出目录存在
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建默认配置
            config_file = create_default_config()
            
            # 如果指定了不同的输出路径，复制配置
            if str(output_path) != config_file:
                import shutil
                shutil.copy2(config_file, output_path)
                config_file = str(output_path)
            
            print(f"已创建默认配置文件: {config_file}")
            return 0
            
        except Exception as e:
            print(f"创建配置文件失败: {e}")
            return 1
    
    def _validate_config(self, args) -> int:
        """验证配置文件"""
        try:
            config = Config(args.config_file)
            validation_result = config.validate()
            
            if validation_result['errors']:
                print("配置验证失败:")
                for error in validation_result['errors']:
                    print(f"  错误: {error}")
                return 1
            
            if validation_result['warnings']:
                print("配置验证警告:")
                for warning in validation_result['warnings']:
                    print(f"  警告: {warning}")
            
            print("配置验证通过!")
            return 0
            
        except Exception as e:
            print(f"配置验证失败: {e}")
            return 1
    
    def _show_config(self, args) -> int:
        """显示配置内容"""
        try:
            config = Config(args.config_file)
            
            if args.section:
                section_config = config.get_section(args.section)
                if section_config:
                    print(json.dumps(section_config, indent=2, ensure_ascii=False))
                else:
                    print(f"配置节不存在: {args.section}")
                    return 1
            else:
                print(json.dumps(config.get_all(), indent=2, ensure_ascii=False))
            
            return 0
            
        except Exception as e:
            print(f"显示配置失败: {e}")
            return 1
    
    def _update_config(self, args) -> int:
        """更新配置值"""
        try:
            config = Config(args.config_file)
            
            # 解析JSON值
            try:
                value = json.loads(args.value)
            except json.JSONDecodeError:
                # 如果不是JSON，作为字符串处理
                value = args.value
            
            # 更新配置
            config.set(args.key, value)
            config.save_to_file(args.config_file)
            
            print(f"已更新配置: {args.key} = {value}")
            return 0
            
        except Exception as e:
            print(f"更新配置失败: {e}")
            return 1
    
    def _process_single_directory(self, input_dir: str, output_dir: str):
        """处理单个目录"""
        pipeline = SemanticCowordPipeline()
        return pipeline.run(input_dir, output_dir)
    
    def _print_process_summary(self, result) -> None:
        """打印处理摘要"""
        print(f"\n处理完成!")
        print(f"  总文件数: {result.total_files}")
        print(f"  处理成功: {result.processed_files}")
        print(f"  处理失败: {result.failed_files}")
        print(f"  处理时间: {result.processing_time:.2f} 秒")
        print(f"  输出文件: {len(result.output_files)} 个")
        
        if result.global_graph:
            print(f"  全局图节点数: {result.global_graph.get_node_count()}")
        
        print(f"  州级子图数: {len(result.state_subgraphs)}")
        
        if result.error_summary.get('summary', {}).get('total_errors', 0) > 0:
            print(f"  错误数量: {result.error_summary['summary']['total_errors']}")
            print("  详细错误信息请查看输出目录中的错误报告")
    
    def _print_batch_summary(self, results, failed_dirs) -> None:
        """打印批处理摘要"""
        print(f"\n批处理完成!")
        print(f"  成功处理: {len(results)} 个目录")
        print(f"  处理失败: {len(failed_dirs)} 个目录")
        
        if failed_dirs:
            print("\n失败的目录:")
            for dir_path, error in failed_dirs:
                print(f"  {dir_path}: {error}")


def main(args=None) -> int:
    """主入口函数"""
    cli = CLIInterface()
    return cli.run(args)


if __name__ == '__main__':
    sys.exit(main())