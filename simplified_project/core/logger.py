"""
日志系统

提供统一的日志记录功能，支持文件和控制台输出。
根据需求10.5提供可追溯的处理过程记录。
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class PipelineLogger:
    """
    管线日志记录器
    
    提供结构化的日志记录功能，支持不同级别的日志输出。
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            config: 日志配置
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_logger()
    
    def _setup_logger(self) -> None:
        """设置日志记录器"""
        # 设置日志级别
        level = self.config.get('level', 'INFO')
        try:
            self.logger.setLevel(getattr(logging, level.upper()))
        except AttributeError:
            # 如果级别无效，使用INFO作为默认值
            self.logger.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter(
            self.config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 添加文件处理器
        file_path = self.config.get('file_path')
        if file_path:
            try:
                # 确保日志目录存在
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                
                # 使用轮转文件处理器
                max_bytes = self.config.get('max_file_size_mb', 100) * 1024 * 1024
                backup_count = self.config.get('backup_count', 5)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except (OSError, PermissionError) as e:
                # 如果文件处理器创建失败，只使用控制台处理器
                self.logger.warning(f"Failed to create file handler: {e}")
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录调试信息"""
        self._log_with_extra(logging.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录信息"""
        self._log_with_extra(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """记录警告"""
        self._log_with_extra(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """记录错误"""
        self._log_with_extra(logging.ERROR, message, extra, exc_info)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False) -> None:
        """记录严重错误"""
        self._log_with_extra(logging.CRITICAL, message, extra, exc_info)
    
    def log_processing_step(self, step_name: str, input_info: Dict[str, Any], 
                          output_info: Dict[str, Any], duration: Optional[float] = None) -> None:
        """
        记录处理步骤
        
        Args:
            step_name: 步骤名称
            input_info: 输入信息
            output_info: 输出信息
            duration: 处理时长（秒）
        """
        step_info = {
            'step': step_name,
            'input': input_info,
            'output': output_info,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        message = f"Processing step completed: {step_name}"
        if duration:
            message += f" (took {duration:.2f}s)"
        
        self.info(message, extra={'step_info': step_info})
    
    def log_configuration_change(self, config_key: str, old_value: Any, new_value: Any) -> None:
        """
        记录配置变更
        
        Args:
            config_key: 配置键
            old_value: 旧值
            new_value: 新值
        """
        change_info = {
            'config_key': config_key,
            'old_value': old_value,
            'new_value': new_value,
            'timestamp': datetime.now().isoformat()
        }
        
        self.info(f"Configuration changed: {config_key}", extra={'config_change': change_info})
    
    def log_performance_metrics(self, operation: str, metrics: Dict[str, Any]) -> None:
        """
        记录性能指标
        
        Args:
            operation: 操作名称
            metrics: 性能指标
        """
        perf_info = {
            'operation': operation,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.info(f"Performance metrics for {operation}", extra={'performance': perf_info})
    
    def log_data_statistics(self, data_type: str, statistics: Dict[str, Any]) -> None:
        """
        记录数据统计信息
        
        Args:
            data_type: 数据类型
            statistics: 统计信息
        """
        stats_info = {
            'data_type': data_type,
            'statistics': statistics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.info(f"Data statistics for {data_type}", extra={'data_stats': stats_info})
    
    def _log_with_extra(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None, 
                       exc_info: bool = False) -> None:
        """
        带额外信息的日志记录
        
        Args:
            level: 日志级别
            message: 消息
            extra: 额外信息
            exc_info: 是否包含异常信息
        """
        if extra:
            # 将额外信息添加到消息中
            extra_str = json.dumps(extra, ensure_ascii=False, default=str)
            full_message = f"{message} | Extra: {extra_str}"
        else:
            full_message = message
        
        self.logger.log(level, full_message, exc_info=exc_info)


class ProcessTracker:
    """
    处理过程追踪器
    
    用于追踪和记录整个处理过程的详细信息，支持需求10.5的可追溯性要求。
    """
    
    def __init__(self, logger: PipelineLogger):
        """
        初始化处理追踪器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger
        self.process_history: list = []
        self.current_process: Optional[Dict[str, Any]] = None
        self.start_time: Optional[datetime] = None
    
    def start_process(self, process_name: str, input_data: Dict[str, Any]) -> None:
        """
        开始新的处理过程
        
        Args:
            process_name: 过程名称
            input_data: 输入数据信息
        """
        self.start_time = datetime.now()
        self.current_process = {
            'process_name': process_name,
            'start_time': self.start_time.isoformat(),
            'input_data': input_data,
            'steps': [],
            'status': 'running'
        }
        
        self.logger.info(f"Started process: {process_name}", extra={'process_start': self.current_process})
    
    def add_step(self, step_name: str, input_info: Dict[str, Any], 
                output_info: Dict[str, Any], duration: Optional[float] = None) -> None:
        """
        添加处理步骤
        
        Args:
            step_name: 步骤名称
            input_info: 输入信息
            output_info: 输出信息
            duration: 处理时长
        """
        if not self.current_process:
            raise ValueError("No active process. Call start_process() first.")
        
        step_info = {
            'step_name': step_name,
            'timestamp': datetime.now().isoformat(),
            'input_info': input_info,
            'output_info': output_info,
            'duration_seconds': duration
        }
        
        self.current_process['steps'].append(step_info)
        self.logger.log_processing_step(step_name, input_info, output_info, duration)
    
    def end_process(self, output_data: Dict[str, Any], status: str = 'completed') -> None:
        """
        结束当前处理过程
        
        Args:
            output_data: 输出数据信息
            status: 处理状态
        """
        if not self.current_process:
            raise ValueError("No active process to end.")
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds() if self.start_time else None
        
        self.current_process.update({
            'end_time': end_time.isoformat(),
            'total_duration_seconds': total_duration,
            'output_data': output_data,
            'status': status
        })
        
        # 添加到历史记录
        self.process_history.append(self.current_process.copy())
        
        self.logger.info(
            f"Completed process: {self.current_process['process_name']} "
            f"(took {total_duration:.2f}s)" if total_duration else "",
            extra={'process_end': self.current_process}
        )
        
        # 清理当前过程
        self.current_process = None
        self.start_time = None
    
    def get_process_history(self) -> list:
        """获取处理历史"""
        import copy
        return copy.deepcopy(self.process_history)
    
    def save_process_history(self, file_path: str) -> None:
        """
        保存处理历史到文件
        
        Args:
            file_path: 文件路径
        """
        try:
            # 确保目录存在
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.process_history, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Process history saved to: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save process history: {e}", exc_info=True)


def setup_logger(name: str, config: Optional[Dict[str, Any]] = None) -> PipelineLogger:
    """
    设置日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称
        config: 日志配置
        
    Returns:
        配置好的日志记录器
    """
    return PipelineLogger(name, config)


def create_process_tracker(logger: PipelineLogger) -> ProcessTracker:
    """
    创建处理过程追踪器的便捷函数
    
    Args:
        logger: 日志记录器
        
    Returns:
        处理过程追踪器
    """
    return ProcessTracker(logger)


# 默认日志配置
DEFAULT_LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_path': 'logs/pipeline.log',
    'max_file_size_mb': 100,
    'backup_count': 5
}