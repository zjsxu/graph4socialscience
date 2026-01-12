"""
错误处理系统

提供统一的错误处理、恢复机制和错误报告功能。
根据设计文档中的错误处理策略实现。
"""

import traceback
import logging
from typing import Dict, Any, Optional, List, Callable, Type
from datetime import datetime
from enum import Enum
import json


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """错误类别"""
    INPUT_VALIDATION = "input_validation"
    PROCESSING = "processing"
    GRAPH_CONSTRUCTION = "graph_construction"
    OUTPUT = "output"
    CONFIGURATION = "configuration"
    SYSTEM = "system"


class PipelineError(Exception):
    """管线基础异常类"""
    
    def __init__(self, message: str, category: ErrorCategory, severity: ErrorSeverity, 
                 context: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'context': self.context,
            'timestamp': self.timestamp,
            'original_error': str(self.original_error) if self.original_error else None
        }


class InputValidationError(PipelineError):
    """输入验证错误"""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None, 
                 original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.INPUT_VALIDATION, ErrorSeverity.HIGH, context, original_error)


class ProcessingError(PipelineError):
    """处理过程错误"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.PROCESSING, severity, context, original_error)


class GraphConstructionError(PipelineError):
    """图构建错误"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.HIGH,
                 context: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.GRAPH_CONSTRUCTION, severity, context, original_error)


class OutputError(PipelineError):
    """输出错误"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(message, ErrorCategory.OUTPUT, severity, context, original_error)


class ErrorHandler:
    """
    错误处理器
    
    提供统一的错误处理、恢复机制和错误报告功能。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化错误处理器
        
        Args:
            config: 错误处理配置
        """
        self.config = config or {}
        self.fallback_strategies = self.config.get('fallback_strategies', {})
        self.error_log: List[Dict[str, Any]] = []
        self.recovery_actions: List[Dict[str, Any]] = []
        
        # 设置日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 注册默认的回退策略
        self._register_default_fallback_strategies()
    
    def handle_error(self, error: Exception, context: str, 
                    allow_fallback: bool = True) -> Any:
        """
        处理错误并尝试恢复
        
        Args:
            error: 异常对象
            context: 错误上下文
            allow_fallback: 是否允许使用回退策略
            
        Returns:
            恢复结果或None
        """
        # 记录错误
        error_info = self._create_error_info(error, context)
        self.error_log.append(error_info)
        
        # 记录到日志
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
        
        # 尝试恢复
        if allow_fallback:
            recovery_result = self._attempt_recovery(error, context)
            if recovery_result is not None:
                self.logger.info(f"Successfully recovered from error in {context}")
                return recovery_result
        
        # 如果无法恢复，重新抛出异常
        if isinstance(error, PipelineError):
            raise error
        else:
            # 包装为管线异常
            raise ProcessingError(
                f"Unhandled error in {context}: {str(error)}",
                context={'original_context': context},
                original_error=error
            )
    
    def handle_input_validation_error(self, error: Exception, input_data: Any, 
                                    field_name: Optional[str] = None) -> None:
        """
        处理输入验证错误
        
        Args:
            error: 异常对象
            input_data: 输入数据
            field_name: 字段名称
        """
        context = {
            'input_type': type(input_data).__name__,
            'field_name': field_name,
            'input_sample': str(input_data)[:200] if input_data else None
        }
        
        raise InputValidationError(
            f"Input validation failed: {str(error)}",
            context=context,
            original_error=error
        )
    
    def handle_processing_error(self, error: Exception, operation: str, 
                              input_data: Any = None, severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Any:
        """
        处理过程错误
        
        Args:
            error: 异常对象
            operation: 操作名称
            input_data: 输入数据
            severity: 错误严重程度
            
        Returns:
            恢复结果或None
        """
        context = {
            'operation': operation,
            'input_type': type(input_data).__name__ if input_data else None
        }
        
        # 尝试使用回退策略
        fallback_key = f"processing.{operation}"
        if fallback_key in self.fallback_strategies:
            try:
                recovery_result = self.fallback_strategies[fallback_key](error, input_data)
                self._record_recovery_action(operation, "fallback_strategy", recovery_result)
                return recovery_result
            except Exception as fallback_error:
                self.logger.warning(f"Fallback strategy failed for {operation}: {fallback_error}")
        
        raise ProcessingError(
            f"Processing error in {operation}: {str(error)}",
            severity=severity,
            context=context,
            original_error=error
        )
    
    def handle_graph_construction_error(self, error: Exception, graph_type: str, 
                                      node_count: int = 0, edge_count: int = 0) -> Any:
        """
        处理图构建错误
        
        Args:
            error: 异常对象
            graph_type: 图类型
            node_count: 节点数量
            edge_count: 边数量
        """
        context = {
            'graph_type': graph_type,
            'node_count': node_count,
            'edge_count': edge_count
        }
        
        # 尝试创建最小图作为回退
        if node_count == 0:
            self.logger.warning("Attempting to create minimal graph with isolated nodes")
            # 这里可以实现创建最小图的逻辑
            # 返回包含孤立节点的最小图
        
        raise GraphConstructionError(
            f"Graph construction failed for {graph_type}: {str(error)}",
            context=context,
            original_error=error
        )
    
    def handle_output_error(self, error: Exception, output_type: str, 
                          file_path: Optional[str] = None) -> None:
        """
        处理输出错误
        
        Args:
            error: 异常对象
            output_type: 输出类型
            file_path: 文件路径
        """
        context = {
            'output_type': output_type,
            'file_path': file_path
        }
        
        # 尝试使用替代输出路径
        if file_path and "permission" in str(error).lower():
            alternative_path = f"/tmp/{output_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.logger.warning(f"Attempting alternative output path: {alternative_path}")
            context['alternative_path'] = alternative_path
        
        raise OutputError(
            f"Output error for {output_type}: {str(error)}",
            context=context,
            original_error=error
        )
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """
        记录警告信息
        
        Args:
            message: 警告消息
            context: 上下文信息
        """
        warning_info = {
            'timestamp': datetime.now().isoformat(),
            'level': 'WARNING',
            'message': message,
            'context': context or {}
        }
        
        self.error_log.append(warning_info)
        self.logger.warning(message)
    
    def generate_error_report(self) -> Dict[str, Any]:
        """
        生成错误报告
        
        Returns:
            包含错误统计和详细信息的报告
        """
        # 统计错误类别和严重程度
        category_counts = {}
        severity_counts = {}
        
        for error_info in self.error_log:
            if 'category' in error_info:
                category = error_info['category']
                category_counts[category] = category_counts.get(category, 0) + 1
            
            if 'severity' in error_info:
                severity = error_info['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'summary': {
                'total_errors': len(self.error_log),
                'total_recoveries': len(self.recovery_actions),
                'category_distribution': category_counts,
                'severity_distribution': severity_counts
            },
            'errors': self.error_log,
            'recovery_actions': self.recovery_actions,
            'generated_at': datetime.now().isoformat()
        }
    
    def save_error_report(self, file_path: str) -> None:
        """
        保存错误报告到文件
        
        Args:
            file_path: 文件路径
        """
        report = self.generate_error_report()
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save error report: {e}")
    
    def clear_error_log(self) -> None:
        """清空错误日志"""
        self.error_log.clear()
        self.recovery_actions.clear()
    
    def _create_error_info(self, error: Exception, context: str) -> Dict[str, Any]:
        """创建错误信息字典"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'error_type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc()
        }
        
        # 如果是管线异常，添加额外信息
        if isinstance(error, PipelineError):
            error_info.update(error.to_dict())
        
        return error_info
    
    def _attempt_recovery(self, error: Exception, context: str) -> Any:
        """尝试错误恢复"""
        # 查找适用的回退策略
        for strategy_key, strategy_func in self.fallback_strategies.items():
            if context.startswith(strategy_key) or strategy_key == "default":
                try:
                    result = strategy_func(error, context)
                    self._record_recovery_action(context, strategy_key, result)
                    return result
                except Exception as recovery_error:
                    self.logger.warning(f"Recovery strategy {strategy_key} failed: {recovery_error}")
        
        return None
    
    def _record_recovery_action(self, context: str, strategy: str, result: Any) -> None:
        """记录恢复动作"""
        recovery_info = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'strategy': strategy,
            'result_type': type(result).__name__ if result is not None else None,
            'success': result is not None
        }
        
        self.recovery_actions.append(recovery_info)
    
    def _register_default_fallback_strategies(self) -> None:
        """注册默认的回退策略"""
        
        def text_processing_fallback(error: Exception, input_data: Any) -> Any:
            """文本处理回退策略：使用简单空格分割"""
            if isinstance(input_data, str):
                return input_data.split()
            return []
        
        def phrase_extraction_fallback(error: Exception, input_data: Any) -> Any:
            """词组抽取回退策略：回退到单词级别"""
            if isinstance(input_data, list):
                return input_data  # 返回原始词列表
            return []
        
        def layout_computation_fallback(error: Exception, input_data: Any) -> Any:
            """布局计算回退策略：使用随机布局"""
            import random
            if hasattr(input_data, 'nodes'):
                positions = {}
                for node in input_data.nodes():
                    positions[node] = (random.random(), random.random())
                return positions
            return {}
        
        # 注册回退策略
        self.fallback_strategies.update({
            'text_processing': text_processing_fallback,
            'phrase_extraction': phrase_extraction_fallback,
            'layout_computation': layout_computation_fallback
        })
    
    def register_fallback_strategy(self, key: str, strategy_func: Callable) -> None:
        """
        注册自定义回退策略
        
        Args:
            key: 策略键
            strategy_func: 策略函数
        """
        self.fallback_strategies[key] = strategy_func
    
    def get_error_count(self, category: Optional[ErrorCategory] = None, 
                       severity: Optional[ErrorSeverity] = None) -> int:
        """
        获取错误数量
        
        Args:
            category: 错误类别过滤
            severity: 严重程度过滤
            
        Returns:
            符合条件的错误数量
        """
        count = 0
        for error_info in self.error_log:
            if category and error_info.get('category') != category.value:
                continue
            if severity and error_info.get('severity') != severity.value:
                continue
            count += 1
        
        return count