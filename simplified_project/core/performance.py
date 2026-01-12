"""
性能监控和优化模块

提供性能监控、内存使用分析和优化建议功能。
根据需求7.1-7.6和性能相关需求实现。
"""

import time
import psutil
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import functools
import tracemalloc


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_start: float
    memory_end: float
    memory_peak: float
    cpu_percent: float
    thread_count: int
    metadata: Dict[str, Any]


@dataclass
class SystemResources:
    """系统资源使用情况"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    thread_count: int
    process_count: int


class PerformanceMonitor:
    """
    性能监控器
    
    监控系统资源使用、操作耗时和内存分配。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化性能监控器
        
        Args:
            config: 性能监控配置
        """
        self.config = config
        self.enabled = config.get('enable_profiling', False)
        self.memory_monitoring = config.get('enable_memory_monitoring', False)
        self.sampling_interval = config.get('sampling_interval', 1.0)
        
        # 性能数据存储
        self.metrics: List[PerformanceMetrics] = []
        self.system_resources: List[SystemResources] = []
        self.operation_stack: List[Dict[str, Any]] = []
        
        # 监控线程
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        
        # 内存追踪
        self.memory_snapshots: Dict[str, Any] = {}
        
        # 进程对象
        self.process = psutil.Process()
        
        if self.memory_monitoring:
            tracemalloc.start()
    
    def start_monitoring(self) -> None:
        """开始系统资源监控"""
        if not self.enabled or self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_system_resources,
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self) -> None:
        """停止系统资源监控"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)
    
    def start_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        开始监控操作
        
        Args:
            operation: 操作名称
            metadata: 操作元数据
            
        Returns:
            操作ID
        """
        if not self.enabled:
            return ""
        
        operation_id = f"{operation}_{len(self.operation_stack)}"
        
        # 获取当前系统状态
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        
        operation_data = {
            'id': operation_id,
            'operation': operation,
            'start_time': time.time(),
            'memory_start': memory_info.rss / 1024 / 1024,  # MB
            'cpu_start': cpu_percent,
            'thread_count': threading.active_count(),
            'metadata': metadata or {}
        }
        
        # 内存快照
        if self.memory_monitoring:
            snapshot = tracemalloc.take_snapshot()
            self.memory_snapshots[operation_id] = snapshot
        
        self.operation_stack.append(operation_data)
        return operation_id
    
    def end_operation(self, operation_id: str) -> Optional[PerformanceMetrics]:
        """
        结束监控操作
        
        Args:
            operation_id: 操作ID
            
        Returns:
            性能指标
        """
        if not self.enabled or not operation_id:
            return None
        
        # 查找操作数据
        operation_data = None
        for i, op in enumerate(self.operation_stack):
            if op['id'] == operation_id:
                operation_data = self.operation_stack.pop(i)
                break
        
        if not operation_data:
            return None
        
        # 计算性能指标
        end_time = time.time()
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        
        # 获取内存峰值
        memory_peak = operation_data['memory_start']
        if self.memory_monitoring and operation_id in self.memory_snapshots:
            current_snapshot = tracemalloc.take_snapshot()
            # 这里可以计算内存使用差异
            memory_peak = max(memory_peak, memory_info.rss / 1024 / 1024)
        
        metrics = PerformanceMetrics(
            operation=operation_data['operation'],
            start_time=operation_data['start_time'],
            end_time=end_time,
            duration=end_time - operation_data['start_time'],
            memory_start=operation_data['memory_start'],
            memory_end=memory_info.rss / 1024 / 1024,
            memory_peak=memory_peak,
            cpu_percent=cpu_percent,
            thread_count=threading.active_count(),
            metadata=operation_data['metadata']
        )
        
        self.metrics.append(metrics)
        
        # 清理内存快照
        if operation_id in self.memory_snapshots:
            del self.memory_snapshots[operation_id]
        
        return metrics
    
    def _monitor_system_resources(self) -> None:
        """监控系统资源使用情况"""
        while self.monitoring_active:
            try:
                # 获取系统资源信息
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # 获取进程信息
                process_count = len(psutil.pids())
                thread_count = threading.active_count()
                
                resource_data = SystemResources(
                    timestamp=time.time(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / 1024 / 1024,
                    memory_available_mb=memory.available / 1024 / 1024,
                    disk_usage_percent=disk.percent,
                    thread_count=thread_count,
                    process_count=process_count
                )
                
                self.system_resources.append(resource_data)
                
                # 限制数据量
                if len(self.system_resources) > 10000:
                    self.system_resources = self.system_resources[-5000:]
                
                time.sleep(self.sampling_interval)
                
            except Exception:
                # 忽略监控错误，避免影响主程序
                time.sleep(self.sampling_interval)
    
    def get_operation_summary(self) -> Dict[str, Any]:
        """获取操作性能摘要"""
        if not self.metrics:
            return {}
        
        # 按操作类型分组
        operations = defaultdict(list)
        for metric in self.metrics:
            operations[metric.operation].append(metric)
        
        summary = {}
        for operation, metrics_list in operations.items():
            durations = [m.duration for m in metrics_list]
            memory_usage = [m.memory_end - m.memory_start for m in metrics_list]
            
            summary[operation] = {
                'count': len(metrics_list),
                'total_duration': sum(durations),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'avg_memory_usage': sum(memory_usage) / len(memory_usage),
                'max_memory_usage': max(memory_usage),
                'total_memory_usage': sum(memory_usage)
            }
        
        return summary
    
    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统资源摘要"""
        if not self.system_resources:
            return {}
        
        cpu_values = [r.cpu_percent for r in self.system_resources]
        memory_values = [r.memory_percent for r in self.system_resources]
        
        return {
            'monitoring_duration': len(self.system_resources) * self.sampling_interval,
            'cpu_usage': {
                'avg': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'memory_usage': {
                'avg': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values)
            },
            'peak_memory_mb': max(r.memory_used_mb for r in self.system_resources),
            'peak_thread_count': max(r.thread_count for r in self.system_resources)
        }
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        return {
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config,
            'operation_summary': self.get_operation_summary(),
            'system_summary': self.get_system_summary(),
            'detailed_metrics': [asdict(m) for m in self.metrics],
            'optimization_recommendations': self._generate_optimization_recommendations()
        }
    
    def save_performance_report(self, output_path: str) -> None:
        """保存性能报告到文件"""
        report = self.generate_performance_report()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def _generate_optimization_recommendations(self) -> List[Dict[str, str]]:
        """生成优化建议"""
        recommendations = []
        
        # 分析操作性能
        operation_summary = self.get_operation_summary()
        for operation, stats in operation_summary.items():
            if stats['avg_duration'] > 10.0:  # 超过10秒
                recommendations.append({
                    'type': 'performance',
                    'component': operation,
                    'issue': f"操作 {operation} 平均耗时 {stats['avg_duration']:.2f} 秒",
                    'recommendation': "考虑优化算法或启用并行处理"
                })
            
            if stats['max_memory_usage'] > 1000:  # 超过1GB
                recommendations.append({
                    'type': 'memory',
                    'component': operation,
                    'issue': f"操作 {operation} 最大内存使用 {stats['max_memory_usage']:.2f} MB",
                    'recommendation': "考虑分批处理或优化数据结构"
                })
        
        # 分析系统资源
        system_summary = self.get_system_summary()
        if system_summary:
            if system_summary['cpu_usage']['avg'] > 80:
                recommendations.append({
                    'type': 'cpu',
                    'component': 'system',
                    'issue': f"平均CPU使用率 {system_summary['cpu_usage']['avg']:.1f}%",
                    'recommendation': "考虑减少并行度或优化计算密集型操作"
                })
            
            if system_summary['memory_usage']['avg'] > 80:
                recommendations.append({
                    'type': 'memory',
                    'component': 'system',
                    'issue': f"平均内存使用率 {system_summary['memory_usage']['avg']:.1f}%",
                    'recommendation': "考虑增加内存或优化内存使用"
                })
        
        return recommendations
    
    def clear_metrics(self) -> None:
        """清空性能指标"""
        self.metrics.clear()
        self.system_resources.clear()
        self.operation_stack.clear()
        self.memory_snapshots.clear()


def performance_monitor(operation_name: str, monitor: Optional[PerformanceMonitor] = None):
    """
    性能监控装饰器
    
    Args:
        operation_name: 操作名称
        monitor: 性能监控器实例
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if monitor and monitor.enabled:
                operation_id = monitor.start_operation(operation_name, {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                })
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    monitor.end_operation(operation_id)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class MemoryProfiler:
    """
    内存使用分析器
    
    提供详细的内存使用分析和泄漏检测。
    """
    
    def __init__(self):
        """初始化内存分析器"""
        self.snapshots: Dict[str, Any] = {}
        self.enabled = False
    
    def start_profiling(self) -> None:
        """开始内存分析"""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self.enabled = True
    
    def stop_profiling(self) -> None:
        """停止内存分析"""
        self.enabled = False
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    
    def take_snapshot(self, name: str) -> None:
        """获取内存快照"""
        if not self.enabled:
            return
        
        snapshot = tracemalloc.take_snapshot()
        self.snapshots[name] = snapshot
    
    def compare_snapshots(self, name1: str, name2: str) -> Dict[str, Any]:
        """比较两个内存快照"""
        if name1 not in self.snapshots or name2 not in self.snapshots:
            return {}
        
        snapshot1 = self.snapshots[name1]
        snapshot2 = self.snapshots[name2]
        
        # 计算差异
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        # 分析结果
        analysis = {
            'total_difference': sum(stat.size_diff for stat in top_stats),
            'top_differences': [],
            'memory_leaks': []
        }
        
        # 获取前10个最大差异
        for stat in top_stats[:10]:
            analysis['top_differences'].append({
                'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'size_diff': stat.size_diff,
                'count_diff': stat.count_diff
            })
        
        # 检测可能的内存泄漏
        for stat in top_stats:
            if stat.size_diff > 1024 * 1024:  # 超过1MB增长
                analysis['memory_leaks'].append({
                    'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size_diff': stat.size_diff,
                    'potential_leak': True
                })
        
        return analysis
    
    def get_current_memory_usage(self) -> Dict[str, Any]:
        """获取当前内存使用情况"""
        if not self.enabled:
            return {}
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        return {
            'total_size': sum(stat.size for stat in top_stats),
            'total_count': sum(stat.count for stat in top_stats),
            'top_allocations': [
                {
                    'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size': stat.size,
                    'count': stat.count
                }
                for stat in top_stats[:10]
            ]
        }


class OptimizationAnalyzer:
    """
    优化分析器
    
    分析性能数据并提供优化建议。
    """
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        """
        初始化优化分析器
        
        Args:
            performance_monitor: 性能监控器
        """
        self.monitor = performance_monitor
    
    def analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """分析性能瓶颈"""
        bottlenecks = []
        
        operation_summary = self.monitor.get_operation_summary()
        
        # 分析耗时操作
        for operation, stats in operation_summary.items():
            if stats['max_duration'] > stats['avg_duration'] * 3:
                bottlenecks.append({
                    'type': 'time_variance',
                    'operation': operation,
                    'severity': 'high' if stats['max_duration'] > 60 else 'medium',
                    'description': f"操作 {operation} 存在显著的时间差异",
                    'details': {
                        'avg_duration': stats['avg_duration'],
                        'max_duration': stats['max_duration'],
                        'variance_ratio': stats['max_duration'] / stats['avg_duration']
                    }
                })
            
            if stats['avg_duration'] > 30:  # 超过30秒
                bottlenecks.append({
                    'type': 'slow_operation',
                    'operation': operation,
                    'severity': 'high',
                    'description': f"操作 {operation} 执行时间过长",
                    'details': {
                        'avg_duration': stats['avg_duration'],
                        'count': stats['count']
                    }
                })
        
        return bottlenecks
    
    def suggest_optimizations(self) -> List[Dict[str, str]]:
        """提供优化建议"""
        suggestions = []
        
        bottlenecks = self.analyze_bottlenecks()
        operation_summary = self.monitor.get_operation_summary()
        
        for bottleneck in bottlenecks:
            operation = bottleneck['operation']
            
            if bottleneck['type'] == 'slow_operation':
                if 'text_processing' in operation:
                    suggestions.append({
                        'component': operation,
                        'optimization': '并行文本处理',
                        'description': '考虑使用多进程处理大量文本文档',
                        'implementation': '启用 performance.enable_parallel_processing'
                    })
                
                elif 'graph_construction' in operation:
                    suggestions.append({
                        'component': operation,
                        'optimization': '稀疏矩阵优化',
                        'description': '使用稀疏矩阵存储共现关系以节省内存',
                        'implementation': '启用 graph_construction.use_sparse_matrix'
                    })
                
                elif 'layout' in operation:
                    suggestions.append({
                        'component': operation,
                        'optimization': '布局算法优化',
                        'description': '减少迭代次数或使用更快的布局算法',
                        'implementation': '调整 layout_engine.max_iterations'
                    })
        
        # 内存优化建议
        for operation, stats in operation_summary.items():
            if stats['max_memory_usage'] > 2000:  # 超过2GB
                suggestions.append({
                    'component': operation,
                    'optimization': '内存使用优化',
                    'description': '考虑分批处理或使用内存映射文件',
                    'implementation': '调整 performance.batch_size 或启用压缩'
                })
        
        return suggestions