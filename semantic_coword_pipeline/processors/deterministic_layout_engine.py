"""
确定性布局引擎

实现固定种子的力导向布局、节点位置缓存机制、层级和社群辅助布局以及可视化过滤功能。
根据需求1.1、1.3、1.4、1.5实现确定性、可复现的网络布局。
"""

import json
import os
import pickle
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import random
import math
from dataclasses import dataclass, field
from pathlib import Path

try:
    import easygraph as eg
except ImportError:
    eg = None

from ..core.data_models import GlobalGraph, StateSubgraph
from ..core.config import Config
from ..core.logger import PipelineLogger
from ..core.error_handler import ErrorHandler


@dataclass
class LayoutParameters:
    """布局参数配置"""
    algorithm: str = 'force_directed'
    random_seed: int = 42
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    spring_constant: float = 1.0
    repulsion_strength: float = 1.0
    cooling_factor: float = 0.95
    initial_temperature: float = 1.0


@dataclass
class VisualizationFilter:
    """可视化过滤配置"""
    min_edge_weight: float = 0.0
    max_nodes: int = 1000
    min_degree: int = 0
    top_k_nodes: Optional[int] = None
    filter_by_centrality: bool = False
    centrality_threshold: float = 0.1


@dataclass
class LayoutResult:
    """布局计算结果"""
    positions: Dict[int, Tuple[float, float]]
    algorithm_used: str
    iterations_completed: int
    converged: bool
    final_energy: float = 0.0
    computation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionCache:
    """节点位置缓存管理器"""
    
    def __init__(self, cache_dir: str = "cache/layouts"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = PipelineLogger(__name__)
    
    def _get_cache_key(self, graph_id: str, algorithm: str, seed: int) -> str:
        """生成缓存键"""
        return f"{graph_id}_{algorithm}_{seed}"
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def save_positions(self, graph_id: str, algorithm: str, seed: int, 
                      positions: Dict[int, Tuple[float, float]]) -> None:
        """保存节点位置到缓存"""
        try:
            cache_key = self._get_cache_key(graph_id, algorithm, seed)
            cache_path = self._get_cache_path(cache_key)
            
            cache_data = {
                'positions': positions,
                'graph_id': graph_id,
                'algorithm': algorithm,
                'seed': seed,
                'timestamp': np.datetime64('now').astype(str)
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.logger.debug(f"Saved layout positions to cache: {cache_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save positions to cache: {e}")
    
    def load_positions(self, graph_id: str, algorithm: str, seed: int) -> Optional[Dict[int, Tuple[float, float]]]:
        """从缓存加载节点位置"""
        try:
            cache_key = self._get_cache_key(graph_id, algorithm, seed)
            cache_path = self._get_cache_path(cache_key)
            
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 验证缓存数据
            if (cache_data.get('graph_id') == graph_id and 
                cache_data.get('algorithm') == algorithm and 
                cache_data.get('seed') == seed):
                
                self.logger.debug(f"Loaded layout positions from cache: {cache_path}")
                return cache_data['positions']
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load positions from cache: {e}")
            return None
    
    def clear_cache(self, graph_id: Optional[str] = None) -> None:
        """清理缓存"""
        try:
            if graph_id is None:
                # 清理所有缓存
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                self.logger.info("Cleared all layout cache")
            else:
                # 清理特定图的缓存
                pattern = f"{graph_id}_*.pkl"
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()
                self.logger.info(f"Cleared layout cache for graph: {graph_id}")
                
        except Exception as e:
            self.logger.warning(f"Failed to clear cache: {e}")


class ForceDirectedLayout:
    """力导向布局算法实现"""
    
    def __init__(self, params: LayoutParameters):
        self.params = params
        self.logger = PipelineLogger(__name__)
    
    def compute(self, graph: Any, node_list: List[int]) -> LayoutResult:
        """计算力导向布局"""
        if eg is None:
            raise ImportError("EasyGraph is required for layout computation")
        
        # 设置随机种子确保确定性
        np.random.seed(self.params.random_seed)
        random.seed(self.params.random_seed)
        
        n_nodes = len(node_list)
        if n_nodes == 0:
            return LayoutResult(
                positions={},
                algorithm_used='force_directed',
                iterations_completed=0,
                converged=True
            )
        
        # 初始化节点位置
        positions = self._initialize_positions(node_list)
        
        # 力导向迭代
        temperature = self.params.initial_temperature
        converged = False
        
        for iteration in range(self.params.max_iterations):
            forces = self._calculate_forces(graph, positions, node_list)
            
            # 更新位置
            max_displacement = 0.0
            for node_id in node_list:
                if node_id in forces:
                    fx, fy = forces[node_id]
                    
                    # 限制位移大小
                    displacement = min(math.sqrt(fx*fx + fy*fy), temperature)
                    if displacement > 0:
                        fx = fx / math.sqrt(fx*fx + fy*fy) * displacement
                        fy = fy / math.sqrt(fx*fx + fy*fy) * displacement
                    
                    # 更新位置
                    old_x, old_y = positions[node_id]
                    new_x, new_y = old_x + fx, old_y + fy
                    positions[node_id] = (new_x, new_y)
                    
                    max_displacement = max(max_displacement, displacement)
            
            # 降温
            temperature *= self.params.cooling_factor
            
            # 检查收敛
            if max_displacement < self.params.convergence_threshold:
                converged = True
                break
        
        return LayoutResult(
            positions=positions,
            algorithm_used='force_directed',
            iterations_completed=iteration + 1,
            converged=converged,
            final_energy=self._calculate_energy(graph, positions, node_list)
        )
    
    def _initialize_positions(self, node_list: List[int]) -> Dict[int, Tuple[float, float]]:
        """初始化节点位置"""
        positions = {}
        n_nodes = len(node_list)
        
        if n_nodes == 1:
            positions[node_list[0]] = (0.0, 0.0)
        else:
            # 在圆形区域内随机分布
            for i, node_id in enumerate(node_list):
                angle = 2 * math.pi * i / n_nodes
                radius = np.random.uniform(0.1, 1.0)
                x = radius * math.cos(angle) + np.random.normal(0, 0.1)
                y = radius * math.sin(angle) + np.random.normal(0, 0.1)
                positions[node_id] = (x, y)
        
        return positions
    
    def _calculate_forces(self, graph: Any, positions: Dict[int, Tuple[float, float]], 
                         node_list: List[int]) -> Dict[int, Tuple[float, float]]:
        """计算节点受力"""
        forces = {node_id: (0.0, 0.0) for node_id in node_list}
        
        # 斥力计算
        for i, node1 in enumerate(node_list):
            for j, node2 in enumerate(node_list):
                if i >= j:
                    continue
                
                x1, y1 = positions[node1]
                x2, y2 = positions[node2]
                
                dx = x1 - x2
                dy = y1 - y2
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > 0:
                    # 库仑斥力
                    repulsion_force = self.params.repulsion_strength / (distance * distance)
                    fx = repulsion_force * dx / distance
                    fy = repulsion_force * dy / distance
                    
                    # 更新受力
                    f1x, f1y = forces[node1]
                    f2x, f2y = forces[node2]
                    forces[node1] = (f1x + fx, f1y + fy)
                    forces[node2] = (f2x - fx, f2y - fy)
        
        # 引力计算（基于边）
        if hasattr(graph, 'edges'):
            for edge in graph.edges():
                if len(edge) >= 2:
                    node1, node2 = edge[0], edge[1]
                    if node1 in positions and node2 in positions:
                        x1, y1 = positions[node1]
                        x2, y2 = positions[node2]
                        
                        dx = x2 - x1
                        dy = y2 - y1
                        distance = math.sqrt(dx*dx + dy*dy)
                        
                        if distance > 0:
                            # 胡克引力
                            spring_force = self.params.spring_constant * distance
                            fx = spring_force * dx / distance
                            fy = spring_force * dy / distance
                            
                            # 更新受力
                            f1x, f1y = forces[node1]
                            f2x, f2y = forces[node2]
                            forces[node1] = (f1x + fx, f1y + fy)
                            forces[node2] = (f2x - fx, f2y - fy)
        
        return forces
    
    def _calculate_energy(self, graph: Any, positions: Dict[int, Tuple[float, float]], 
                         node_list: List[int]) -> float:
        """计算系统能量"""
        energy = 0.0
        
        # 斥力能量
        for i, node1 in enumerate(node_list):
            for j, node2 in enumerate(node_list):
                if i >= j:
                    continue
                
                x1, y1 = positions[node1]
                x2, y2 = positions[node2]
                distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                
                if distance > 0:
                    energy += self.params.repulsion_strength / distance
        
        # 引力能量
        if hasattr(graph, 'edges'):
            for edge in graph.edges():
                if len(edge) >= 2:
                    node1, node2 = edge[0], edge[1]
                    if node1 in positions and node2 in positions:
                        x1, y1 = positions[node1]
                        x2, y2 = positions[node2]
                        distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        energy += 0.5 * self.params.spring_constant * distance * distance
        
        return energy


class HierarchicalLayout:
    """层级布局算法实现"""
    
    def __init__(self, params: LayoutParameters):
        self.params = params
        self.logger = PipelineLogger(__name__)
    
    def compute(self, graph: Any, node_list: List[int]) -> LayoutResult:
        """计算层级布局"""
        if eg is None:
            raise ImportError("EasyGraph is required for layout computation")
        
        # 设置随机种子
        np.random.seed(self.params.random_seed)
        random.seed(self.params.random_seed)
        
        n_nodes = len(node_list)
        if n_nodes == 0:
            return LayoutResult(
                positions={},
                algorithm_used='hierarchical',
                iterations_completed=0,
                converged=True
            )
        
        # 计算节点层级
        levels = self._compute_node_levels(graph, node_list)
        
        # 按层级分组
        level_groups = {}
        for node_id, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node_id)
        
        # 计算位置
        positions = {}
        max_level = max(level_groups.keys()) if level_groups else 0
        
        for level, nodes in level_groups.items():
            y = max_level - level  # 顶层在上方
            n_nodes_in_level = len(nodes)
            
            if n_nodes_in_level == 1:
                positions[nodes[0]] = (0.0, float(y))
            else:
                for i, node_id in enumerate(nodes):
                    x = (i - (n_nodes_in_level - 1) / 2) * 2.0
                    positions[node_id] = (x, float(y))
        
        return LayoutResult(
            positions=positions,
            algorithm_used='hierarchical',
            iterations_completed=1,
            converged=True
        )
    
    def _compute_node_levels(self, graph: Any, node_list: List[int]) -> Dict[int, int]:
        """计算节点层级（基于BFS）"""
        levels = {}
        
        if not node_list:
            return levels
        
        # 选择起始节点（度最大的节点）
        start_node = node_list[0]
        max_degree = 0
        
        for node_id in node_list:
            degree = 0
            if hasattr(graph, 'degree'):
                degree = graph.degree(node_id)
            elif hasattr(graph, 'edges'):
                # 手动计算度
                for edge in graph.edges():
                    if len(edge) >= 2 and (edge[0] == node_id or edge[1] == node_id):
                        degree += 1
            
            if degree > max_degree:
                max_degree = degree
                start_node = node_id
        
        # BFS计算层级
        visited = set()
        queue = [(start_node, 0)]
        levels[start_node] = 0
        visited.add(start_node)
        
        while queue:
            current_node, current_level = queue.pop(0)
            
            # 获取邻居节点
            neighbors = []
            if hasattr(graph, 'neighbors'):
                neighbors = list(graph.neighbors(current_node))
            elif hasattr(graph, 'edges'):
                for edge in graph.edges():
                    if len(edge) >= 2:
                        if edge[0] == current_node and edge[1] in node_list:
                            neighbors.append(edge[1])
                        elif edge[1] == current_node and edge[0] in node_list:
                            neighbors.append(edge[0])
            
            for neighbor in neighbors:
                if neighbor not in visited and neighbor in node_list:
                    levels[neighbor] = current_level + 1
                    visited.add(neighbor)
                    queue.append((neighbor, current_level + 1))
        
        # 为未访问的节点分配层级
        for node_id in node_list:
            if node_id not in levels:
                levels[node_id] = 0
        
        return levels


class DeterministicLayoutEngine:
    """
    确定性布局引擎
    
    实现固定种子的力导向布局、节点位置缓存机制、层级和社群辅助布局以及可视化过滤功能。
    根据需求1.1、1.3、1.4、1.5确保布局的确定性和可复现性。
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化布局引擎
        
        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or Config()
        self.logger = PipelineLogger(__name__)
        self.error_handler = ErrorHandler()
        
        # 加载布局参数
        self.params = self._load_layout_parameters()
        
        # 初始化缓存管理器
        cache_dir = self.config.get('output.base_path', 'output') + '/cache/layouts'
        self.position_cache = PositionCache(cache_dir)
        
        # 初始化布局算法
        self.force_directed = ForceDirectedLayout(self.params)
        self.hierarchical = HierarchicalLayout(self.params)
        
        self.logger.info(f"Initialized DeterministicLayoutEngine with algorithm: {self.params.algorithm}")
    
    def _load_layout_parameters(self) -> LayoutParameters:
        """加载布局参数"""
        return LayoutParameters(
            algorithm=self.config.get('layout_engine.algorithm', 'force_directed'),
            random_seed=self.config.get('layout_engine.random_seed', 42),
            max_iterations=self.config.get('layout_engine.max_iterations', 1000),
            convergence_threshold=self.config.get('layout_engine.convergence_threshold', 1e-6),
            spring_constant=self.config.get('layout_engine.spring_constant', 1.0),
            repulsion_strength=self.config.get('layout_engine.repulsion_strength', 1.0),
            cooling_factor=self.config.get('layout_engine.cooling_factor', 0.95),
            initial_temperature=self.config.get('layout_engine.initial_temperature', 1.0)
        )
    
    def compute_layout(self, graph: Any, graph_id: str, 
                      force_recompute: bool = False) -> LayoutResult:
        """
        计算确定性布局
        
        Args:
            graph: EasyGraph图对象
            graph_id: 图的唯一标识符
            force_recompute: 是否强制重新计算（忽略缓存）
            
        Returns:
            LayoutResult: 布局计算结果
        """
        try:
            # 检查缓存
            if not force_recompute and self.config.get('layout_engine.cache_enabled', True):
                cached_positions = self.position_cache.load_positions(
                    graph_id, self.params.algorithm, self.params.random_seed
                )
                if cached_positions is not None:
                    self.logger.debug(f"Using cached layout for graph: {graph_id}")
                    return LayoutResult(
                        positions=cached_positions,
                        algorithm_used=self.params.algorithm,
                        iterations_completed=0,
                        converged=True,
                        metadata={'from_cache': True}
                    )
            
            # 获取节点列表
            node_list = self._get_node_list(graph)
            
            if not node_list:
                self.logger.warning(f"Graph {graph_id} has no nodes")
                return LayoutResult(
                    positions={},
                    algorithm_used=self.params.algorithm,
                    iterations_completed=0,
                    converged=True
                )
            
            # 计算布局
            self.logger.info(f"Computing {self.params.algorithm} layout for graph {graph_id} "
                           f"with {len(node_list)} nodes")
            
            if self.params.algorithm == 'force_directed':
                result = self.force_directed.compute(graph, node_list)
            elif self.params.algorithm == 'hierarchical':
                result = self.hierarchical.compute(graph, node_list)
            else:
                raise ValueError(f"Unsupported layout algorithm: {self.params.algorithm}")
            
            # 缓存结果
            if self.config.get('layout_engine.cache_enabled', True):
                self.position_cache.save_positions(
                    graph_id, self.params.algorithm, self.params.random_seed, result.positions
                )
            
            self.logger.info(f"Layout computation completed for graph {graph_id}: "
                           f"{result.iterations_completed} iterations, "
                           f"converged: {result.converged}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Layout computation failed for graph {graph_id}: {e}")
            
            # 使用错误处理器的回退策略
            fallback_result = self.error_handler.handle_error(
                e, 'layout_computation', {'graph': graph, 'graph_id': graph_id}
            )
            
            if isinstance(fallback_result, dict):
                return LayoutResult(
                    positions=fallback_result,
                    algorithm_used='fallback_random',
                    iterations_completed=0,
                    converged=False,
                    metadata={'error': str(e), 'fallback_used': True}
                )
            else:
                raise
    
    def _get_node_list(self, graph: Any) -> List[int]:
        """获取图的节点列表"""
        try:
            if hasattr(graph, 'nodes'):
                return list(graph.nodes())
            elif hasattr(graph, 'number_of_nodes'):
                # 假设节点ID是连续的整数
                return list(range(graph.number_of_nodes()))
            else:
                return []
        except Exception as e:
            self.logger.warning(f"Failed to get node list: {e}")
            return []
    
    def apply_visualization_filter(self, graph: Any, positions: Dict[int, Tuple[float, float]], 
                                 filter_config: Optional[VisualizationFilter] = None) -> Tuple[Any, Dict[int, Tuple[float, float]]]:
        """
        应用可视化过滤
        
        Args:
            graph: 原始图对象
            positions: 节点位置
            filter_config: 过滤配置
            
        Returns:
            Tuple[Any, Dict]: 过滤后的图和位置
        """
        if filter_config is None:
            filter_config = VisualizationFilter(
                min_edge_weight=self.config.get('layout_engine.min_edge_weight', 0.0),
                max_nodes=self.config.get('layout_engine.max_nodes', 1000),
                min_degree=self.config.get('layout_engine.min_degree', 0)
            )
        
        try:
            # 获取节点列表
            all_nodes = list(positions.keys())
            
            # 按度过滤节点
            filtered_nodes = []
            for node_id in all_nodes:
                degree = 0
                if hasattr(graph, 'degree'):
                    degree = graph.degree(node_id)
                elif hasattr(graph, 'edges'):
                    for edge in graph.edges():
                        if len(edge) >= 2 and (edge[0] == node_id or edge[1] == node_id):
                            degree += 1
                
                if degree >= filter_config.min_degree:
                    filtered_nodes.append(node_id)
            
            # 限制节点数量
            if filter_config.max_nodes and len(filtered_nodes) > filter_config.max_nodes:
                # 按度排序，保留度最高的节点
                node_degrees = []
                for node_id in filtered_nodes:
                    degree = 0
                    if hasattr(graph, 'degree'):
                        degree = graph.degree(node_id)
                    node_degrees.append((node_id, degree))
                
                node_degrees.sort(key=lambda x: x[1], reverse=True)
                filtered_nodes = [node_id for node_id, _ in node_degrees[:filter_config.max_nodes]]
            
            # 过滤位置
            filtered_positions = {node_id: positions[node_id] for node_id in filtered_nodes if node_id in positions}
            
            self.logger.info(f"Visualization filter applied: {len(all_nodes)} -> {len(filtered_nodes)} nodes")
            
            return graph, filtered_positions
            
        except Exception as e:
            self.logger.warning(f"Visualization filtering failed: {e}")
            return graph, positions
    
    def update_subgraph_positions(self, subgraph: StateSubgraph, 
                                global_positions: Dict[int, Tuple[float, float]]) -> None:
        """
        更新子图的节点位置，确保与全局图一致
        
        Args:
            subgraph: 州级子图对象
            global_positions: 全局图的节点位置
        """
        try:
            active_nodes = subgraph.get_active_nodes()
            
            for node_id in active_nodes:
                if node_id in global_positions:
                    subgraph.set_node_position(node_id, global_positions[node_id])
            
            self.logger.debug(f"Updated positions for {len(active_nodes)} nodes in subgraph {subgraph.state_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to update subgraph positions: {e}")
    
    def clear_cache(self, graph_id: Optional[str] = None) -> None:
        """
        清理布局缓存
        
        Args:
            graph_id: 要清理的图ID，如果为None则清理所有缓存
        """
        self.position_cache.clear_cache(graph_id)
    
    def get_layout_info(self) -> Dict[str, Any]:
        """获取布局引擎信息"""
        return {
            'algorithm': self.params.algorithm,
            'random_seed': self.params.random_seed,
            'max_iterations': self.params.max_iterations,
            'convergence_threshold': self.params.convergence_threshold,
            'cache_enabled': self.config.get('layout_engine.cache_enabled', True),
            'cache_dir': str(self.position_cache.cache_dir)
        }