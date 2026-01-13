#!/usr/bin/env python3
"""
Plotly交互式可视化生成器

使用Plotly库生成高质量的交互式网络可视化，提供更好的用户体验和视觉效果。
"""

import os
import sys
import json
import numpy as np
import networkx as nx
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly未安装，请运行: pip install plotly kaleido")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class PlotlyNetworkVisualizer:
    """Plotly网络可视化器"""
    
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Plotly配置
        self.plotly_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'network_visualization',
                'height': 1200,
                'width': 1600,
                'scale': 2
            }
        }
        
        # 视觉配置
        self.visual_config = {
            'node_size_range': (10, 50),
            'edge_width_range': (0.5, 5),
            'opacity_range': (0.3, 0.9),
            'color_palette': px.colors.qualitative.Set3,
            'background_color': 'white',
            'grid_color': 'lightgray',
            'text_color': 'black'
        }
    
    def create_network_layout(self, graph: nx.Graph, layout_type='spring') -> Dict[str, Tuple[float, float]]:
        """创建网络布局"""
        print(f"🎯 计算{layout_type}布局...")
        
        if layout_type == 'spring':
            # Spring布局，适合大多数网络
            pos = nx.spring_layout(graph, 
                                 k=1.0/np.sqrt(graph.number_of_nodes()),
                                 iterations=50,
                                 seed=self.random_seed)
        elif layout_type == 'circular':
            # 圆形布局，适合小型网络
            pos = nx.circular_layout(graph)
        elif layout_type == 'kamada_kawai':
            # Kamada-Kawai布局，适合中等规模网络
            pos = nx.kamada_kawai_layout(graph)
        elif layout_type == 'fruchterman_reingold':
            # Fruchterman-Reingold布局
            pos = nx.fruchterman_reingold_layout(graph, seed=self.random_seed)
        else:
            # 默认使用spring布局
            pos = nx.spring_layout(graph, seed=self.random_seed)
        
        return pos
    
    def prepare_node_data(self, graph: nx.Graph, positions: Dict) -> Dict[str, List]:
        """准备节点数据"""
        print("📊 准备节点数据...")
        
        node_data = {
            'x': [],
            'y': [],
            'text': [],
            'hovertext': [],
            'size': [],
            'color': [],
            'symbol': [],
            'node_id': []
        }
        
        # 获取节点属性
        communities = nx.get_node_attributes(graph, 'community')
        importance_scores = nx.get_node_attributes(graph, 'importance')
        node_roles = nx.get_node_attributes(graph, 'role')
        tf_idf_scores = nx.get_node_attributes(graph, 'tf_idf_score')
        frequencies = nx.get_node_attributes(graph, 'frequency')
        
        # 计算节点大小和颜色
        for node in graph.nodes():
            pos = positions.get(node, (0, 0))
            node_data['x'].append(pos[0])
            node_data['y'].append(pos[1])
            node_data['node_id'].append(node)
            
            # 节点标签
            node_data['text'].append(node if len(node) <= 15 else node[:12] + "...")
            
            # 悬停信息
            community = communities.get(node, 0)
            importance = importance_scores.get(node, 0)
            role = node_roles.get(node, 'unknown')
            tfidf = tf_idf_scores.get(node, 0)
            freq = frequencies.get(node, 0)
            degree = graph.degree(node)
            
            hover_info = f"""
            <b>{node}</b><br>
            Community: {community}<br>
            Role: {role}<br>
            Degree: {degree}<br>
            Importance: {importance:.3f}<br>
            TF-IDF: {tfidf:.3f}<br>
            Frequency: {freq}
            """
            node_data['hovertext'].append(hover_info)
            
            # 节点大小（基于重要性或TF-IDF）
            if tf_idf_scores:
                size_score = tfidf
                max_score = max(tf_idf_scores.values()) if tf_idf_scores.values() else 1
            else:
                size_score = importance
                max_score = max(importance_scores.values()) if importance_scores.values() else 1
            
            normalized_size = size_score / max_score if max_score > 0 else 0.1
            size = self.visual_config['node_size_range'][0] + \
                   (self.visual_config['node_size_range'][1] - self.visual_config['node_size_range'][0]) * normalized_size
            node_data['size'].append(size)
            
            # 节点颜色（基于社区）
            color_idx = community % len(self.visual_config['color_palette'])
            node_data['color'].append(self.visual_config['color_palette'][color_idx])
            
            # 节点形状（基于角色）
            if role == 'core':
                node_data['symbol'].append('diamond')  # 核心节点用菱形
            else:
                node_data['symbol'].append('circle')   # 外围节点用圆形
        
        return node_data
    
    def prepare_edge_data(self, graph: nx.Graph, positions: Dict) -> Dict[str, List]:
        """准备边数据"""
        print("🔗 准备边数据...")
        
        edge_data = {
            'x': [],
            'y': [],
            'hovertext': [],
            'width': [],
            'color': []
        }
        
        # 获取边权重
        edge_weights = [data.get('weight', 1) for u, v, data in graph.edges(data=True)]
        max_weight = max(edge_weights) if edge_weights else 1
        min_weight = min(edge_weights) if edge_weights else 1
        
        # 获取社区信息用于边着色
        communities = nx.get_node_attributes(graph, 'community')
        
        for u, v, data in graph.edges(data=True):
            pos_u = positions.get(u, (0, 0))
            pos_v = positions.get(v, (0, 0))
            
            # 边的坐标（包括断点用于分离边）
            edge_data['x'].extend([pos_u[0], pos_v[0], None])
            edge_data['y'].extend([pos_u[1], pos_v[1], None])
            
            # 边的悬停信息
            weight = data.get('weight', 1)
            hover_info = f"<b>{u}</b> ↔ <b>{v}</b><br>Weight: {weight:.3f}"
            edge_data['hovertext'].extend([hover_info, hover_info, None])
            
            # 边的宽度（基于权重）
            if max_weight > min_weight:
                normalized_weight = (weight - min_weight) / (max_weight - min_weight)
            else:
                normalized_weight = 0.5
            
            width = self.visual_config['edge_width_range'][0] + \
                   (self.visual_config['edge_width_range'][1] - self.visual_config['edge_width_range'][0]) * normalized_weight
            edge_data['width'].extend([width, width, None])
            
            # 边的颜色（基于是否为社区内连接）
            u_community = communities.get(u, 0)
            v_community = communities.get(v, 0)
            
            if u_community == v_community:
                # 社区内连接 - 较深颜色
                edge_color = 'rgba(100, 100, 100, 0.6)'
            else:
                # 社区间连接 - 较浅颜色
                edge_color = 'rgba(150, 150, 150, 0.3)'
            
            edge_data['color'].extend([edge_color, edge_color, None])
        
        return edge_data
    
    def create_interactive_network(self, graph: nx.Graph, positions: Dict, 
                                 title: str = "Interactive Network Visualization") -> go.Figure:
        """创建交互式网络图"""
        print("🎨 创建交互式网络图...")
        
        # 准备数据
        node_data = self.prepare_node_data(graph, positions)
        edge_data = self.prepare_edge_data(graph, positions)
        
        # 创建图形
        fig = go.Figure()
        
        # 添加边
        fig.add_trace(go.Scatter(
            x=edge_data['x'],
            y=edge_data['y'],
            mode='lines',
            line=dict(
                width=1,
                color='rgba(125, 125, 125, 0.4)'
            ),
            hoverinfo='skip',
            showlegend=False,
            name='Edges'
        ))
        
        # 按社区分组添加节点
        communities = nx.get_node_attributes(graph, 'community')
        unique_communities = sorted(set(communities.values())) if communities else [0]
        
        for community in unique_communities:
            # 筛选该社区的节点
            community_indices = [i for i, node_id in enumerate(node_data['node_id']) 
                               if communities.get(node_id, 0) == community]
            
            if not community_indices:
                continue
            
            community_x = [node_data['x'][i] for i in community_indices]
            community_y = [node_data['y'][i] for i in community_indices]
            community_text = [node_data['text'][i] for i in community_indices]
            community_hovertext = [node_data['hovertext'][i] for i in community_indices]
            community_size = [node_data['size'][i] for i in community_indices]
            community_symbol = [node_data['symbol'][i] for i in community_indices]
            
            # 社区颜色
            color_idx = community % len(self.visual_config['color_palette'])
            community_color = self.visual_config['color_palette'][color_idx]
            
            fig.add_trace(go.Scatter(
                x=community_x,
                y=community_y,
                mode='markers+text',
                marker=dict(
                    size=community_size,
                    color=community_color,
                    symbol=community_symbol,
                    line=dict(width=1, color='black'),
                    opacity=0.8
                ),
                text=community_text,
                textposition="middle center",
                textfont=dict(size=8, color='black'),
                hovertext=community_hovertext,
                hoverinfo='text',
                name=f'Community {community}',
                showlegend=True
            ))
        
        # 更新布局
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=16, color=self.visual_config['text_color'])
            ),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text=f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}<br>" +
                         f"Communities: {len(unique_communities)}, Seed: {self.random_seed}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=10, color=self.visual_config['text_color'])
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor=self.visual_config['background_color'],
            paper_bgcolor=self.visual_config['background_color']
        )
        
        return fig
    
    def create_network_statistics_dashboard(self, graph: nx.Graph, 
                                          communities: Dict = None) -> go.Figure:
        """创建网络统计仪表板"""
        print("📊 创建网络统计仪表板...")
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Degree Distribution', 'Community Sizes', 
                          'Centrality Measures', 'Network Metrics'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "indicator"}]]
        )
        
        # 1. 度分布
        degrees = [graph.degree(node) for node in graph.nodes()]
        degree_counts = {}
        for degree in degrees:
            degree_counts[degree] = degree_counts.get(degree, 0) + 1
        
        fig.add_trace(
            go.Bar(x=list(degree_counts.keys()), 
                   y=list(degree_counts.values()),
                   name="Degree Distribution"),
            row=1, col=1
        )
        
        # 2. 社区大小
        if communities:
            community_sizes = {}
            for community in communities.values():
                community_sizes[community] = community_sizes.get(community, 0) + 1
            
            fig.add_trace(
                go.Pie(labels=[f"Community {k}" for k in community_sizes.keys()],
                       values=list(community_sizes.values()),
                       name="Community Sizes"),
                row=1, col=2
            )
        
        # 3. 中心性度量
        if graph.number_of_nodes() > 0:
            betweenness = nx.betweenness_centrality(graph)
            closeness = nx.closeness_centrality(graph)
            
            nodes = list(graph.nodes())[:20]  # 只显示前20个节点
            bet_values = [betweenness.get(node, 0) for node in nodes]
            clo_values = [closeness.get(node, 0) for node in nodes]
            
            fig.add_trace(
                go.Scatter(x=bet_values, y=clo_values,
                          mode='markers+text',
                          text=[node[:10] for node in nodes],
                          textposition="top center",
                          name="Centrality"),
                row=2, col=1
            )
        
        # 4. 网络指标
        density = nx.density(graph)
        clustering = nx.average_clustering(graph)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=density,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Network Density"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 1], 'color': "gray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 0.9}}),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Network Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        return fig
    
    def save_visualization(self, fig: go.Figure, output_path: str, 
                          format: str = 'html') -> str:
        """保存可视化"""
        print(f"💾 保存可视化到: {output_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format == 'html':
            # 保存为交互式HTML
            fig.write_html(output_path, config=self.plotly_config)
        elif format == 'png':
            # 保存为静态PNG
            fig.write_image(output_path, width=1600, height=1200, scale=2)
        elif format == 'pdf':
            # 保存为PDF
            fig.write_image(output_path, width=1600, height=1200, scale=2)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        return output_path


def test_plotly_visualization():
    """测试Plotly可视化"""
    print("🧪 测试Plotly网络可视化")
    print("=" * 50)
    
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly未安装，无法运行测试")
        return False
    
    try:
        # 导入主程序
        from complete_usage_guide import ResearchPipelineCLI
        
        # 初始化管线
        print("🔄 初始化管线...")
        app = ResearchPipelineCLI()
        
        # 设置输出目录
        output_dir = "/Users/zhangjingsen/Desktop/python/graph4socialscience/hajimi/七周目"
        app.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📁 输出目录: {output_dir}")
        
        # 创建示例数据
        print("📊 创建示例数据...")
        app.create_sample_research_data()
        
        # 运行管线步骤
        print("\\n🔄 运行管线步骤...")
        
        # 文本清理
        print("1️⃣ 文本清理...")
        app.clean_and_normalize_text()
        
        # 词组提取
        print("2️⃣ 词组提取...")
        app.extract_tokens_and_phrases()
        
        # 全局图构建
        print("3️⃣ 全局图构建...")
        app.build_global_graph()
        
        # 子图激活
        print("4️⃣ 子图激活...")
        app.activate_state_subgraphs()
        
        # 检查图对象
        if not hasattr(app, 'global_graph_object') or app.global_graph_object is None:
            print("❌ 全局图对象不存在")
            return False
        
        graph = app.global_graph_object
        positions = app.global_layout_positions
        
        print(f"📊 图统计: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")
        
        # 初始化Plotly可视化器
        print("\\n🎨 初始化Plotly可视化器...")
        visualizer = PlotlyNetworkVisualizer(random_seed=42)
        
        # 创建可视化目录
        viz_dir = os.path.join(output_dir, "plotly_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 创建交互式全局网络图
        print("\\n🌐 创建交互式全局网络图...")
        global_fig = visualizer.create_interactive_network(
            graph, positions, 
            title=f"Interactive Global Co-occurrence Network (Seed: 42)"
        )
        
        # 保存HTML版本（交互式）
        global_html_path = os.path.join(viz_dir, f"global_network_interactive_{timestamp}.html")
        visualizer.save_visualization(global_fig, global_html_path, format='html')
        print(f"✅ 保存交互式版本: {global_html_path}")
        
        # 保存PNG版本（静态）
        try:
            global_png_path = os.path.join(viz_dir, f"global_network_static_{timestamp}.png")
            visualizer.save_visualization(global_fig, global_png_path, format='png')
            print(f"✅ 保存静态版本: {global_png_path}")
        except Exception as e:
            print(f"⚠️ PNG保存失败 (需要kaleido): {e}")
        
        # 2. 创建网络统计仪表板
        print("\\n📊 创建网络统计仪表板...")
        communities = nx.get_node_attributes(graph, 'community')
        dashboard_fig = visualizer.create_network_statistics_dashboard(graph, communities)
        
        dashboard_html_path = os.path.join(viz_dir, f"network_dashboard_{timestamp}.html")
        visualizer.save_visualization(dashboard_fig, dashboard_html_path, format='html')
        print(f"✅ 保存仪表板: {dashboard_html_path}")
        
        # 3. 为每个状态创建子图可视化
        print("\\n🗺️ 创建状态子图可视化...")
        if hasattr(app, 'state_subgraph_objects') and app.state_subgraph_objects:
            
            for state, subgraph in app.state_subgraph_objects.items():
                if subgraph.number_of_nodes() > 0:
                    print(f"   🎨 处理状态: {state}")
                    
                    # 使用全局位置确保一致性
                    subgraph_positions = {node: positions[node] for node in subgraph.nodes() 
                                        if node in positions}
                    
                    # 创建子图可视化
                    subgraph_fig = visualizer.create_interactive_network(
                        subgraph, subgraph_positions,
                        title=f"State {state} Thematic Network ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)"
                    )
                    
                    # 保存子图
                    subgraph_html_path = os.path.join(viz_dir, f"state_{state}_network_{timestamp}.html")
                    visualizer.save_visualization(subgraph_fig, subgraph_html_path, format='html')
                    print(f"      ✅ 保存: state_{state}_network_{timestamp}.html")
        
        # 4. 创建布局对比图
        print("\\n🔄 创建不同布局对比...")
        layout_types = ['spring', 'circular', 'kamada_kawai']
        
        for layout_type in layout_types:
            try:
                print(f"   🎯 创建{layout_type}布局...")
                layout_positions = visualizer.create_network_layout(graph, layout_type)
                
                layout_fig = visualizer.create_interactive_network(
                    graph, layout_positions,
                    title=f"Global Network - {layout_type.title()} Layout"
                )
                
                layout_html_path = os.path.join(viz_dir, f"global_network_{layout_type}_{timestamp}.html")
                visualizer.save_visualization(layout_fig, layout_html_path, format='html')
                print(f"      ✅ 保存: global_network_{layout_type}_{timestamp}.html")
                
            except Exception as e:
                print(f"      ⚠️ {layout_type}布局失败: {e}")
        
        # 生成总结报告
        print("\\n📋 生成可视化总结...")
        
        summary_report = f"""# Plotly网络可视化报告

## 生成时间
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 网络统计
- 节点数: {graph.number_of_nodes()}
- 边数: {graph.number_of_edges()}
- 密度: {nx.density(graph):.4f}
- 社区数: {len(set(communities.values())) if communities else 0}

## 生成的可视化文件

### 交互式网络图
- `global_network_interactive_{timestamp}.html` - 主要的交互式网络图
- `network_dashboard_{timestamp}.html` - 网络统计仪表板

### 状态子图
"""
        
        if hasattr(app, 'state_subgraph_objects'):
            for state in app.state_subgraph_objects.keys():
                summary_report += f"- `state_{state}_network_{timestamp}.html` - {state}州网络图\\n"
        
        summary_report += f"""
### 布局对比
- `global_network_spring_{timestamp}.html` - Spring布局
- `global_network_circular_{timestamp}.html` - 圆形布局  
- `global_network_kamada_kawai_{timestamp}.html` - Kamada-Kawai布局

## 使用说明
1. 打开HTML文件在浏览器中查看交互式可视化
2. 使用鼠标缩放、平移和悬停查看详细信息
3. 点击图例可以隐藏/显示特定社区
4. 仪表板提供网络的统计分析

## 优势
- 🎯 交互式操作，可缩放和平移
- 📊 丰富的悬停信息
- 🎨 美观的视觉效果
- 📱 响应式设计，支持移动设备
- 💾 可导出为静态图像
"""
        
        summary_path = os.path.join(viz_dir, f"visualization_summary_{timestamp}.md")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"✅ 生成总结报告: {summary_path}")
        
        print("\\n🎉 Plotly可视化测试完成！")
        print(f"📁 所有文件保存在: {viz_dir}")
        print("\\n📊 生成的文件:")
        
        # 列出生成的文件
        for file in os.listdir(viz_dir):
            if file.endswith(('.html', '.png', '.md')):
                file_path = os.path.join(viz_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"   📄 {file} ({file_size:,} bytes)")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🎨 Plotly网络可视化生成器")
    print("=" * 60)
    
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly未安装")
        print("请运行以下命令安装:")
        print("pip install plotly kaleido")
        return 1
    
    # 运行测试
    success = test_plotly_visualization()
    
    if success:
        print("\\n✅ Plotly可视化生成成功！")
        print("🎯 相比matplotlib的优势:")
        print("   - 🖱️ 交互式操作（缩放、平移、悬停）")
        print("   - 📊 丰富的统计仪表板")
        print("   - 🎨 更美观的视觉效果")
        print("   - 📱 响应式设计")
        print("   - 🔍 详细的节点和边信息")
        print("   - 💾 多种导出格式")
        return 0
    else:
        print("\\n❌ Plotly可视化生成失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())