#!/usr/bin/env python3
"""
语义共词网络分析器 - 整合版本

这是一个整合了所有核心功能的独立脚本，用于实际运行语义共词网络分析。
简化了复杂的文件结构，提供了直接可用的分析功能。

使用方法:
    python semantic_coword_analyzer.py input_folder output_folder
    python semantic_coword_analyzer.py --help

作者: 语义共词网络分析管线项目组
版本: 1.0.0
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re
import math

# 尝试导入可选依赖
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("警告: NLTK不可用，将使用基本分词功能")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("警告: jieba不可用，中文处理功能受限")

try:
    import numpy as np
    import scipy.sparse
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: scipy不可用，将使用基本矩阵操作")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib不可用，可视化功能受限")

try:
    import easygraph as eg
    EASYGRAPH_AVAILABLE = True
except ImportError:
    EASYGRAPH_AVAILABLE = False
    print("警告: EasyGraph不可用，将使用基本图功能")


# ============================================================================
# 数据模型定义
# ============================================================================

@dataclass
class TOCDocument:
    """TOC文档数据模型"""
    segment_id: str
    title: str
    level: int
    order: int
    text: str
    state: Optional[str] = None
    language: Optional[str] = None

@dataclass
class ProcessedDocument:
    """处理后的文档数据模型"""
    original_doc: TOCDocument
    cleaned_text: str
    tokens: List[str]
    phrases: List[str]
    language: str

@dataclass
class AnalysisResult:
    """分析结果数据模型"""
    total_documents: int
    processed_documents: int
    total_phrases: int
    unique_phrases: int
    global_graph_nodes: int
    global_graph_edges: int
    state_subgraphs: Dict[str, Dict[str, Any]]
    processing_time: float
    output_files: List[str]


# ============================================================================
# 核心分析器类
# ============================================================================

class SemanticCowordAnalyzer:
    """语义共词网络分析器 - 整合版本"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化分析器"""
        # 合并默认配置和用户配置
        default_config = self._get_default_config()
        if config:
            # 深度合并配置
            self.config = self._merge_configs(default_config, config)
        else:
            self.config = default_config
            
        self.logger = self._setup_logger()
        
        # 初始化组件
        self.text_processor = TextProcessor(self.config.get('text_processing', {}))
        self.phrase_extractor = PhraseExtractor(self.config.get('text_processing', {}))
        self.stopword_discoverer = StopwordDiscoverer(self.config.get('stopword_discovery', {}))
        self.graph_builder = GraphBuilder(self.config.get('graph_construction', {}))
        self.output_manager = OutputManager(self.config.get('output', {}))
        
        self.logger.info("语义共词网络分析器初始化完成")
    
    def _merge_configs(self, default: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并配置字典"""
        result = default.copy()
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'text_processing': {
                'min_phrase_frequency': 2,
                'ngram_size': 2,
                'remove_punctuation': True,
                'convert_to_lowercase': True
            },
            'stopword_discovery': {
                'tfidf_threshold': 0.01,  # 降低阈值，保留更多词
                'frequency_threshold': 0.95,  # 提高阈值，只过滤最常见的词
                'enable_dynamic_discovery': True
            },
            'graph_construction': {
                'preserve_isolated_nodes': True,
                'edge_weight_method': 'binary',
                'min_cooccurrence_count': 1
            },
            'output': {
                'generate_visualizations': True,
                'export_formats': ['json', 'csv'],
                'save_intermediate_results': True
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('SemanticCowordAnalyzer')
        logger.setLevel(getattr(logging, self.config['logging']['level']))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(self.config['logging'].get('format', '%(asctime)s - %(levelname)s - %(message)s'))
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze(self, input_dir: str, output_dir: str) -> AnalysisResult:
        """执行完整的语义共词网络分析"""
        start_time = time.time()
        
        self.logger.info(f"开始分析: {input_dir} -> {output_dir}")
        
        # 1. 加载和预处理文档
        documents = self._load_documents(input_dir)
        self.logger.info(f"加载了 {len(documents)} 个文档")
        
        # 2. 文本处理和词组抽取
        processed_docs = []
        all_phrases = []
        
        for doc in documents:
            processed_doc = self.text_processor.process_document(doc)
            phrases = self.phrase_extractor.extract_phrases(processed_doc)
            processed_doc.phrases = phrases
            processed_docs.append(processed_doc)
            all_phrases.extend(phrases)
        
        self.logger.info(f"抽取了 {len(all_phrases)} 个词组")
        
        # 3. 动态停词发现和过滤
        if self.config['stopword_discovery']['enable_dynamic_discovery']:
            stopwords = self.stopword_discoverer.discover_stopwords([doc.phrases for doc in processed_docs])
            self.logger.info(f"发现了 {len(stopwords)} 个动态停词")
            
            # 过滤停词
            for doc in processed_docs:
                doc.phrases = [p for p in doc.phrases if p not in stopwords]
        
        # 4. 构建全局图
        global_graph = self.graph_builder.build_global_graph(processed_docs)
        self.logger.info(f"构建了全局图: {global_graph['node_count']} 节点, {global_graph['edge_count']} 边")
        
        # 5. 构建州级子图
        state_subgraphs = {}
        states = set(doc.original_doc.state for doc in processed_docs if doc.original_doc.state)
        
        for state in states:
            state_docs = [doc for doc in processed_docs if doc.original_doc.state == state]
            subgraph = self.graph_builder.build_state_subgraph(global_graph, state_docs, state)
            state_subgraphs[state] = subgraph
            self.logger.info(f"构建了 {state} 子图: {subgraph['node_count']} 节点, {subgraph['edge_count']} 边")
        
        # 6. 生成输出
        output_files = self.output_manager.generate_outputs(
            global_graph, state_subgraphs, processed_docs, output_dir
        )
        
        processing_time = time.time() - start_time
        
        # 7. 创建分析结果
        result = AnalysisResult(
            total_documents=len(documents),
            processed_documents=len(processed_docs),
            total_phrases=len(all_phrases),
            unique_phrases=len(set(all_phrases)),
            global_graph_nodes=global_graph['node_count'],
            global_graph_edges=global_graph['edge_count'],
            state_subgraphs={k: {'nodes': v['node_count'], 'edges': v['edge_count']} 
                           for k, v in state_subgraphs.items()},
            processing_time=processing_time,
            output_files=output_files
        )
        
        self.logger.info(f"分析完成，耗时 {processing_time:.2f} 秒")
        return result
    
    def _load_documents(self, input_dir: str) -> List[TOCDocument]:
        """加载输入文档"""
        documents = []
        input_path = Path(input_dir)
        
        if not input_path.exists():
            raise ValueError(f"输入目录不存在: {input_dir}")
        
        for file_path in input_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 处理单个文档或文档列表
                if isinstance(data, list):
                    for item in data:
                        doc = TOCDocument(**item)
                        documents.append(doc)
                else:
                    doc = TOCDocument(**data)
                    documents.append(doc)
                    
            except Exception as e:
                self.logger.warning(f"跳过文件 {file_path}: {e}")
        
        return documents


# ============================================================================
# 文本处理器
# ============================================================================

class TextProcessor:
    """文本处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化停词表
        self.english_stopwords = set()
        self.chinese_stopwords = set()
        
        if NLTK_AVAILABLE:
            try:
                self.english_stopwords = set(stopwords.words('english'))
            except:
                pass
        
        # 基本英文停词
        basic_english_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
        self.english_stopwords.update(basic_english_stopwords)
        
        # 基本中文停词
        basic_chinese_stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '里', '就是', '还', '把', '比', '或者', '因为', '所以'
        }
        self.chinese_stopwords.update(basic_chinese_stopwords)
    
    def process_document(self, doc: TOCDocument) -> ProcessedDocument:
        """处理单个文档"""
        # 检测语言
        language = self._detect_language(doc.text)
        
        # 清洗文本
        cleaned_text = self._clean_text(doc.text)
        
        # 分词
        tokens = self._tokenize(cleaned_text, language)
        
        return ProcessedDocument(
            original_doc=doc,
            cleaned_text=cleaned_text,
            tokens=tokens,
            phrases=[],  # 将在词组抽取器中填充
            language=language
        )
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单的语言检测：统计中文字符比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fff]', text))
        
        if total_chars == 0:
            return 'english'
        
        chinese_ratio = chinese_chars / total_chars
        return 'chinese' if chinese_ratio > 0.3 else 'english'
    
    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 移除标点符号（可选）
        if self.config.get('remove_punctuation', True):
            text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 转换为小写（可选）
        if self.config.get('convert_to_lowercase', True):
            text = text.lower()
        
        return text
    
    def _tokenize(self, text: str, language: str) -> List[str]:
        """分词"""
        if language == 'chinese' and JIEBA_AVAILABLE:
            tokens = list(jieba.cut(text))
        elif language == 'english' and NLTK_AVAILABLE:
            tokens = word_tokenize(text)
        else:
            # 基本分词
            tokens = text.split()
        
        # 过滤停词和短词
        stopwords = self.chinese_stopwords if language == 'chinese' else self.english_stopwords
        tokens = [token for token in tokens if len(token) > 1 and token not in stopwords]
        
        return tokens


# ============================================================================
# 词组抽取器
# ============================================================================

class PhraseExtractor:
    """词组抽取器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_frequency = self.config.get('min_phrase_frequency', 2)
        self.ngram_size = self.config.get('ngram_size', 2)
    
    def extract_phrases(self, doc: ProcessedDocument) -> List[str]:
        """从文档中抽取词组"""
        if doc.language == 'chinese':
            return self._extract_chinese_phrases(doc.tokens)
        else:
            return self._extract_english_bigrams(doc.tokens)
    
    def _extract_english_bigrams(self, tokens: List[str]) -> List[str]:
        """抽取英文2-gram"""
        bigrams = []
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            bigrams.append(bigram)
        return bigrams
    
    def _extract_chinese_phrases(self, tokens: List[str]) -> List[str]:
        """抽取中文短语"""
        phrases = []
        
        # 2-gram
        for i in range(len(tokens) - 1):
            phrase = f"{tokens[i]}{tokens[i+1]}"
            if len(phrase) >= 2:  # 确保短语长度
                phrases.append(phrase)
        
        # 3-gram（可选）
        for i in range(len(tokens) - 2):
            phrase = f"{tokens[i]}{tokens[i+1]}{tokens[i+2]}"
            if len(phrase) >= 3:
                phrases.append(phrase)
        
        return phrases


# ============================================================================
# 停词发现器
# ============================================================================

class StopwordDiscoverer:
    """动态停词发现器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tfidf_threshold = self.config.get('tfidf_threshold', 0.1)
        self.frequency_threshold = self.config.get('frequency_threshold', 0.8)
    
    def discover_stopwords(self, phrase_corpus: List[List[str]]) -> set:
        """发现动态停词"""
        if not phrase_corpus:
            return set()
        
        # 计算词频
        phrase_freq = Counter()
        doc_freq = Counter()
        
        for doc_phrases in phrase_corpus:
            phrase_freq.update(doc_phrases)
            doc_freq.update(set(doc_phrases))
        
        total_docs = len(phrase_corpus)
        stopwords = set()
        
        # 基于文档频率发现停词
        for phrase, freq in doc_freq.items():
            doc_ratio = freq / total_docs
            if doc_ratio > self.frequency_threshold:
                stopwords.add(phrase)
        
        # 基于TF-IDF发现停词（简化版本）
        if SCIPY_AVAILABLE:
            for phrase, freq in phrase_freq.items():
                if phrase in doc_freq:
                    tf = freq / sum(phrase_freq.values())
                    idf = math.log(total_docs / doc_freq[phrase])
                    tfidf = tf * idf
                    
                    if tfidf < self.tfidf_threshold:
                        stopwords.add(phrase)
        
        return stopwords


# ============================================================================
# 图构建器
# ============================================================================

class GraphBuilder:
    """图构建器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preserve_isolated = self.config.get('preserve_isolated_nodes', True)
        self.min_cooccurrence = self.config.get('min_cooccurrence_count', 1)
    
    def build_global_graph(self, processed_docs: List[ProcessedDocument]) -> Dict[str, Any]:
        """构建全局共现图"""
        # 创建统一词表
        all_phrases = []
        for doc in processed_docs:
            all_phrases.extend(doc.phrases)
        
        unique_phrases = list(set(all_phrases))
        phrase_to_id = {phrase: i for i, phrase in enumerate(unique_phrases)}
        
        # 计算共现矩阵
        n_phrases = len(unique_phrases)
        if SCIPY_AVAILABLE:
            cooccurrence_matrix = scipy.sparse.lil_matrix((n_phrases, n_phrases))
        else:
            cooccurrence_matrix = [[0] * n_phrases for _ in range(n_phrases)]
        
        # 统计共现
        for doc in processed_docs:
            doc_phrases = doc.phrases
            for i, phrase1 in enumerate(doc_phrases):
                for j, phrase2 in enumerate(doc_phrases):
                    if i != j and phrase1 in phrase_to_id and phrase2 in phrase_to_id:
                        id1, id2 = phrase_to_id[phrase1], phrase_to_id[phrase2]
                        if SCIPY_AVAILABLE:
                            cooccurrence_matrix[id1, id2] += 1
                        else:
                            cooccurrence_matrix[id1][id2] += 1
        
        # 构建边表
        edges = []
        for i in range(n_phrases):
            for j in range(i + 1, n_phrases):  # 无向图，只考虑上三角
                if SCIPY_AVAILABLE:
                    weight = cooccurrence_matrix[i, j] + cooccurrence_matrix[j, i]
                else:
                    weight = cooccurrence_matrix[i][j] + cooccurrence_matrix[j][i]
                
                if weight >= self.min_cooccurrence:
                    edges.append({
                        'source': unique_phrases[i],
                        'target': unique_phrases[j],
                        'weight': weight,
                        'source_id': i,
                        'target_id': j
                    })
        
        return {
            'vocabulary': phrase_to_id,
            'reverse_vocabulary': {i: phrase for phrase, i in phrase_to_id.items()},
            'cooccurrence_matrix': cooccurrence_matrix,
            'edges': edges,
            'node_count': n_phrases,
            'edge_count': len(edges),
            'nodes': [{'id': i, 'label': phrase} for i, phrase in enumerate(unique_phrases)]
        }
    
    def build_state_subgraph(self, global_graph: Dict[str, Any], 
                           state_docs: List[ProcessedDocument], 
                           state_name: str) -> Dict[str, Any]:
        """构建州级子图"""
        # 收集该州的所有词组
        state_phrases = set()
        for doc in state_docs:
            state_phrases.update(doc.phrases)
        
        # 过滤全局图中的节点和边
        vocabulary = global_graph['vocabulary']
        active_nodes = []
        active_node_ids = set()
        
        for phrase in state_phrases:
            if phrase in vocabulary:
                node_id = vocabulary[phrase]
                active_nodes.append({'id': node_id, 'label': phrase})
                active_node_ids.add(node_id)
        
        # 过滤边
        active_edges = []
        for edge in global_graph['edges']:
            if edge['source_id'] in active_node_ids and edge['target_id'] in active_node_ids:
                active_edges.append(edge)
        
        return {
            'state_name': state_name,
            'nodes': active_nodes,
            'edges': active_edges,
            'node_count': len(active_nodes),
            'edge_count': len(active_edges),
            'vocabulary': {node['label']: node['id'] for node in active_nodes}
        }


# ============================================================================
# 输出管理器
# ============================================================================

class OutputManager:
    """输出管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.export_formats = self.config.get('export_formats', ['json', 'csv'])
        self.generate_viz = self.config.get('generate_visualizations', True)
    
    def generate_outputs(self, global_graph: Dict[str, Any], 
                        state_subgraphs: Dict[str, Dict[str, Any]], 
                        processed_docs: List[ProcessedDocument],
                        output_dir: str) -> List[str]:
        """生成所有输出文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        
        # 1. 导出全局图
        if 'json' in self.export_formats:
            global_json = output_path / 'global_graph.json'
            self._export_graph_json(global_graph, global_json)
            output_files.append(str(global_json))
        
        if 'csv' in self.export_formats:
            # 导出节点表
            nodes_csv = output_path / 'global_nodes.csv'
            self._export_nodes_csv(global_graph['nodes'], nodes_csv)
            output_files.append(str(nodes_csv))
            
            # 导出边表
            edges_csv = output_path / 'global_edges.csv'
            self._export_edges_csv(global_graph['edges'], edges_csv)
            output_files.append(str(edges_csv))
        
        # 2. 导出州级子图
        for state_name, subgraph in state_subgraphs.items():
            state_dir = output_path / f'states/{state_name}'
            state_dir.mkdir(parents=True, exist_ok=True)
            
            if 'json' in self.export_formats:
                state_json = state_dir / f'{state_name}_graph.json'
                self._export_graph_json(subgraph, state_json)
                output_files.append(str(state_json))
            
            if 'csv' in self.export_formats:
                state_nodes_csv = state_dir / f'{state_name}_nodes.csv'
                self._export_nodes_csv(subgraph['nodes'], state_nodes_csv)
                output_files.append(str(state_nodes_csv))
                
                state_edges_csv = state_dir / f'{state_name}_edges.csv'
                self._export_edges_csv(subgraph['edges'], state_edges_csv)
                output_files.append(str(state_edges_csv))
        
        # 3. 生成统计报告
        stats_file = output_path / 'analysis_statistics.json'
        self._generate_statistics_report(global_graph, state_subgraphs, processed_docs, stats_file)
        output_files.append(str(stats_file))
        
        # 4. 生成可视化（如果可用）
        if self.generate_viz and MATPLOTLIB_AVAILABLE:
            viz_files = self._generate_visualizations(global_graph, state_subgraphs, output_path)
            output_files.extend(viz_files)
        
        return output_files
    
    def _export_graph_json(self, graph: Dict[str, Any], file_path: Path):
        """导出图为JSON格式"""
        # 准备可序列化的数据
        export_data = {
            'nodes': graph['nodes'],
            'edges': graph['edges'],
            'node_count': graph['node_count'],
            'edge_count': graph['edge_count']
        }
        
        if 'state_name' in graph:
            export_data['state_name'] = graph['state_name']
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_nodes_csv(self, nodes: List[Dict[str, Any]], file_path: Path):
        """导出节点为CSV格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('id,label\n')
            for node in nodes:
                f.write(f"{node['id']},{node['label']}\n")
    
    def _export_edges_csv(self, edges: List[Dict[str, Any]], file_path: Path):
        """导出边为CSV格式"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('source,target,weight,source_id,target_id\n')
            for edge in edges:
                f.write(f"{edge['source']},{edge['target']},{edge['weight']},{edge['source_id']},{edge['target_id']}\n")
    
    def _generate_statistics_report(self, global_graph: Dict[str, Any], 
                                  state_subgraphs: Dict[str, Dict[str, Any]], 
                                  processed_docs: List[ProcessedDocument],
                                  file_path: Path):
        """生成统计报告"""
        stats = {
            'analysis_summary': {
                'total_documents': len(processed_docs),
                'total_states': len(state_subgraphs),
                'global_graph': {
                    'nodes': global_graph['node_count'],
                    'edges': global_graph['edge_count']
                }
            },
            'state_statistics': {},
            'document_statistics': {
                'by_language': {},
                'by_state': {}
            }
        }
        
        # 州级统计
        for state_name, subgraph in state_subgraphs.items():
            stats['state_statistics'][state_name] = {
                'nodes': subgraph['node_count'],
                'edges': subgraph['edge_count'],
                'density': subgraph['edge_count'] / (subgraph['node_count'] * (subgraph['node_count'] - 1) / 2) if subgraph['node_count'] > 1 else 0
            }
        
        # 文档统计
        lang_count = Counter(doc.language for doc in processed_docs)
        state_count = Counter(doc.original_doc.state for doc in processed_docs if doc.original_doc.state)
        
        stats['document_statistics']['by_language'] = dict(lang_count)
        stats['document_statistics']['by_state'] = dict(state_count)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    
    def _generate_visualizations(self, global_graph: Dict[str, Any], 
                               state_subgraphs: Dict[str, Dict[str, Any]], 
                               output_path: Path) -> List[str]:
        """生成可视化图像"""
        viz_files = []
        viz_dir = output_path / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        try:
            # 全局图统计可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # 节点度分布
            if global_graph['edges']:
                node_degrees = Counter()
                for edge in global_graph['edges']:
                    node_degrees[edge['source_id']] += 1
                    node_degrees[edge['target_id']] += 1
                
                degrees = list(node_degrees.values())
                ax1.hist(degrees, bins=20, alpha=0.7)
                ax1.set_xlabel('节点度')
                ax1.set_ylabel('频次')
                ax1.set_title('全局图节点度分布')
            
            # 州级子图对比
            if state_subgraphs:
                states = list(state_subgraphs.keys())
                node_counts = [subgraph['node_count'] for subgraph in state_subgraphs.values()]
                edge_counts = [subgraph['edge_count'] for subgraph in state_subgraphs.values()]
                
                x = range(len(states))
                width = 0.35
                
                ax2.bar([i - width/2 for i in x], node_counts, width, label='节点数', alpha=0.7)
                ax2.bar([i + width/2 for i in x], edge_counts, width, label='边数', alpha=0.7)
                ax2.set_xlabel('州/地区')
                ax2.set_ylabel('数量')
                ax2.set_title('各州子图规模对比')
                ax2.set_xticks(x)
                ax2.set_xticklabels(states, rotation=45)
                ax2.legend()
            
            plt.tight_layout()
            viz_file = viz_dir / 'network_statistics.png'
            plt.savefig(viz_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files.append(str(viz_file))
            
        except Exception as e:
            print(f"可视化生成失败: {e}")
        
        return viz_files


# ============================================================================
# 命令行接口
# ============================================================================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='语义共词网络分析器 - 整合版本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python semantic_coword_analyzer.py input_data/ output_results/
  python semantic_coword_analyzer.py input_data/ output_results/ --config config.json
  python semantic_coword_analyzer.py --help
        """
    )
    
    parser.add_argument('input_dir', help='输入目录路径（包含JSON文件）')
    parser.add_argument('output_dir', help='输出目录路径')
    parser.add_argument('--config', help='配置文件路径（JSON格式）')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    parser.add_argument('--version', action='version', version='语义共词网络分析器 v1.0.0')
    
    args = parser.parse_args()
    
    # 加载配置
    config = None
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            sys.exit(1)
    
    # 设置日志级别
    if args.verbose:
        if config is None:
            config = {}
        if 'logging' not in config:
            config['logging'] = {}
        config['logging']['level'] = 'DEBUG'
    
    try:
        # 创建分析器并运行
        analyzer = SemanticCowordAnalyzer(config)
        result = analyzer.analyze(args.input_dir, args.output_dir)
        
        # 输出结果摘要
        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)
        print(f"处理文档数: {result.processed_documents}/{result.total_documents}")
        print(f"抽取词组数: {result.total_phrases} (唯一: {result.unique_phrases})")
        print(f"全局图规模: {result.global_graph_nodes} 节点, {result.global_graph_edges} 边")
        print(f"州级子图数: {len(result.state_subgraphs)}")
        print(f"处理时间: {result.processing_time:.2f} 秒")
        print(f"输出文件数: {len(result.output_files)}")
        print(f"输出目录: {args.output_dir}")
        
        if result.state_subgraphs:
            print("\n州级子图详情:")
            for state, stats in result.state_subgraphs.items():
                print(f"  {state}: {stats['nodes']} 节点, {stats['edges']} 边")
        
        print("\n主要输出文件:")
        for file_path in result.output_files[:10]:  # 显示前10个文件
            print(f"  {file_path}")
        if len(result.output_files) > 10:
            print(f"  ... 还有 {len(result.output_files) - 10} 个文件")
        
    except Exception as e:
        print(f"分析失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()