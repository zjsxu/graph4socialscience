"""
文本处理器模块

实现文本预处理、语言检测、分词和规范化功能。
根据需求5.3和3.4，支持英文和中文的文本处理流程。
"""

import re
import string
from typing import Dict, List, Any, Optional
import logging
from tqdm import tqdm
from dataclasses import dataclass

# 导入NLTK和jieba
try:
    import nltk
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK not available, falling back to basic tokenization")

try:
    import jieba
    import jieba.posseg as pseg
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    logging.warning("jieba not available, Chinese processing will be limited")

from ..core.data_models import TOCDocument, ProcessedDocument, Window
from ..core.config import Config
from ..core.error_handler import ErrorHandler


@dataclass
class LanguageDetectionResult:
    """语言检测结果"""
    language: str
    confidence: float
    detected_features: Dict[str, Any]


class LanguageDetector:
    """
    语言检测器
    
    使用简单的启发式规则检测文本语言（英文/中文）。
    """
    
    def __init__(self):
        # 中文字符范围
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        # 英文字符范围
        self.english_pattern = re.compile(r'[a-zA-Z]')
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        检测文本语言
        
        Args:
            text: 输入文本
            
        Returns:
            LanguageDetectionResult: 检测结果
        """
        if not text or not text.strip():
            return LanguageDetectionResult(
                language='unknown',
                confidence=0.0,
                detected_features={}
            )
        
        # 统计中文和英文字符数量
        chinese_chars = len(self.chinese_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        total_chars = len(text.strip())
        
        # 计算比例
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        english_ratio = english_chars / total_chars if total_chars > 0 else 0
        
        # 检测特征
        features = {
            'chinese_chars': chinese_chars,
            'english_chars': english_chars,
            'total_chars': total_chars,
            'chinese_ratio': chinese_ratio,
            'english_ratio': english_ratio
        }
        
        # 语言判断逻辑
        if chinese_ratio > 0.1:  # 如果中文字符超过10%，认为是中文
            language = 'chinese'
            confidence = min(chinese_ratio * 2, 1.0)
        elif english_ratio > 0.3:  # 如果英文字符超过30%，认为是英文
            language = 'english'
            confidence = min(english_ratio * 1.5, 1.0)
        else:
            # 默认判断为英文
            language = 'english'
            confidence = 0.5
        
        return LanguageDetectionResult(
            language=language,
            confidence=confidence,
            detected_features=features
        )


class EnglishTokenizer:
    """
    英文分词器
    
    使用NLTK进行英文文本的分词处理。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_stemming = config.get('use_stemming', False)
        self.remove_stopwords = config.get('remove_stopwords', False)
        
        # 初始化NLTK组件
        if NLTK_AVAILABLE:
            try:
                # 确保必要的NLTK数据已下载
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logging.info("Downloading NLTK punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                logging.info("Downloading NLTK stopwords...")
                nltk.download('stopwords', quiet=True)
            
            self.stemmer = PorterStemmer() if self.use_stemming else None
            self.stop_words = set(stopwords.words('english')) if self.remove_stopwords else set()
        else:
            self.stemmer = None
            self.stop_words = set()
    
    def tokenize(self, text: str) -> List[str]:
        """
        英文分词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        if not text or not text.strip():
            return []
        
        if NLTK_AVAILABLE:
            try:
                # 使用NLTK分词
                tokens = word_tokenize(text.lower())
            except Exception as e:
                logging.warning(f"NLTK tokenization failed: {e}, falling back to simple split")
                tokens = self._simple_tokenize(text)
        else:
            tokens = self._simple_tokenize(text)
        
        # 过滤和处理
        processed_tokens = []
        for token in tokens:
            # 跳过标点符号和空白
            if token in string.punctuation or not token.strip():
                continue
            
            # 跳过停词
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue
            
            # 词干提取
            if self.stemmer:
                token = self.stemmer.stem(token)
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """
        简单分词（后备方案）
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        # 移除标点符号并按空格分割
        text = re.sub(r'[^\w\s]', ' ', text)
        return [token.strip().lower() for token in text.split() if token.strip()]


class ChineseTokenizer:
    """
    中文分词器
    
    使用jieba进行中文文本的分词处理。
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_pos_tagging = config.get('use_pos_tagging', False)
        self.remove_stopwords = config.get('remove_stopwords', False)
        
        # 中文字符模式
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        
        # 初始化jieba
        if JIEBA_AVAILABLE:
            # 设置jieba日志级别
            jieba.setLogLevel(logging.WARNING)
            
            # 加载用户词典（如果有）
            user_dict_path = config.get('user_dict_path')
            if user_dict_path:
                try:
                    jieba.load_userdict(user_dict_path)
                    logging.info(f"Loaded user dictionary: {user_dict_path}")
                except Exception as e:
                    logging.warning(f"Failed to load user dictionary: {e}")
        
        # 中文停词（基础集合）
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
            '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
            '自己', '这', '那', '里', '就是', '还', '把', '比', '或者', '等', '但是', '如果'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """
        中文分词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        if not text or not text.strip():
            return []
        
        if JIEBA_AVAILABLE:
            try:
                if self.use_pos_tagging:
                    # 使用词性标注
                    words = pseg.cut(text)
                    tokens = [word for word, pos in words if self._is_valid_pos(pos)]
                else:
                    # 普通分词
                    tokens = list(jieba.cut(text))
            except Exception as e:
                logging.warning(f"jieba tokenization failed: {e}, falling back to simple split")
                tokens = self._simple_tokenize(text)
        else:
            tokens = self._simple_tokenize(text)
        
        # 过滤和处理
        processed_tokens = []
        for token in tokens:
            token = token.strip()
            
            # 跳过空白和纯标点符号
            if not token or token in '，。！？；：""''（）【】':
                continue
            
            # 跳过单字符（除非是中文字符）
            if len(token) == 1 and not self.chinese_pattern.match(token):
                continue
            
            # 跳过纯数字
            if token.isdigit():
                continue
            
            # 跳过停词
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    def _is_valid_pos(self, pos: str) -> bool:
        """
        检查词性是否有效
        
        Args:
            pos: 词性标签
            
        Returns:
            bool: 是否有效
        """
        # 保留名词、动词、形容词等实词
        valid_pos = {'n', 'v', 'a', 'nr', 'ns', 'nt', 'nz', 'vn', 'an'}
        return pos and any(pos.startswith(p) for p in valid_pos)
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """
        简单中文分词（后备方案）
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        # 按标点符号分割，然后按字符分割
        sentences = re.split(r'[。！？；，、]', text)
        tokens = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # 简单按字符分割（不理想，但作为后备）
                chars = list(sentence)
                tokens.extend([char for char in chars if char.strip() and not char.isspace()])
        
        return tokens


class TextProcessor:
    """
    文本处理器主类
    
    根据需求5.3和3.4实现文本预处理、语言检测、分词和规范化功能。
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        初始化文本处理器
        
        Args:
            config: 配置对象
        """
        self.config = config or Config()
        self.error_handler = ErrorHandler(self.config.get_section('error_handling'))
        
        # 初始化组件
        self.language_detector = LanguageDetector()
        
        # 获取文本处理配置
        text_config = self.config.get_section('text_processing')
        
        self.english_tokenizer = EnglishTokenizer(text_config)
        self.chinese_tokenizer = ChineseTokenizer(text_config)
        
        # 处理选项
        self.normalize_text = text_config.get('normalize_text', True)
        self.remove_punctuation = text_config.get('remove_punctuation', True)
        self.convert_to_lowercase = text_config.get('convert_to_lowercase', True)
        
        logging.info("TextProcessor initialized successfully")
    
    def process_document(self, doc: TOCDocument) -> ProcessedDocument:
        """
        处理单个TOC文档
        
        Args:
            doc: TOC文档
            
        Returns:
            ProcessedDocument: 处理后的文档
        """
        try:
            # 检测语言
            if not doc.language:
                detection_result = self.detect_language(doc.text)
                doc.language = detection_result.language
                logging.debug(f"Detected language for document {doc.segment_id}: {doc.language}")
            
            # 文本规范化
            cleaned_text = self.normalize_text_content(doc.text, doc.language)
            
            # 分词
            tokens = self.tokenize(cleaned_text, doc.language)
            
            # 创建窗口（每个segment作为一个窗口）
            window = Window(
                window_id=f"{doc.segment_id}_window",
                phrases=tokens,  # 这里先用tokens，后续会被词组替换
                source_doc=doc.segment_id,
                state=doc.state or 'unknown',
                segment_id=doc.segment_id
            )
            
            # 创建处理后的文档
            processed_doc = ProcessedDocument(
                original_doc=doc,
                cleaned_text=cleaned_text,
                tokens=tokens,
                phrases=[],  # 词组抽取在后续步骤进行
                windows=[window],
                processing_metadata={
                    'language': doc.language,
                    'token_count': len(tokens),
                    'original_text_length': len(doc.text),
                    'cleaned_text_length': len(cleaned_text)
                }
            )
            
            logging.debug(f"Processed document {doc.segment_id}: {len(tokens)} tokens")
            return processed_doc
            
        except Exception as e:
            error_msg = f"Failed to process document {doc.segment_id}: {e}"
            logging.error(error_msg)
            return self.error_handler.handle_processing_error(e, f"process_document:{doc.segment_id}")
    
    def detect_language(self, text: str) -> LanguageDetectionResult:
        """
        检测文本语言
        
        Args:
            text: 输入文本
            
        Returns:
            LanguageDetectionResult: 检测结果
        """
        try:
            return self.language_detector.detect_language(text)
        except Exception as e:
            logging.warning(f"Language detection failed: {e}, defaulting to English")
            return LanguageDetectionResult(
                language='english',
                confidence=0.5,
                detected_features={}
            )
    
    def normalize_text_content(self, text: str, language: str) -> str:
        """
        文本规范化处理
        
        Args:
            text: 输入文本
            language: 文本语言
            
        Returns:
            str: 规范化后的文本
        """
        if not text:
            return ""
        
        try:
            # 基础清理
            normalized = text.strip()
            
            # 移除多余的空白字符
            normalized = re.sub(r'\s+', ' ', normalized)
            
            # 根据语言进行特定处理
            if language == 'english':
                normalized = self._normalize_english_text(normalized)
            elif language == 'chinese':
                normalized = self._normalize_chinese_text(normalized)
            
            return normalized
            
        except Exception as e:
            logging.warning(f"Text normalization failed: {e}, returning original text")
            return text
    
    def tokenize(self, text: str, language: str) -> List[str]:
        """
        分词处理
        
        Args:
            text: 输入文本
            language: 文本语言
            
        Returns:
            List[str]: 分词结果
        """
        if not text:
            return []
        
        try:
            if language == 'english':
                return self.english_tokenizer.tokenize(text)
            elif language == 'chinese':
                return self.chinese_tokenizer.tokenize(text)
            else:
                # 默认使用英文分词
                logging.warning(f"Unknown language {language}, using English tokenizer")
                return self.english_tokenizer.tokenize(text)
                
        except Exception as e:
            logging.error(f"Tokenization failed: {e}")
            # 后备方案：简单分割
            return text.split()
    
    def _normalize_english_text(self, text: str) -> str:
        """
        英文文本规范化
        
        Args:
            text: 输入文本
            
        Returns:
            str: 规范化后的文本
        """
        # 转换为小写
        if self.convert_to_lowercase:
            text = text.lower()
        
        # 移除或替换特殊字符
        if self.remove_punctuation:
            # 保留句号、逗号等基本标点，移除其他特殊字符
            text = re.sub(r'[^\w\s.,;:!?-]', ' ', text)
        
        # 处理缩写
        text = re.sub(r"'s\b", ' is', text)
        text = re.sub(r"'re\b", ' are', text)
        text = re.sub(r"'ve\b", ' have', text)
        text = re.sub(r"'ll\b", ' will', text)
        text = re.sub(r"'d\b", ' would', text)
        text = re.sub(r"n't\b", ' not', text)
        
        # 规范化空白
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _normalize_chinese_text(self, text: str) -> str:
        """
        中文文本规范化
        
        Args:
            text: 输入文本
            
        Returns:
            str: 规范化后的文本
        """
        # 移除或替换特殊字符
        if self.remove_punctuation:
            # 保留中文标点符号
            text = re.sub(r'[^\u4e00-\u9fff\w\s。，、；：！？（）【】""''—…]', ' ', text)
        
        # 统一标点符号
        text = text.replace('，', ',')
        text = text.replace('。', '.')
        text = text.replace('！', '!')
        text = text.replace('？', '?')
        text = text.replace('；', ';')
        text = text.replace('：', ':')
        
        # 规范化空白
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def batch_process_documents(self, documents: List[TOCDocument]) -> List[ProcessedDocument]:
        """
        批量处理文档
        
        Args:
            documents: 文档列表
            
        Returns:
            List[ProcessedDocument]: 处理后的文档列表
        """
        processed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                processed_doc = self.process_document(doc)
                processed_docs.append(processed_doc)
                
                if (i + 1) % 100 == 0:
                    logging.info(f"Processed {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                logging.error(f"Failed to process document {doc.segment_id}: {e}")
                # 继续处理其他文档
                continue
        
        logging.info(f"Batch processing completed: {len(processed_docs)}/{len(documents)} documents processed")
        return processed_docs
    
    def get_processing_statistics(self, processed_docs: List[ProcessedDocument]) -> Dict[str, Any]:
        """
        获取处理统计信息
        
        Args:
            processed_docs: 处理后的文档列表
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not processed_docs:
            return {}
        
        # 语言分布
        language_counts = {}
        total_tokens = 0
        total_original_length = 0
        total_cleaned_length = 0
        
        for doc in tqdm(processed_docs, desc="📄 Processing documents", unit="doc"):
            lang = doc.processing_metadata.get('language', 'unknown')
            language_counts[lang] = language_counts.get(lang, 0) + 1
            
            total_tokens += doc.processing_metadata.get('token_count', 0)
            total_original_length += doc.processing_metadata.get('original_text_length', 0)
            total_cleaned_length += doc.processing_metadata.get('cleaned_text_length', 0)
        
        return {
            'total_documents': len(processed_docs),
            'language_distribution': language_counts,
            'total_tokens': total_tokens,
            'average_tokens_per_doc': total_tokens / len(processed_docs),
            'total_original_length': total_original_length,
            'total_cleaned_length': total_cleaned_length,
            'text_reduction_ratio': 1 - (total_cleaned_length / total_original_length) if total_original_length > 0 else 0
        }