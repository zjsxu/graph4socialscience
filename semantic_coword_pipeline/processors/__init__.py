"""
处理器模块

包含文本处理、词组抽取、停词发现等处理器组件。
"""

from .text_processor import (
    TextProcessor,
    LanguageDetector,
    EnglishTokenizer,
    ChineseTokenizer,
    LanguageDetectionResult
)

from .phrase_extractor import (
    PhraseExtractor,
    EnglishBigramExtractor,
    ChinesePhraseExtractor,
    StatisticalFilter,
    PhraseCandidate,
    StatisticalScores
)

from .dynamic_stopword_discoverer import (
    DynamicStopwordDiscoverer,
    TFIDFScore,
    StopwordDiscoveryResult
)

from .global_graph_builder import (
    GlobalGraphBuilder,
    CooccurrenceCalculator,
    create_empty_global_graph,
    merge_global_graphs
)

from .state_subgraph_activator import (
    StateSubgraphActivator,
    SubgraphComparator,
    create_empty_state_subgraph,
    merge_activation_masks
)

from .deterministic_layout_engine import (
    DeterministicLayoutEngine,
    LayoutParameters,
    VisualizationFilter,
    LayoutResult,
    PositionCache,
    ForceDirectedLayout,
    HierarchicalLayout
)

from .batch_processor import (
    BatchProcessor,
    BatchProcessingResult
)

from .output_manager import (
    OutputManager
)

from .easygraph_interface import (
    EasyGraphInterface,
    MultiViewGraph,
    FusionResult,
    GraphFormat,
    FusionStrategy,
    create_easygraph_from_matrix,
    validate_multi_view_consistency
)

from .document_generator import (
    DocumentGenerator,
    TraceabilityManager,
    TechnicalChoice,
    ProcessingStep,
    ComparisonMetrics,
    ExperimentTrace
)

__all__ = [
    'TextProcessor',
    'LanguageDetector', 
    'EnglishTokenizer',
    'ChineseTokenizer',
    'LanguageDetectionResult',
    'PhraseExtractor',
    'EnglishBigramExtractor',
    'ChinesePhraseExtractor',
    'StatisticalFilter',
    'PhraseCandidate',
    'StatisticalScores',
    'DynamicStopwordDiscoverer',
    'TFIDFScore',
    'StopwordDiscoveryResult',
    'GlobalGraphBuilder',
    'CooccurrenceCalculator',
    'create_empty_global_graph',
    'merge_global_graphs',
    'StateSubgraphActivator',
    'SubgraphComparator',
    'create_empty_state_subgraph',
    'merge_activation_masks',
    'DeterministicLayoutEngine',
    'LayoutParameters',
    'VisualizationFilter',
    'LayoutResult',
    'PositionCache',
    'ForceDirectedLayout',
    'HierarchicalLayout',
    'BatchProcessor',
    'BatchProcessingResult',
    'OutputManager',
    'EasyGraphInterface',
    'MultiViewGraph',
    'FusionResult',
    'GraphFormat',
    'FusionStrategy',
    'create_easygraph_from_matrix',
    'validate_multi_view_consistency',
    'DocumentGenerator',
    'TraceabilityManager',
    'TechnicalChoice',
    'ProcessingStep',
    'ComparisonMetrics',
    'ExperimentTrace'
]