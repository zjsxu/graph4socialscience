# Multi-View Co-Occurrence Graph Project Plan

## Project Overview

Build a modular system for constructing phrase-level co-occurrence graphs from text, creating multiple graph views (co-occurrence, semantic similarity), and performing graph fusion operations on top of EasyGraph.

## High-Level Architecture

```
Text Input → Phrase Extraction → Multi-View Graph Construction → Graph Fusion → Analysis
```

## Module 1: Text Processing and Phrase Extraction

### Components
- **TextPreprocessor**: Clean and normalize text data
- **PhraseExtractor**: Extract meaningful phrases using NLP techniques
- **CoOccurrenceAnalyzer**: Calculate phrase co-occurrence statistics

### Key Functions
```python
class TextPreprocessor:
    def clean_text(text: str) -> str
    def tokenize(text: str) -> List[str]
    def remove_stopwords(tokens: List[str]) -> List[str]

class PhraseExtractor:
    def extract_ngrams(text: str, n: int) -> List[str]
    def extract_named_entities(text: str) -> List[str]
    def extract_noun_phrases(text: str) -> List[str]

class CoOccurrenceAnalyzer:
    def calculate_cooccurrence_matrix(phrases: List[str], window_size: int) -> Dict
    def calculate_pmi(cooccurrence_matrix: Dict) -> Dict
```

## Module 2: Multi-View Graph Construction

### Components
- **GraphViewBuilder**: Abstract base for different view types
- **CoOccurrenceGraphBuilder**: Build co-occurrence graphs
- **SemanticGraphBuilder**: Build semantic similarity graphs
- **MultiViewGraphManager**: Manage multiple graph views

### Key Classes
```python
class GraphViewBuilder(ABC):
    @abstractmethod
    def build_graph(self, data: Any) -> eg.Graph

class CoOccurrenceGraphBuilder(GraphViewBuilder):
    def build_graph(self, cooccurrence_data: Dict) -> eg.Graph:
        # Create EasyGraph instance
        # Add phrases as nodes
        # Add co-occurrence edges with weights
        
class SemanticGraphBuilder(GraphViewBuilder):
    def __init__(self, embedding_model: str):
        # Initialize sentence transformer or word2vec model
        
    def build_graph(self, phrases: List[str]) -> eg.Graph:
        # Calculate semantic similarities
        # Create graph with similarity-based edges

class MultiViewGraphManager:
    def __init__(self):
        self.views: Dict[str, eg.Graph] = {}
        
    def add_view(self, name: str, graph: eg.Graph)
    def get_view(self, name: str) -> eg.Graph
    def list_views(self) -> List[str]
```

## Module 3: Graph Fusion Framework

### Components
- **GraphFusionStrategy**: Abstract fusion strategy interface
- **WeightedAverageFusion**: Combine graphs using weighted averages
- **ConsensusBasedFusion**: Find consensus edges across views
- **RankAggregationFusion**: Aggregate edge rankings from multiple views

### Key Classes
```python
class GraphFusionStrategy(ABC):
    @abstractmethod
    def fuse_graphs(self, graphs: Dict[str, eg.Graph]) -> eg.Graph

class WeightedAverageFusion(GraphFusionStrategy):
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        
    def fuse_graphs(self, graphs: Dict[str, eg.Graph]) -> eg.Graph:
        # Create new EasyGraph instance
        # Combine nodes from all views
        # Calculate weighted average of edge weights
        
class ConsensusBasedFusion(GraphFusionStrategy):
    def __init__(self, threshold: float):
        self.threshold = threshold
        
    def fuse_graphs(self, graphs: Dict[str, eg.Graph]) -> eg.Graph:
        # Keep edges that appear in multiple views
        # Weight by consensus strength

class GraphFuser:
    def __init__(self, strategy: GraphFusionStrategy):
        self.strategy = strategy
        
    def fuse(self, multi_view_manager: MultiViewGraphManager) -> eg.Graph
```

## Module 4: Analysis and Utilities

### Components
- **MultiViewAnalyzer**: Cross-view analysis utilities
- **GraphComparator**: Compare graphs across views
- **VisualizationHelper**: Visualization utilities for multi-view graphs

### Key Functions
```python
class MultiViewAnalyzer:
    def compare_centralities(self, graphs: Dict[str, eg.Graph]) -> Dict
    def find_view_specific_nodes(self, graphs: Dict[str, eg.Graph]) -> Dict
    def calculate_view_agreement(self, graphs: Dict[str, eg.Graph]) -> float

class GraphComparator:
    def jaccard_similarity(self, g1: eg.Graph, g2: eg.Graph) -> float
    def edge_overlap(self, g1: eg.Graph, g2: eg.Graph) -> float
    def node_ranking_correlation(self, g1: eg.Graph, g2: eg.Graph, metric: str) -> float
```

## Module 5: Pipeline Integration

### Components
- **CoOccurrencePipeline**: End-to-end pipeline class
- **ConfigurationManager**: Handle pipeline parameters
- **ResultsExporter**: Export results in various formats

### Main Pipeline Class
```python
class CoOccurrencePipeline:
    def __init__(self, config: Dict):
        self.text_processor = TextPreprocessor(config['preprocessing'])
        self.phrase_extractor = PhraseExtractor(config['phrase_extraction'])
        self.cooccurrence_analyzer = CoOccurrenceAnalyzer(config['cooccurrence'])
        self.view_builders = self._initialize_view_builders(config['views'])
        self.fusion_strategy = self._initialize_fusion_strategy(config['fusion'])
        
    def process_text(self, text: str) -> eg.Graph:
        # Step 1: Preprocess text
        clean_text = self.text_processor.clean_text(text)
        
        # Step 2: Extract phrases
        phrases = self.phrase_extractor.extract_phrases(clean_text)
        
        # Step 3: Build co-occurrence data
        cooccurrence_data = self.cooccurrence_analyzer.calculate_cooccurrence_matrix(phrases)
        
        # Step 4: Build multiple views
        multi_view_manager = MultiViewGraphManager()
        for view_name, builder in self.view_builders.items():
            if view_name == 'cooccurrence':
                graph = builder.build_graph(cooccurrence_data)
            elif view_name == 'semantic':
                graph = builder.build_graph(phrases)
            multi_view_manager.add_view(view_name, graph)
        
        # Step 5: Fuse graphs
        fused_graph = self.fusion_strategy.fuse_graphs(multi_view_manager.views)
        
        return fused_graph
```

## Implementation Strategy

### Phase 1: Core Infrastructure
1. Implement text preprocessing and phrase extraction
2. Build basic co-occurrence graph construction
3. Create multi-view graph manager
4. Develop simple fusion strategies

### Phase 2: Advanced Features
1. Add semantic similarity graph construction
2. Implement sophisticated fusion algorithms
3. Build analysis and comparison utilities
4. Add visualization capabilities

### Phase 3: Integration and Optimization
1. Create end-to-end pipeline
2. Add configuration management
3. Optimize performance for large datasets
4. Add comprehensive testing and documentation

## Key Design Principles

### Modularity
- Each component has a single responsibility
- Clear interfaces between modules
- Easy to swap implementations

### EasyGraph Integration
- Use EasyGraph for all core graph operations
- Leverage EasyGraph's performance optimizations
- Maintain compatibility with EasyGraph's ecosystem

### Extensibility
- Abstract base classes for easy extension
- Plugin architecture for new view types
- Configurable fusion strategies

### Performance Considerations
- Leverage EasyGraph's C++ backend where possible
- Use sparse representations for large graphs
- Implement caching for expensive operations
- Consider GPU acceleration for large-scale processing

This modular design allows for incremental development while building on EasyGraph's solid foundation, providing the multi-view and fusion capabilities that EasyGraph lacks natively.