# Enhanced Text Processing Guide

## Overview

This guide describes the enhanced text processing pipeline for semantic co-occurrence network analysis. The pipeline implements a linguistically and statistically grounded approach following academic NLP best practices.

## 6-Step Processing Pipeline

### Step 1: Linguistic Preprocessing (NO stopword removal yet)

**Purpose**: Clean and analyze text while preserving linguistic structure needed for parsing.

**Technology**: spaCy for tokenization, POS tagging, dependency parsing

**Process**:
- Clean text (HTML tags, special characters)
- Sentence segmentation
- Tokenization
- POS tagging
- Dependency parsing
- **Keep stopwords** (needed for accurate parsing)

**Output**: Cleaned text with full linguistic annotations

### Step 2: Phrase/Keyphrase Candidate Extraction

**Purpose**: Generate linguistically meaningful phrase-level node candidates.

**Technology**: spaCy Matcher with rule-based patterns

**Extraction Rules**:
- English: `(ADJ)*(NOUN)+` patterns
- Dependency relations: `amod` (adjectival modifier), `compound`
- Chinese: Multi-character noun phrases after segmentation

**Examples of Valid Candidates**:
- "natural language processing"
- "student discipline policy" 
- "attendance requirement"

**Filtering**:
- No single stopwords
- No pure function phrases
- No single punctuation/numeric tokens
- Must be linguistically meaningful

### Step 3: Static Stopword Filtering (Lightweight)

**Purpose**: Remove obvious stopwords while avoiding over-filtering.

**Static Stopword Lists**:
- English: articles, prepositions, pronouns, auxiliary verbs
- Chinese: common function words (的, 了, 在, 是, etc.)

**Rules**:
- Remove phrases consisting entirely of stopwords
- Remove phrases shorter than 2 tokens (unless justified)
- **Conservative approach** - don't over-filter

### Step 4: Corpus-level Statistics

**Purpose**: Calculate statistical measures for all phrases across the corpus.

**Metrics Calculated**:
- **TF**: Total frequency across corpus
- **DF**: Number of segments containing the phrase  
- **IDF**: log(N / (DF + 1))
- **TF-IDF**: TF × IDF

**Output**: Structured statistics table for all phrases

### Step 5: Dynamic Stopword Identification (Core Requirement)

**Purpose**: Automatically identify high-frequency, low-discriminative phrases.

**Criteria for Dynamic Stopwords**:
1. **High Document Frequency**: Appears in large proportion of segments (DF/N ≥ threshold)
2. **Low TF-IDF**: Below corpus distribution threshold
3. **Sufficient Frequency**: Minimum occurrence count

**Examples of Dynamic Stopwords**:
- "general policy requirement"
- "in accordance with"
- "school district shall"

**Output**: 
- `dynamic_stopwords.txt` with statistics
- Traceability information for each decision

### Step 6: Final Phrase Filtering

**Purpose**: Apply combined stopword filtering for final clean phrase list.

**Process**:
1. Combine static + dynamic stopwords
2. Filter phrase candidates
3. Generate phrase → segment mappings
4. Produce final statistics

**Outputs**:
- Cleaned phrase list
- Phrase → segment mappings
- Full statistics (TF, DF, TF-IDF)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements_enhanced.txt
```

### 2. Install spaCy Language Models

```bash
# English model
python -m spacy download en_core_web_sm

# Chinese model  
python -m spacy download zh_core_web_sm
```

### 3. Verify Installation

```bash
python -c "import spacy; print('spaCy version:', spacy.__version__)"
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('English model loaded')"
```

## Usage

### Basic Usage

```python
from semantic_coword_pipeline.core.config import Config
from semantic_coword_pipeline.core.data_models import TOCDocument
from semantic_coword_pipeline.processors.enhanced_text_processor import EnhancedTextProcessor

# Initialize configuration
config = Config()

# Optional: Adjust thresholds
config.set('enhanced_text_processing.df_threshold_ratio', 0.8)
config.set('enhanced_text_processing.tfidf_threshold', 0.1)
config.set('enhanced_text_processing.min_frequency_for_stopword', 5)

# Initialize processor
processor = EnhancedTextProcessor(config)

# Create documents (from your TOC-segmented JSON)
documents = [
    TOCDocument(
        segment_id="CA_001",
        title="Student Policy",
        text="Your policy text here...",
        state="California"
    ),
    # ... more documents
]

# Process through 6-step pipeline
results = processor.process_documents(documents)

# Save results
processor.save_results(results, "output/enhanced_processing")
```

### Configuration Options

```python
# Enhanced text processing configuration
config = {
    'enhanced_text_processing': {
        # Phrase extraction
        'min_phrase_length': 2,
        'max_phrase_length': 4,
        
        # Static stopwords
        'english_stopwords_file': 'path/to/english_stopwords.txt',
        'chinese_stopwords_file': 'path/to/chinese_stopwords.txt',
        
        # Dynamic stopword thresholds
        'df_threshold_ratio': 0.8,      # High frequency threshold
        'tfidf_threshold': 0.1,         # Low discrimination threshold  
        'min_frequency_for_stopword': 5, # Minimum frequency
        
        # spaCy settings
        'use_spacy': True,
        'spacy_models': {
            'english': 'en_core_web_sm',
            'chinese': 'zh_core_web_sm'
        }
    }
}
```

### Output Files

The processor generates several output files:

1. **`dynamic_stopwords.json`**: Dynamic stopwords with explanations
2. **`dynamic_stopwords.txt`**: Simple text list of dynamic stopwords
3. **`corpus_statistics.json`**: TF-IDF statistics for all phrases
4. **`final_phrases.json`**: Clean phrase list with mappings
5. **`processing_summary.txt`**: Human-readable summary

### Results Structure

```python
results = {
    'preprocessing_results': [...],      # Step 1 results
    'phrase_candidates': [...],          # Step 2 results  
    'filtered_candidates': [...],        # Step 3 results
    'corpus_statistics': {...},          # Step 4 results
    'dynamic_stopwords': {...},          # Step 5 results
    'final_phrases': [...],              # Step 6 results
    'phrase_to_segments': {...},         # Phrase mappings
    'processing_metadata': {...}         # Summary statistics
}
```

## Testing

Run the test script to verify everything works:

```bash
python test_enhanced_text_processor.py
```

This will:
- Create sample documents
- Process through all 6 steps
- Display results and statistics
- Save output files
- Show step-by-step breakdown

## Integration with Existing Pipeline

The enhanced text processor is designed to replace or augment the existing text processing components:

### Replace Current Components

```python
# Instead of:
from semantic_coword_pipeline.processors.text_processor import TextProcessor
from semantic_coword_pipeline.processors.phrase_extractor import PhraseExtractor

# Use:
from semantic_coword_pipeline.processors.enhanced_text_processor import EnhancedTextProcessor
```

### Integration Points

1. **Input**: Same TOCDocument format
2. **Output**: Compatible with existing graph construction
3. **Configuration**: Extends existing config system
4. **Logging**: Uses existing logging framework

## Key Differences from Original Implementation

### Linguistic Grounding
- **Original**: Basic tokenization with NLTK/jieba
- **Enhanced**: Full linguistic analysis with spaCy (POS, dependencies)

### Phrase Extraction  
- **Original**: Simple n-gram generation
- **Enhanced**: Rule-based extraction with linguistic patterns

### Stopword Handling
- **Original**: Static lists only
- **Enhanced**: Dynamic discovery based on TF-IDF statistics

### Academic Rigor
- **Original**: Practical approach
- **Enhanced**: Follows academic NLP best practices

### Reproducibility
- **Original**: Basic determinism
- **Enhanced**: Full traceability and explanation

## Troubleshooting

### spaCy Model Issues

```bash
# If models not found:
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm

# Check installed models:
python -m spacy info
```

### Memory Issues with Large Datasets

```python
# Process in batches
config.set('performance.batch_size', 500)
config.set('performance.max_workers', 2)
```

### No Dynamic Stopwords Found

This is normal for small datasets. Adjust thresholds:

```python
config.set('enhanced_text_processing.df_threshold_ratio', 0.5)  # Lower threshold
config.set('enhanced_text_processing.min_frequency_for_stopword', 2)  # Lower minimum
```

## Performance Considerations

- **spaCy Processing**: More thorough but slower than basic tokenization
- **Memory Usage**: Keeps linguistic annotations in memory
- **Scalability**: Designed for document collections up to 10K segments
- **Caching**: Results can be cached for repeated analysis

## Academic Compliance

This implementation follows academic NLP standards:

- **Reproducible**: Fixed algorithms, no randomness in core processing
- **Interpretable**: Full explanation of stopword decisions
- **Lightweight**: No deep learning dependencies
- **Traceable**: Complete audit trail of processing steps

The pipeline is suitable for academic publication and meets the requirements for computational social science research.