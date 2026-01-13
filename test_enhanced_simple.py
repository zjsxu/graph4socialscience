#!/usr/bin/env python3
"""
Simple test for Enhanced Text Processor

This test demonstrates the enhanced text processing pipeline with
simpler, shorter text examples that work well with fallback processing.
"""

import json
import logging
from pathlib import Path
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from semantic_coword_pipeline.core.data_models import TOCDocument
from semantic_coword_pipeline.core.config import Config
from semantic_coword_pipeline.processors.enhanced_text_processor import EnhancedTextProcessor


def create_simple_documents() -> List[TOCDocument]:
    """Create simple test documents with shorter, cleaner text"""
    
    documents = [
        TOCDocument(
            segment_id="CA_001",
            title="Student Discipline",
            level=1,
            order=1,
            text="Student discipline policy. Academic standards. School district requirements.",
            state="California",
            language="english"
        ),
        TOCDocument(
            segment_id="CA_002", 
            title="Attendance Policy",
            level=1,
            order=2,
            text="Attendance requirements. School attendance. Student attendance policy.",
            state="California",
            language="english"
        ),
        TOCDocument(
            segment_id="TX_001",
            title="Academic Standards",
            level=1,
            order=1,
            text="Academic standards. Educational quality. Teacher certification requirements.",
            state="Texas",
            language="english"
        ),
        TOCDocument(
            segment_id="TX_002",
            title="Teacher Policy",
            level=1,
            order=2,
            text="Teacher certification. Professional development. Educational training programs.",
            state="Texas",
            language="english"
        ),
        TOCDocument(
            segment_id="NY_001",
            title="School Policy",
            level=1,
            order=1,
            text="School district policy. Educational standards. Academic requirements.",
            state="New York",
            language="english"
        ),
        TOCDocument(
            segment_id="NY_002",
            title="Student Requirements",
            level=1,
            order=2,
            text="Student requirements. Academic standards. School attendance policy.",
            state="New York",
            language="english"
        )
    ]
    
    return documents


def main():
    """Main test function"""
    print("Enhanced Text Processor - Simple Test")
    print("=" * 50)
    
    # Create sample documents
    documents = create_simple_documents()
    print(f"Created {len(documents)} sample documents")
    
    # Initialize configuration with adjusted settings for small dataset
    config = Config()
    
    # Adjust thresholds for small dataset
    config.set('enhanced_text_processing.df_threshold_ratio', 0.5)  # Lower threshold
    config.set('enhanced_text_processing.tfidf_threshold', 0.01)    # Lower threshold
    config.set('enhanced_text_processing.min_frequency_for_stopword', 2)  # Lower minimum
    config.set('enhanced_text_processing.max_phrase_length', 3)     # Shorter phrases
    
    # Initialize enhanced text processor
    processor = EnhancedTextProcessor(config)
    
    # Process documents
    print("\nProcessing documents through 6-step pipeline...")
    results = processor.process_documents(documents)
    
    # Display results
    print("\nProcessing Results:")
    print("-" * 30)
    
    metadata = results['processing_metadata']
    print(f"Total documents: {metadata['total_documents']}")
    print(f"Languages detected: {', '.join(metadata['languages_detected'])}")
    print(f"Initial phrase candidates: {metadata['total_candidates']}")
    print(f"After static filtering: {metadata['after_static_filtering']}")
    print(f"Dynamic stopwords identified: {metadata['dynamic_stopwords_count']}")
    print(f"Final phrases: {metadata['final_phrases_count']}")
    
    # Show final phrases
    print(f"\nFinal Phrases ({len(results['final_phrases'])}):")
    for i, phrase in enumerate(results['final_phrases'], 1):
        print(f"  {i:2d}. {phrase}")
    
    # Show dynamic stopwords if any
    if results['dynamic_stopwords']:
        print(f"\nDynamic Stopwords Identified:")
        for stopword in sorted(results['dynamic_stopwords']):
            explanation = results['dynamic_stopword_explanations'][stopword]
            print(f"  - '{stopword}' (TF-IDF: {explanation['tfidf']:.4f}, DF: {explanation['df']}/{metadata['total_documents']})")
    else:
        print(f"\nNo dynamic stopwords identified (thresholds: DF≥{config.get('enhanced_text_processing.df_threshold_ratio')}, TF-IDF<{config.get('enhanced_text_processing.tfidf_threshold')})")
    
    # Show phrase statistics
    print(f"\nTop Phrases by TF-IDF:")
    stats_items = sorted(results['corpus_statistics'].items(), 
                        key=lambda x: x[1].tfidf, reverse=True)[:10]
    for phrase, stat in stats_items:
        print(f"  '{phrase}': TF={stat.tf}, DF={stat.df}, TF-IDF={stat.tfidf:.4f}")
    
    # Show phrase-to-segments mapping
    print(f"\nPhrase Distribution (first 5):")
    for i, (phrase, segments) in enumerate(list(results['phrase_to_segments'].items())[:5]):
        states = set()
        for segment in segments:
            for doc in documents:
                if doc.segment_id == segment:
                    states.add(doc.state)
        print(f"  '{phrase}': {len(segments)} segments, {len(states)} states")
    
    # Save results
    output_dir = "output/enhanced_simple_test"
    print(f"\nSaving results to {output_dir}...")
    processor.save_results(results, output_dir)
    
    print("\nTest completed successfully!")
    print(f"Check {output_dir} for detailed results.")
    
    # Show configuration used
    print(f"\nConfiguration Used:")
    print(f"  DF threshold ratio: {config.get('enhanced_text_processing.df_threshold_ratio')}")
    print(f"  TF-IDF threshold: {config.get('enhanced_text_processing.tfidf_threshold')}")
    print(f"  Min frequency for stopword: {config.get('enhanced_text_processing.min_frequency_for_stopword')}")
    print(f"  Max phrase length: {config.get('enhanced_text_processing.max_phrase_length')}")


if __name__ == "__main__":
    main()