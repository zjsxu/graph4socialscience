#!/usr/bin/env python3
"""
Test script for the Enhanced Text Processor

This script demonstrates the 6-step linguistically and statistically grounded
text processing pipeline for semantic co-occurrence network analysis.
"""

import json
import logging
from pathlib import Path
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from semantic_coword_pipeline.core.data_models import TOCDocument
from semantic_coword_pipeline.core.config import Config
from semantic_coword_pipeline.processors.enhanced_text_processor import EnhancedTextProcessor


def create_sample_documents() -> List[TOCDocument]:
    """Create sample TOC documents for testing"""
    
    # English policy documents
    english_docs = [
        TOCDocument(
            segment_id="CA_001",
            title="Student Discipline Policy",
            level=1,
            order=1,
            text="The student discipline policy establishes clear guidelines for maintaining classroom order. "
                 "Natural language processing techniques help analyze policy effectiveness. "
                 "School district administrators must ensure consistent policy implementation across all schools.",
            state="California",
            language="english"
        ),
        TOCDocument(
            segment_id="CA_002", 
            title="Attendance Requirements",
            level=1,
            order=2,
            text="Student attendance requirements are essential for academic success. "
                 "The attendance policy defines mandatory school days and excused absences. "
                 "School district officials monitor attendance patterns using data analysis tools.",
            state="California",
            language="english"
        ),
        TOCDocument(
            segment_id="TX_001",
            title="Academic Standards",
            level=1,
            order=1,
            text="Academic standards ensure quality education delivery across the state. "
                 "The academic policy framework guides curriculum development and assessment procedures. "
                 "Educational institutions must align their programs with state academic requirements.",
            state="Texas",
            language="english"
        ),
        TOCDocument(
            segment_id="TX_002",
            title="Teacher Certification",
            level=1,
            order=2,
            text="Teacher certification requirements maintain educational quality standards. "
                 "The certification process includes background checks and competency assessments. "
                 "Professional development programs support ongoing teacher education and training.",
            state="Texas",
            language="english"
        )
    ]
    
    # Chinese policy documents (if needed)
    chinese_docs = [
        TOCDocument(
            segment_id="CN_001",
            title="学生纪律政策",
            level=1,
            order=1,
            text="学生纪律政策建立了维护课堂秩序的明确指导原则。自然语言处理技术有助于分析政策有效性。"
                 "学区管理人员必须确保所有学校政策实施的一致性。教育质量标准对于学生发展至关重要。",
            state="Beijing",
            language="chinese"
        ),
        TOCDocument(
            segment_id="CN_002",
            title="出勤要求",
            level=1,
            order=2,
            text="学生出勤要求对学业成功至关重要。出勤政策定义了强制上学日和准假缺勤。"
                 "学区官员使用数据分析工具监控出勤模式。教育管理系统支持学生信息管理。",
            state="Beijing",
            language="chinese"
        )
    ]
    
    return english_docs + chinese_docs


def main():
    """Main test function"""
    print("Enhanced Text Processor Test")
    print("=" * 50)
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"Created {len(documents)} sample documents")
    
    # Initialize configuration
    config = Config()
    
    # Override some settings for testing
    config.set('enhanced_text_processing.df_threshold_ratio', 0.6)  # Lower threshold for small dataset
    config.set('enhanced_text_processing.tfidf_threshold', 0.05)    # Lower threshold for testing
    config.set('enhanced_text_processing.min_frequency_for_stopword', 2)  # Lower for small dataset
    
    # Initialize enhanced text processor
    processor = EnhancedTextProcessor(config)
    
    # Process documents through the 6-step pipeline
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
    
    # Show some examples
    print(f"\nSample Final Phrases (first 10):")
    for i, phrase in enumerate(results['final_phrases'][:10]):
        print(f"  {i+1}. {phrase}")
    
    print(f"\nDynamic Stopwords Identified:")
    for stopword in sorted(results['dynamic_stopwords']):
        print(f"  - {stopword}")
    
    # Show some corpus statistics
    print(f"\nSample Corpus Statistics:")
    stats_items = list(results['corpus_statistics'].items())[:5]
    for phrase, stat in stats_items:
        print(f"  '{phrase}': TF={stat.tf}, DF={stat.df}, TF-IDF={stat.tfidf:.4f}")
    
    # Save results
    output_dir = "output/enhanced_processing_test"
    print(f"\nSaving results to {output_dir}...")
    processor.save_results(results, output_dir)
    
    print("\nTest completed successfully!")
    print(f"Check {output_dir} for detailed results.")
    
    # Demonstrate step-by-step breakdown
    print("\n" + "="*50)
    print("STEP-BY-STEP BREAKDOWN")
    print("="*50)
    
    print("\nStep 1: Linguistic Preprocessing")
    print("-" * 30)
    sample_doc = documents[0]
    lang_processor = processor.language_processor
    language = lang_processor.detect_language(sample_doc.text)
    preprocessing_result = lang_processor.linguistic_preprocessing(sample_doc.text, language)
    
    print(f"Original text: {sample_doc.text[:100]}...")
    print(f"Detected language: {language}")
    print(f"Cleaned text: {preprocessing_result['cleaned_text'][:100]}...")
    print(f"Tokens (first 10): {preprocessing_result['tokens'][:10]}")
    print(f"POS tags (first 10): {preprocessing_result['pos_tags'][:10]}")
    
    print("\nStep 2: Phrase Candidate Extraction")
    print("-" * 30)
    candidates = processor.phrase_extractor.extract_phrase_candidates(
        preprocessing_result, sample_doc.segment_id, sample_doc.state, language
    )
    print(f"Extracted {len(candidates)} phrase candidates:")
    for i, candidate in enumerate(candidates[:5]):
        print(f"  {i+1}. '{candidate.text}' (tokens: {candidate.tokens}, POS: {candidate.pos_tags})")
    
    print("\nStep 3: Static Stopword Filtering")
    print("-" * 30)
    filtered = processor.static_filter.filter_candidates(candidates, language)
    removed = len(candidates) - len(filtered)
    print(f"Removed {removed} candidates with static stopwords")
    print(f"Sample static stopwords: {list(processor.static_filter.english_stopwords)[:10]}")
    
    print("\nStep 4: Corpus Statistics")
    print("-" * 30)
    sample_stats = list(results['corpus_statistics'].items())[:3]
    for phrase, stat in sample_stats:
        print(f"  '{phrase}':")
        print(f"    TF: {stat.tf} (total frequency)")
        print(f"    DF: {stat.df} (document frequency)")
        print(f"    IDF: {stat.idf:.4f} (inverse document frequency)")
        print(f"    TF-IDF: {stat.tfidf:.4f}")
        print(f"    States: {list(stat.states)}")
    
    print("\nStep 5: Dynamic Stopword Identification")
    print("-" * 30)
    print(f"Threshold settings:")
    print(f"  DF ratio threshold: {processor.dynamic_identifier.df_threshold_ratio}")
    print(f"  TF-IDF threshold: {processor.dynamic_identifier.tfidf_threshold}")
    print(f"  Min frequency: {processor.dynamic_identifier.min_frequency}")
    
    if results['dynamic_stopwords']:
        print(f"Dynamic stopwords found:")
        for stopword in results['dynamic_stopwords']:
            explanation = results['dynamic_stopword_explanations'][stopword]
            print(f"  '{stopword}': TF-IDF={explanation['tfidf']:.4f}, DF_ratio={explanation['df_ratio']:.3f}")
    else:
        print("No dynamic stopwords identified (dataset too small or thresholds too strict)")
    
    print("\nStep 6: Final Filtering")
    print("-" * 30)
    print(f"Combined {len(processor.static_filter.english_stopwords)} English + {len(processor.static_filter.chinese_stopwords)} Chinese static stopwords")
    print(f"Plus {len(results['dynamic_stopwords'])} dynamic stopwords")
    print(f"Final result: {len(results['final_phrases'])} clean phrases")


if __name__ == "__main__":
    main()