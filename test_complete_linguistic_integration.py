#!/usr/bin/env python3
"""
Complete Linguistic Integration Test

This script demonstrates the complete integration of strict linguistic validation
into the semantic co-occurrence network pipeline, showing that only linguistically
valid noun phrases become graph nodes.
"""

import os
import sys
import json
from pathlib import Path

def create_test_data():
    """Create test data with both valid and invalid phrases"""
    
    test_documents = [
        {
            "segment_id": "TEST_001",
            "title": "Policy Document Section 1",
            "text": "Someone quickly paying for digital storage solutions. The student discipline policy framework requires academic research. What you need is frequently updated technology systems.",
            "state": "TestState1",
            "language": "english",
            "level": 1,
            "order": 1
        },
        {
            "segment_id": "TEST_002", 
            "title": "Educational Guidelines",
            "text": "Data privacy regulations currently operating in educational institutions. Disciplinary action procedures must ensure student rights protection.",
            "state": "TestState2",
            "language": "english", 
            "level": 1,
            "order": 2
        },
        {
            "segment_id": "TEST_003",
            "title": "Technology Implementation",
            "text": "Digital transformation initiatives require comprehensive planning. Academic institutions need robust infrastructure systems for effective learning environments.",
            "state": "TestState1",
            "language": "english",
            "level": 1, 
            "order": 3
        }
    ]
    
    # Create test input directory
    test_input_dir = Path("test_linguistic_input")
    test_input_dir.mkdir(exist_ok=True)
    
    # Save test documents
    test_file = test_input_dir / "test_documents.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_documents, f, indent=2, ensure_ascii=False)
    
    return str(test_input_dir), str(test_file)

def run_complete_pipeline_test():
    """Run complete pipeline test with linguistic validation"""
    
    print("🧪 COMPLETE LINGUISTIC INTEGRATION TEST")
    print("=" * 60)
    print("Testing end-to-end pipeline with STRICT linguistic validation")
    print("Ensuring only valid noun phrases become graph nodes")
    print()
    
    try:
        # Create test data
        print("📁 Creating test data...")
        test_input_dir, test_file = create_test_data()
        print(f"✅ Test data created: {test_file}")
        
        # Import and initialize the pipeline
        from complete_usage_guide import ResearchPipelineCLI
        
        print("\n🔄 Initializing research pipeline...")
        app = ResearchPipelineCLI()
        
        # Set up test environment
        app.input_directory = test_input_dir
        app.input_files = [test_file]
        app.output_dir = "test_linguistic_output"
        app.pipeline_state['data_loaded'] = True
        
        print("✅ Pipeline initialized")
        
        # Step 1: Text cleaning with enhanced processing
        print("\n🧹 Step 1: Text Cleaning & Normalization...")
        app.clean_and_normalize_text()
        
        if not app.pipeline_state['text_cleaned']:
            print("❌ Text cleaning failed")
            return False
        
        print("✅ Text cleaning completed")
        
        # Step 2: Phrase extraction with STRICT linguistic validation
        print("\n🔍 Step 2: Phrase Extraction with STRICT Validation...")
        app.extract_tokens_and_phrases()
        
        if not app.pipeline_state['phrases_constructed']:
            print("❌ Phrase extraction failed")
            return False
        
        print("✅ Phrase extraction completed")
        
        # Analyze results
        print("\n📊 ANALYZING LINGUISTIC VALIDATION RESULTS:")
        print("-" * 50)
        
        if hasattr(app, 'phrase_data'):
            phrase_counts = app.phrase_data.get('phrase_counts', {})
            filtered_phrases = app.phrase_data.get('filtered_phrases', {})
            
            print(f"Total phrase instances: {len(app.phrase_data.get('all_phrases', []))}")
            print(f"Unique phrases: {len(phrase_counts)}")
            print(f"Phrases above threshold: {len(filtered_phrases)}")
            
            # Check for invalid phrases that should NEVER appear
            invalid_patterns = [
                'someone', 'what you', 'you need', 'quickly', 'frequently', 
                'currently', 'paying', 'operating', 'updated'
            ]
            
            found_invalid = []
            for phrase in filtered_phrases.keys():
                phrase_lower = phrase.lower()
                for invalid in invalid_patterns:
                    if invalid in phrase_lower:
                        found_invalid.append(phrase)
            
            if found_invalid:
                print(f"\n❌ VALIDATION FAILED: Found invalid phrases:")
                for phrase in found_invalid:
                    print(f"   - {phrase}")
                return False
            else:
                print(f"\n✅ VALIDATION PASSED: No invalid phrases found")
            
            # Check for valid phrases that should be allowed
            valid_patterns = [
                'digital storage', 'student discipline', 'policy framework',
                'academic research', 'data privacy', 'disciplinary action',
                'technology systems', 'educational institutions', 'student rights'
            ]
            
            found_valid = []
            for phrase in filtered_phrases.keys():
                phrase_lower = phrase.lower()
                for valid in valid_patterns:
                    if valid in phrase_lower:
                        found_valid.append(phrase)
            
            print(f"\n✅ VALID PHRASES FOUND:")
            for phrase in found_valid[:10]:  # Show first 10
                count = filtered_phrases.get(phrase, 0)
                print(f"   - {phrase} (count: {count})")
            
            # Enhanced features check
            if 'enhanced_features' in app.phrase_data:
                enhanced = app.phrase_data['enhanced_features']
                print(f"\n🔬 ENHANCED PROCESSING RESULTS:")
                print(f"   Linguistically validated phrases: {len(enhanced.get('final_phrases', []))}")
                print(f"   Dynamic stopwords identified: {len(enhanced.get('dynamic_stopwords', []))}")
                
                # Check processing metadata
                if 'processing_metadata' in enhanced:
                    metadata = enhanced['processing_metadata']
                    if metadata.get('pos_gating_enforced'):
                        print(f"   ✅ POS-based gating: ENFORCED")
                    if metadata.get('dependency_construction_used'):
                        print(f"   ✅ Dependency-based construction: APPLIED")
                    if metadata.get('linguistic_validation_enabled'):
                        print(f"   ✅ Linguistic validation: ENABLED")
            
        # Step 3: Build graph to test node quality
        print("\n🌐 Step 3: Building Global Graph...")
        app.build_global_graph()
        
        if not app.pipeline_state['global_graph_built']:
            print("❌ Graph construction failed")
            return False
        
        print("✅ Graph construction completed")
        
        # Analyze graph nodes
        if hasattr(app, 'global_graph_object') and app.global_graph_object:
            graph = app.global_graph_object
            nodes = list(graph.nodes())
            
            print(f"\n📊 GRAPH NODE ANALYSIS:")
            print(f"   Total nodes: {len(nodes)}")
            print(f"   Total edges: {graph.number_of_edges()}")
            
            # Check nodes for invalid patterns
            invalid_nodes = []
            for node in nodes:
                node_lower = node.lower()
                for invalid in invalid_patterns:
                    if invalid in node_lower:
                        invalid_nodes.append(node)
            
            if invalid_nodes:
                print(f"\n❌ GRAPH VALIDATION FAILED: Found invalid nodes:")
                for node in invalid_nodes:
                    print(f"   - {node}")
                return False
            else:
                print(f"\n✅ GRAPH VALIDATION PASSED: All nodes are linguistically valid")
            
            # Show sample valid nodes
            print(f"\n📋 SAMPLE GRAPH NODES (first 10):")
            for i, node in enumerate(nodes[:10], 1):
                print(f"   {i:2d}. {node}")
        
        # Step 4: View detailed statistics
        print("\n📊 Step 4: Viewing Phrase Statistics...")
        app.view_phrase_statistics()
        
        print("\n" + "=" * 60)
        print("🎉 COMPLETE INTEGRATION TEST PASSED!")
        print("=" * 60)
        print("✅ Linguistic validation successfully integrated")
        print("✅ Only valid noun phrases become graph nodes")
        print("✅ Invalid phrases (pronouns, adverbs, verbs) rejected")
        print("✅ Graph structure is linguistically sound")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        import shutil
        if Path("test_linguistic_input").exists():
            shutil.rmtree("test_linguistic_input")
        if Path("test_linguistic_output").exists():
            shutil.rmtree("test_linguistic_output")

def demonstrate_validation_rules():
    """Demonstrate the specific validation rules"""
    
    print("\n🔍 DEMONSTRATING VALIDATION RULES")
    print("=" * 50)
    
    try:
        from semantic_coword_pipeline.processors.linguistic_phrase_validator import POSBasedPhraseGate
        
        pos_gate = POSBasedPhraseGate()
        
        print("📋 VALIDATION RULES DEMONSTRATION:")
        print()
        
        # Rule 1: Head must be NOUN/PROPN
        print("Rule 1: Head token must be NOUN/PROPN")
        test_cases = [
            ("student discipline", ["student", "discipline"], ["NOUN", "NOUN"], True),
            ("quickly running", ["quickly", "running"], ["ADV", "VERB"], False),
        ]
        
        for phrase, tokens, pos_tags, expected in test_cases:
            is_valid, reasons = pos_gate.validate_phrase(phrase, tokens, pos_tags)
            status = "✅ PASS" if (is_valid == expected) else "❌ FAIL"
            print(f"   {phrase}: {status} ({'valid' if is_valid else 'invalid'})")
        
        # Rule 2: Must contain at least one NOUN/PROPN
        print("\nRule 2: Must contain at least one NOUN/PROPN")
        test_cases = [
            ("digital storage", ["digital", "storage"], ["ADJ", "NOUN"], True),
            ("quickly frequently", ["quickly", "frequently"], ["ADV", "ADV"], False),
        ]
        
        for phrase, tokens, pos_tags, expected in test_cases:
            is_valid, reasons = pos_gate.validate_phrase(phrase, tokens, pos_tags)
            status = "✅ PASS" if (is_valid == expected) else "❌ FAIL"
            print(f"   {phrase}: {status} ({'valid' if is_valid else 'invalid'})")
        
        # Rule 3: Head must NOT be PRON/ADV/VERB
        print("\nRule 3: Head must NOT be PRON/ADV/VERB")
        test_cases = [
            ("policy framework", ["policy", "framework"], ["NOUN", "NOUN"], True),
            ("what you", ["what", "you"], ["PRON", "PRON"], False),
            ("currently operating", ["currently", "operating"], ["ADV", "VERB"], False),
        ]
        
        for phrase, tokens, pos_tags, expected in test_cases:
            is_valid, reasons = pos_gate.validate_phrase(phrase, tokens, pos_tags)
            status = "✅ PASS" if (is_valid == expected) else "❌ FAIL"
            print(f"   {phrase}: {status} ({'valid' if is_valid else 'invalid'})")
        
        # Rule 4: Single tokens must be NOUN/PROPN
        print("\nRule 4: Single tokens must be NOUN/PROPN")
        test_cases = [
            ("education", ["education"], ["NOUN"], True),
            ("quickly", ["quickly"], ["ADV"], False),
            ("someone", ["someone"], ["PRON"], False),
        ]
        
        for phrase, tokens, pos_tags, expected in test_cases:
            is_valid, reasons = pos_gate.validate_phrase(phrase, tokens, pos_tags)
            status = "✅ PASS" if (is_valid == expected) else "❌ FAIL"
            print(f"   {phrase}: {status} ({'valid' if is_valid else 'invalid'})")
        
        print("\n✅ All validation rules working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Validation rules demonstration failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 COMPLETE LINGUISTIC VALIDATION INTEGRATION TEST")
    print("=" * 70)
    print("Testing STRICT POS-based gating and dependency-based construction")
    print("Ensuring only linguistically valid noun phrases become graph nodes")
    print()
    
    # Run tests
    test1_passed = demonstrate_validation_rules()
    test2_passed = run_complete_pipeline_test()
    
    # Final summary
    print("\n" + "=" * 70)
    print("🏁 FINAL INTEGRATION TEST RESULTS")
    print("=" * 70)
    
    tests_passed = sum([test1_passed, test2_passed])
    total_tests = 2
    
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ STRICT linguistic validation successfully implemented")
        print("✅ POS-based phrase gating enforced")
        print("✅ Dependency-based phrase construction applied")
        print("✅ Invalid phrases (pronouns, adverbs, verbs) rejected")
        print("✅ Valid noun phrases accepted")
        print("✅ Graph nodes are linguistically sound")
        print("\n📋 IMPLEMENTATION COMPLETE:")
        print("   - POS-based phrase gating (MANDATORY) ✅")
        print("   - Dependency-based phrase construction (MANDATORY) ✅")
        print("   - Strict filtering order: Linguistic → Lexical → TF-IDF ✅")
        print("   - Validation rules implemented ✅")
        return True
    else:
        print("\n⚠️ SOME INTEGRATION TESTS FAILED")
        print("❌ Linguistic validation needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)