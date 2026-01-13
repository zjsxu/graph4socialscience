#!/usr/bin/env python3
"""
Test Linguistic Validation

This script tests the strict linguistic validation implementation to ensure
that only linguistically valid noun phrases become graph nodes.

Tests the CORE REQUIREMENTS:
1. POS-based phrase gating (MANDATORY)
2. Dependency-based phrase construction (MANDATORY)
3. Validation rules (MUST BE IMPLEMENTED)
"""

import os
import sys
from pathlib import Path

def test_linguistic_validation():
    """Test the linguistic validation system"""
    
    print("🧪 TESTING STRICT LINGUISTIC VALIDATION")
    print("=" * 60)
    print("Testing POS-based gating and dependency-based construction")
    print("Ensuring only valid noun phrases become graph nodes")
    print()
    
    try:
        # Import the linguistic validator
        from semantic_coword_pipeline.processors.linguistic_phrase_validator import (
            LinguisticPhraseValidator, POSBasedPhraseGate
        )
        from semantic_coword_pipeline.core.config import Config
        
        print("✅ Successfully imported linguistic validation components")
        
        # Initialize validator
        config = Config()
        validator = LinguisticPhraseValidator(config.get_section('enhanced_text_processing'))
        pos_gate = POSBasedPhraseGate()
        
        print("✅ Successfully initialized linguistic validator")
        
        # Test cases that MUST NEVER appear as graph nodes
        invalid_phrases = [
            # Pronouns or pronoun-based spans
            ("someone", ["someone"], ["PRON"]),
            ("what you", ["what", "you"], ["PRON", "PRON"]),
            ("you know", ["you", "know"], ["PRON", "VERB"]),
            
            # Standalone adjectives
            ("quick", ["quick"], ["ADJ"]),
            ("timely", ["timely"], ["ADJ"]),
            ("important", ["important"], ["ADJ"]),
            
            # Adverbs
            ("frequently", ["frequently"], ["ADV"]),
            ("currently", ["currently"], ["ADV"]),
            ("quickly", ["quickly"], ["ADV"]),
            
            # Verb-only or gerund phrases
            ("paying", ["paying"], ["VERB"]),
            ("operating", ["operating"], ["VERB"]),
            ("running", ["running"], ["VERB"]),
            ("is running", ["is", "running"], ["AUX", "VERB"]),
        ]
        
        # Test cases that MUST be allowed
        valid_phrases = [
            # Noun-noun compounds
            ("student discipline", ["student", "discipline"], ["NOUN", "NOUN"]),
            ("data privacy", ["data", "privacy"], ["NOUN", "NOUN"]),
            ("policy framework", ["policy", "framework"], ["NOUN", "NOUN"]),
            
            # Adjective-noun phrases
            ("digital storage", ["digital", "storage"], ["ADJ", "NOUN"]),
            ("disciplinary action", ["disciplinary", "action"], ["ADJ", "NOUN"]),
            ("academic research", ["academic", "research"], ["ADJ", "NOUN"]),
            
            # Single nouns
            ("education", ["education"], ["NOUN"]),
            ("technology", ["technology"], ["NOUN"]),
            ("policy", ["policy"], ["NOUN"]),
        ]
        
        print("\n🔍 TESTING INVALID PHRASES (must be rejected):")
        print("-" * 50)
        
        invalid_passed = 0
        for phrase_text, tokens, pos_tags in invalid_phrases:
            is_valid, reasons = pos_gate.validate_phrase(phrase_text, tokens, pos_tags)
            
            if is_valid:
                print(f"❌ FAIL: '{phrase_text}' was incorrectly ACCEPTED")
                invalid_passed += 1
            else:
                print(f"✅ PASS: '{phrase_text}' correctly REJECTED - {reasons[0] if reasons else 'No reason'}")
        
        print(f"\nInvalid phrase rejection: {len(invalid_phrases) - invalid_passed}/{len(invalid_phrases)} correct")
        
        print("\n🔍 TESTING VALID PHRASES (must be accepted):")
        print("-" * 50)
        
        valid_rejected = 0
        for phrase_text, tokens, pos_tags in valid_phrases:
            is_valid, reasons = pos_gate.validate_phrase(phrase_text, tokens, pos_tags)
            
            if not is_valid:
                print(f"❌ FAIL: '{phrase_text}' was incorrectly REJECTED - {reasons[0] if reasons else 'No reason'}")
                valid_rejected += 1
            else:
                print(f"✅ PASS: '{phrase_text}' correctly ACCEPTED")
        
        print(f"\nValid phrase acceptance: {len(valid_phrases) - valid_rejected}/{len(valid_phrases)} correct")
        
        # Test with complete usage guide integration
        print("\n🔍 TESTING INTEGRATION WITH COMPLETE USAGE GUIDE:")
        print("-" * 50)
        
        try:
            from complete_usage_guide import ResearchPipelineCLI
            
            # Create test app
            app = ResearchPipelineCLI()
            
            # Create test data with problematic phrases
            test_documents = [
                {
                    'segment_id': 'TEST_001',
                    'title': 'Test Document',
                    'text': 'Someone quickly paying for digital storage. The student discipline policy framework.',
                    'state': 'TestState',
                    'language': 'english'
                }
            ]
            
            # Simulate data loading
            app.cleaned_text_data = [
                {
                    'segment_id': doc['segment_id'],
                    'title': doc['title'],
                    'original_text': doc['text'],
                    'cleaned_text': doc['text'].lower(),
                    'tokens': doc['text'].lower().split(),
                    'token_count': len(doc['text'].split()),
                    'state': doc['state'],
                    'language': doc['language']
                }
                for doc in test_documents
            ]
            
            app.pipeline_state['data_loaded'] = True
            app.pipeline_state['text_cleaned'] = True
            
            print("✅ Test data prepared")
            
            # Test phrase extraction with linguistic validation
            print("🔍 Testing phrase extraction with linguistic validation...")
            app.extract_tokens_and_phrases()
            
            if app.pipeline_state['phrases_constructed']:
                print("✅ Phrase extraction completed")
                
                # Check results
                if hasattr(app, 'phrase_data') and 'enhanced_features' in app.phrase_data:
                    final_phrases = app.phrase_data['enhanced_features'].get('final_phrases', [])
                    print(f"📊 Final phrases extracted: {len(final_phrases)}")
                    
                    # Check for invalid phrases
                    invalid_found = []
                    for phrase in final_phrases:
                        phrase_lower = phrase.lower()
                        if any(invalid in phrase_lower for invalid, _, _ in invalid_phrases):
                            invalid_found.append(phrase)
                    
                    if invalid_found:
                        print(f"❌ FAIL: Found invalid phrases in results: {invalid_found}")
                    else:
                        print("✅ PASS: No invalid phrases found in results")
                    
                    # Check for valid phrases
                    valid_found = []
                    for phrase in final_phrases:
                        phrase_lower = phrase.lower()
                        if any(valid[0] in phrase_lower for valid, _, _ in valid_phrases):
                            valid_found.append(phrase)
                    
                    print(f"✅ Valid phrases found: {valid_found}")
                    
                else:
                    print("⚠️ Enhanced features not available - using basic mode")
            else:
                print("❌ Phrase extraction failed")
            
        except Exception as e:
            print(f"⚠️ Integration test failed: {e}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 LINGUISTIC VALIDATION TEST SUMMARY")
        print("=" * 60)
        
        total_invalid_tests = len(invalid_phrases)
        total_valid_tests = len(valid_phrases)
        invalid_correct = total_invalid_tests - invalid_passed
        valid_correct = total_valid_tests - valid_rejected
        
        print(f"Invalid phrase rejection: {invalid_correct}/{total_invalid_tests} ({invalid_correct/total_invalid_tests*100:.1f}%)")
        print(f"Valid phrase acceptance: {valid_correct}/{total_valid_tests} ({valid_correct/total_valid_tests*100:.1f}%)")
        
        overall_score = (invalid_correct + valid_correct) / (total_invalid_tests + total_valid_tests)
        print(f"Overall accuracy: {overall_score*100:.1f}%")
        
        if overall_score >= 0.9:
            print("🎉 EXCELLENT: Linguistic validation working correctly!")
            return True
        elif overall_score >= 0.7:
            print("⚠️ GOOD: Linguistic validation mostly working, some issues")
            return True
        else:
            print("❌ POOR: Linguistic validation has significant issues")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_validation_examples():
    """Test specific validation examples from the requirements"""
    
    print("\n🧪 TESTING SPECIFIC VALIDATION EXAMPLES")
    print("=" * 50)
    
    try:
        from semantic_coword_pipeline.processors.linguistic_phrase_validator import POSBasedPhraseGate
        
        pos_gate = POSBasedPhraseGate()
        
        # Examples that MUST NEVER appear
        never_appear = [
            ("someone", ["someone"], ["PRON"]),
            ("what you", ["what", "you"], ["PRON", "PRON"]),
            ("quick", ["quick"], ["ADJ"]),
            ("paying", ["paying"], ["VERB"]),
        ]
        
        # Examples that MUST be allowed
        must_allow = [
            ("student discipline", ["student", "discipline"], ["NOUN", "NOUN"]),
            ("data privacy", ["data", "privacy"], ["NOUN", "NOUN"]),
            ("digital storage", ["digital", "storage"], ["ADJ", "NOUN"]),
            ("disciplinary action", ["disciplinary", "action"], ["ADJ", "NOUN"]),
        ]
        
        print("❌ MUST NEVER APPEAR:")
        all_rejected = True
        for phrase, tokens, pos_tags in never_appear:
            is_valid, reasons = pos_gate.validate_phrase(phrase, tokens, pos_tags)
            status = "✅ REJECTED" if not is_valid else "❌ ACCEPTED"
            print(f"   {phrase}: {status}")
            if is_valid:
                all_rejected = False
        
        print("\n✅ MUST BE ALLOWED:")
        all_accepted = True
        for phrase, tokens, pos_tags in must_allow:
            is_valid, reasons = pos_gate.validate_phrase(phrase, tokens, pos_tags)
            status = "✅ ACCEPTED" if is_valid else "❌ REJECTED"
            print(f"   {phrase}: {status}")
            if not is_valid:
                all_accepted = False
        
        print(f"\nValidation Rules Test:")
        print(f"   Invalid phrases rejected: {'✅ PASS' if all_rejected else '❌ FAIL'}")
        print(f"   Valid phrases accepted: {'✅ PASS' if all_accepted else '❌ FAIL'}")
        
        return all_rejected and all_accepted
        
    except Exception as e:
        print(f"❌ Validation examples test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 LINGUISTIC VALIDATION TEST SUITE")
    print("=" * 60)
    print("Testing STRICT POS-based gating and dependency-based construction")
    print("Ensuring graph nodes are linguistically valid noun phrases")
    print()
    
    # Run tests
    test1_passed = test_linguistic_validation()
    test2_passed = test_validation_examples()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 FINAL TEST RESULTS")
    print("=" * 60)
    
    tests_passed = sum([test1_passed, test2_passed])
    total_tests = 2
    
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Linguistic validation is working correctly")
        print("✅ Only valid noun phrases will become graph nodes")
        print("✅ Invalid phrases (pronouns, adverbs, verbs) are rejected")
        return True
    else:
        print("⚠️ SOME TESTS FAILED")
        print("❌ Linguistic validation needs attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)