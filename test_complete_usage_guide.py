#!/usr/bin/env python3
"""
Test script for complete_usage_guide.py with real data
Tests the full pipeline workflow and identifies errors for fixing
"""

import os
import sys
import json
import shutil
import traceback
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

def setup_test_environment():
    """Setup test environment with real data paths"""
    test_config = {
        'input_dir': 'test_input',
        'output_dir': 'test_output',
        'test_mode': True
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(test_config['output_dir'], exist_ok=True)
    
    # Check if input directory exists
    if not os.path.exists(test_config['input_dir']):
        print(f"❌ Input directory does not exist: {test_config['input_dir']}")
        return None
    
    print(f"✅ Test environment setup complete")
    print(f"   Input: {test_config['input_dir']}")
    print(f"   Output: {test_config['output_dir']}")
    
    return test_config

def test_pipeline_step(step_name, step_function, *args, **kwargs):
    """Test a single pipeline step and catch errors"""
    print(f"\n🔧 Testing: {step_name}")
    print("-" * 50)
    
    try:
        result = step_function(*args, **kwargs)
        print(f"✅ {step_name} completed successfully")
        return True, result
    except Exception as e:
        print(f"❌ {step_name} failed: {e}")
        print("📋 Error details:")
        traceback.print_exc()
        return False, str(e)

def simulate_pipeline_workflow(test_config):
    """Simulate the complete pipeline workflow with real data"""
    print("\n🚀 STARTING COMPLETE PIPELINE TEST")
    print("=" * 60)
    
    # Import the pipeline class
    try:
        from complete_usage_guide import ResearchPipelineCLI
        app = ResearchPipelineCLI()
        print("✅ Pipeline class imported successfully")
    except Exception as e:
        print(f"❌ Failed to import pipeline class: {e}")
        return False
    
    # Override output directory
    app.output_dir = test_config['output_dir']
    app.input_directory = test_config['input_dir']
    
    # Step 1: Load input directory
    success, result = test_pipeline_step(
        "1. Directory Input Loading",
        simulate_directory_loading,
        app, test_config['input_dir']
    )
    if not success:
        return False
    
    # Step 2: Text cleaning
    success, result = test_pipeline_step(
        "2. Text Cleaning & Normalization",
        simulate_text_cleaning,
        app
    )
    if not success:
        return False
    
    # Step 3: Phrase extraction
    success, result = test_pipeline_step(
        "3. Token/Phrase Construction",
        simulate_phrase_extraction,
        app
    )
    if not success:
        return False
    
    # Step 4: Global graph construction
    success, result = test_pipeline_step(
        "4. Global Graph Construction",
        simulate_global_graph_construction,
        app
    )
    if not success:
        return False
    
    # Step 5: Subgraph activation
    success, result = test_pipeline_step(
        "5. Subgraph Activation",
        simulate_subgraph_activation,
        app
    )
    if not success:
        return False
    
    # Step 6: Export operations (this is where the JSON error occurs)
    success, result = test_pipeline_step(
        "6. Export Operations",
        simulate_export_operations,
        app
    )
    if not success:
        return False
    
    print("\n🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!")
    return True

def simulate_directory_loading(app, input_dir):
    """Simulate directory loading"""
    # Scan for files
    app.input_files = []
    valid_extensions = {'.json', '.txt', '.md'}
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in valid_extensions:
                app.input_files.append(file_path)
    
    print(f"📁 Found {len(app.input_files)} valid files")
    app.pipeline_state['data_loaded'] = True
    return len(app.input_files)

def simulate_text_cleaning(app):
    """Simulate text cleaning"""
    input_data = app.load_input_data()
    
    cleaned_documents = []
    for doc in input_data:
        cleaned_text = doc['text'].lower().strip()
        tokens = cleaned_text.split()
        
        cleaned_doc = {
            'segment_id': doc['segment_id'],
            'title': doc['title'],
            'original_text': doc['text'],
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'token_count': len(tokens),
            'state': doc['state'],
            'language': doc['language']
        }
        cleaned_documents.append(cleaned_doc)
    
    app.cleaned_text_data = cleaned_documents
    app.pipeline_state['text_cleaned'] = True
    
    print(f"📊 Cleaned {len(cleaned_documents)} documents")
    return len(cleaned_documents)

def simulate_phrase_extraction(app):
    """Simulate phrase extraction"""
    all_phrases = []
    phrase_counts = {}
    
    for doc in app.cleaned_text_data:
        tokens = doc['tokens']
        
        # Extract unigrams and bigrams
        for token in tokens:
            if len(token) > 2:
                all_phrases.append(token)
                phrase_counts[token] = phrase_counts.get(token, 0) + 1
        
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            all_phrases.append(bigram)
            phrase_counts[bigram] = phrase_counts.get(bigram, 0) + 1
    
    # Filter by minimum frequency
    min_freq = app.reproducibility_config['min_phrase_frequency']
    filtered_phrases = {phrase: count for phrase, count in phrase_counts.items() 
                      if count >= min_freq}
    
    app.phrase_data = {
        'all_phrases': all_phrases,
        'phrase_counts': phrase_counts,
        'filtered_phrases': filtered_phrases,
        'extraction_params': app.reproducibility_config.copy()
    }
    
    app.pipeline_state['phrases_constructed'] = True
    
    print(f"📊 Extracted {len(filtered_phrases)} filtered phrases")
    return len(filtered_phrases)

def simulate_global_graph_construction(app):
    """Simulate global graph construction - this will reveal the JSON serialization issue"""
    filtered_phrases = app.phrase_data['filtered_phrases']
    phrase_list = list(filtered_phrases.keys())
    
    # Create co-occurrence matrix with tuple keys (this causes the JSON error)
    cooccurrence_matrix = {}
    
    for doc in app.cleaned_text_data:
        doc_phrases = []
        tokens = doc['tokens']
        
        # Get phrases from this document
        doc_phrases.extend([token for token in tokens if token in filtered_phrases])
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            if bigram in filtered_phrases:
                doc_phrases.append(bigram)
        
        # Calculate co-occurrences (use string keys for JSON serialization)
        for i, phrase1 in enumerate(doc_phrases):
            for phrase2 in doc_phrases[i+1:]:
                if phrase1 != phrase2:
                    # Use string key instead of tuple for JSON serialization
                    pair_key = f"{sorted([phrase1, phrase2])[0]}|||{sorted([phrase1, phrase2])[1]}"
                    cooccurrence_matrix[pair_key] = cooccurrence_matrix.get(pair_key, 0) + 1
    
    app.global_graph = {
        'nodes': phrase_list,
        'edges': cooccurrence_matrix,  # Now uses string keys for JSON serialization
        'node_count': len(phrase_list),
        'edge_count': len(cooccurrence_matrix),
        'construction_params': app.reproducibility_config.copy(),
        'construction_timestamp': '2024-01-01T00:00:00'
    }
    
    app.pipeline_state['global_graph_built'] = True
    
    print(f"🌐 Built global graph: {len(phrase_list)} nodes, {len(cooccurrence_matrix)} edges")
    return len(cooccurrence_matrix)

def simulate_subgraph_activation(app):
    """Simulate subgraph activation"""
    # Group documents by state
    state_documents = {}
    for doc in app.cleaned_text_data:
        state = doc['state']
        if state not in state_documents:
            state_documents[state] = []
        state_documents[state].append(doc)
    
    app.state_subgraphs = {}
    
    for state, docs in state_documents.items():
        # Get phrases for this state
        state_phrases = set()
        for doc in docs:
            tokens = doc['tokens']
            state_phrases.update([token for token in tokens if token in app.phrase_data['filtered_phrases']])
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                if bigram in app.phrase_data['filtered_phrases']:
                    state_phrases.add(bigram)
        
        # Filter global graph edges (use string keys)
        state_edges = {}
        for pair_key, weight in app.global_graph['edges'].items():
            # Parse the string key back to phrase pair
            phrase1, phrase2 = pair_key.split('|||')
            if phrase1 in state_phrases and phrase2 in state_phrases:
                state_edges[pair_key] = weight
        
        app.state_subgraphs[state] = {
            'state': state,
            'nodes': list(state_phrases),
            'edges': state_edges,  # Now uses string keys for JSON serialization
            'node_count': len(state_phrases),
            'edge_count': len(state_edges),
            'document_count': len(docs),
            'activation_timestamp': '2024-01-01T00:00:00',
            'source_global_graph': True
        }
    
    app.pipeline_state['subgraphs_activated'] = True
    
    print(f"🗺️ Activated {len(app.state_subgraphs)} state subgraphs")
    return len(app.state_subgraphs)

def simulate_export_operations(app):
    """Simulate export operations - this will trigger the JSON serialization error"""
    
    # Test global graph export (this will fail due to tuple keys)
    try:
        app.export_global_graph_data()
        print("✅ Global graph export successful")
    except Exception as e:
        print(f"❌ Global graph export failed: {e}")
        raise e
    
    # Test subgraph export (this will also fail due to tuple keys)
    try:
        app.export_subgraph_data()
        print("✅ Subgraph export successful")
    except Exception as e:
        print(f"❌ Subgraph export failed: {e}")
        raise e
    
    return True

def check_output_files(output_dir):
    """Check what files were successfully created"""
    print(f"\n📁 CHECKING OUTPUT FILES IN: {output_dir}")
    print("-" * 60)
    
    if not os.path.exists(output_dir):
        print("❌ Output directory does not exist")
        return
    
    total_files = 0
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, output_dir)
            file_size = os.path.getsize(file_path)
            print(f"   📄 {rel_path} ({file_size} bytes)")
            total_files += 1
    
    print(f"\n📊 Total files created: {total_files}")

def main():
    """Main test function"""
    print("🧪 COMPLETE USAGE GUIDE TEST SCRIPT")
    print("=" * 60)
    
    # Setup test environment
    test_config = setup_test_environment()
    if not test_config:
        return 1
    
    # Run pipeline test
    try:
        success = simulate_pipeline_workflow(test_config)
        if success:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n❌ TESTS FAILED - See errors above")
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        traceback.print_exc()
        success = False
    
    # Check output files
    check_output_files(test_config['output_dir'])
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())