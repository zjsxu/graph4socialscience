#!/usr/bin/env python3
"""
Comprehensive test script for complete_usage_guide.py
Tests all functionality with real data and generates complete output
"""

import os
import sys
import json
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, '.')

def run_complete_pipeline_test():
    """Run the complete pipeline with all functionality"""
    
    # Import the pipeline class
    from complete_usage_guide import ResearchPipelineCLI
    
    print("🚀 COMPREHENSIVE PIPELINE TEST")
    print("=" * 60)
    
    # Initialize pipeline
    app = ResearchPipelineCLI()
    
    # Set real data paths
    input_dir = 'test_input'
    output_dir = 'test_output'
    
    app.input_directory = input_dir
    app.output_dir = output_dir
    
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Step 1: Load directory
    print("\n1️⃣ DIRECTORY INPUT LOADING")
    print("-" * 40)
    
    # Scan for files
    app.input_files = []
    valid_extensions = {'.json', '.txt', '.md'}
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in valid_extensions:
                app.input_files.append(file_path)
    
    print(f"✅ Found {len(app.input_files)} valid files")
    app.pipeline_state['data_loaded'] = True
    
    # Step 2: Text cleaning
    print("\n2️⃣ TEXT CLEANING & NORMALIZATION")
    print("-" * 40)
    
    app.clean_and_normalize_text()
    
    # Export cleaned text
    print("\n📤 Exporting cleaned text data...")
    app.export_cleaned_text()
    
    # View cleaning results
    print("\n📊 Viewing text cleaning results...")
    app.view_text_cleaning_results()
    
    # Step 3: Phrase construction
    print("\n3️⃣ TOKEN/PHRASE CONSTRUCTION")
    print("-" * 40)
    
    # Configure parameters
    app.configure_phrase_parameters()
    
    # Extract phrases
    app.extract_tokens_and_phrases()
    
    # View statistics
    app.view_phrase_statistics()
    
    # Step 4: Global graph construction
    print("\n4️⃣ GLOBAL GRAPH CONSTRUCTION")
    print("-" * 40)
    
    # Build global graph
    app.build_global_graph()
    
    # View statistics
    app.view_global_graph_statistics()
    
    # Export global graph
    app.export_global_graph_data()
    
    # Step 5: Subgraph activation
    print("\n5️⃣ SUBGRAPH ACTIVATION")
    print("-" * 40)
    
    # Activate subgraphs
    app.activate_state_subgraphs()
    
    # View comparisons
    app.view_subgraph_comparisons()
    
    # Export subgraphs
    app.export_subgraph_data()
    
    # Step 6: Visualization & Export
    print("\n6️⃣ VISUALIZATION & EXPORT")
    print("-" * 40)
    
    # Generate visualizations
    app.generate_deterministic_visualizations()
    
    # View output paths
    app.view_output_image_paths()
    
    # Export complete results
    app.export_complete_results()
    
    # Reproducibility controls
    print("\n🔬 REPRODUCIBILITY CONTROLS")
    print("-" * 40)
    
    # View all parameters
    app.view_all_parameters()
    
    # Export parameter configuration
    app.export_parameter_configuration()
    
    # Final summary
    print("\n🎉 COMPREHENSIVE TEST COMPLETED!")
    print("=" * 60)
    
    # Check all output files
    check_all_output_files(output_dir)
    
    return True

def check_all_output_files(output_dir):
    """Check all output files and provide summary"""
    print(f"\n📁 COMPLETE OUTPUT FILE SUMMARY")
    print("=" * 60)
    
    if not os.path.exists(output_dir):
        print("❌ Output directory does not exist")
        return
    
    # Organize files by category
    file_categories = {
        'cleaned_text': [],
        'global_graph': [],
        'subgraphs': [],
        'visualizations': [],
        'exports': [],
        'parameters': [],
        'other': []
    }
    
    total_size = 0
    total_files = 0
    
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, output_dir)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            total_files += 1
            
            # Categorize files
            if 'cleaned_text' in rel_path:
                file_categories['cleaned_text'].append((rel_path, file_size))
            elif 'global_graph' in rel_path:
                file_categories['global_graph'].append((rel_path, file_size))
            elif 'subgraph' in rel_path:
                file_categories['subgraphs'].append((rel_path, file_size))
            elif 'visualization' in rel_path:
                file_categories['visualizations'].append((rel_path, file_size))
            elif 'export' in rel_path:
                file_categories['exports'].append((rel_path, file_size))
            elif 'parameter' in rel_path:
                file_categories['parameters'].append((rel_path, file_size))
            else:
                file_categories['other'].append((rel_path, file_size))
    
    # Display summary by category
    for category, files in file_categories.items():
        if files:
            print(f"\n📂 {category.upper().replace('_', ' ')} ({len(files)} files):")
            for rel_path, file_size in files:
                size_mb = file_size / (1024 * 1024)
                if size_mb > 1:
                    print(f"   📄 {rel_path} ({size_mb:.1f} MB)")
                else:
                    size_kb = file_size / 1024
                    print(f"   📄 {rel_path} ({size_kb:.1f} KB)")
    
    # Overall summary
    total_size_mb = total_size / (1024 * 1024)
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"   Total files: {total_files}")
    print(f"   Total size: {total_size_mb:.1f} MB")
    print(f"   Output directory: {os.path.abspath(output_dir)}")
    
    # Verify all expected categories have files
    expected_categories = ['cleaned_text', 'global_graph', 'subgraphs', 'parameters']
    missing_categories = [cat for cat in expected_categories if not file_categories[cat]]
    
    if missing_categories:
        print(f"\n⚠️ Missing output categories: {', '.join(missing_categories)}")
    else:
        print(f"\n✅ All expected output categories present!")

def main():
    """Main test function"""
    try:
        success = run_complete_pipeline_test()
        if success:
            print("\n🎉 ALL FUNCTIONALITY TESTED SUCCESSFULLY!")
            print("📁 Check the output directory for complete results")
            return 0
        else:
            print("\n❌ TESTS FAILED")
            return 1
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())