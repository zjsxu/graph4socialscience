#!/usr/bin/env python3
"""
Automated pipeline runner using paths from memo.txt

This script automatically runs the complete usage guide pipeline with the data paths
specified in memo.txt, eliminating the need for manual menu navigation.
"""

import os
import sys
from pathlib import Path
from complete_usage_guide import ResearchPipelineCLI

def run_automated_pipeline():
    """Run the pipeline automatically with memo data paths"""
    
    print("🚀 AUTOMATED PIPELINE RUNNER")
    print("=" * 50)
    print("📋 Using data paths from memo.txt")
    
    # Data paths from memo
    chinese_data_path = "/Users/zhangjingsen/Desktop/python/graph4socialscience/semantic-node-refinement-test/data/raw"
    english_toc_path = "/Users/zhangjingsen/Desktop/python/graph4socialscience/toc_doc"
    output_base_path = "/Users/zhangjingsen/Desktop/python/graph4socialscience/hajimi/to/"
    
    # Check which data paths exist
    chinese_exists = os.path.exists(chinese_data_path)
    english_exists = os.path.exists(english_toc_path)
    
    print(f"📂 Chinese data path: {chinese_data_path}")
    print(f"   Status: {'✅ Found' if chinese_exists else '❌ Not found'}")
    print(f"📂 English TOC path: {english_toc_path}")
    print(f"   Status: {'✅ Found' if english_exists else '❌ Not found'}")
    print(f"📂 Output base path: {output_base_path}")
    
    # Choose data source
    if chinese_exists and english_exists:
        print("\n🔍 Both data sources available. Which would you like to process?")
        print("1. Chinese data (semantic-node-refinement-test)")
        print("2. English TOC data (toc_doc)")
        print("3. Both (run separately)")
        
        choice = input("Enter choice (1/2/3): ").strip()
        
        if choice == "1":
            data_paths = [(chinese_data_path, "chinese")]
        elif choice == "2":
            data_paths = [(english_toc_path, "english_toc")]
        elif choice == "3":
            data_paths = [(chinese_data_path, "chinese"), (english_toc_path, "english_toc")]
        else:
            print("❌ Invalid choice")
            return
    elif chinese_exists:
        data_paths = [(chinese_data_path, "chinese")]
        print("📊 Using Chinese data (only available source)")
    elif english_exists:
        data_paths = [(english_toc_path, "english_toc")]
        print("📊 Using English TOC data (only available source)")
    else:
        print("❌ No data sources found. Please check paths in memo.txt")
        return
    
    # Process each data source
    for data_path, data_type in data_paths:
        print(f"\n🔄 Processing {data_type} data from: {data_path}")
        
        # Create output directory for this data type
        output_dir = os.path.join(output_base_path, f"{data_type}_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize pipeline
        try:
            pipeline = ResearchPipelineCLI()
            
            # Set up pipeline with data paths
            pipeline.input_directory = data_path
            pipeline.output_dir = output_dir
            
            # Scan for input files
            print("🔍 Scanning for input files...")
            pipeline.input_files = []
            valid_extensions = {'.json', '.txt', '.md'}
            
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    if file_ext in valid_extensions:
                        pipeline.input_files.append(file_path)
            
            if not pipeline.input_files:
                print(f"⚠️ No valid files found in {data_path}")
                continue
            
            print(f"✅ Found {len(pipeline.input_files)} files to process")
            pipeline.pipeline_state['data_loaded'] = True
            
            # Run pipeline steps automatically
            print("\n📋 AUTOMATED PIPELINE EXECUTION")
            print("-" * 40)
            
            # Step 1: Clean and normalize text
            print("🧹 Step 1: Cleaning and normalizing text...")
            pipeline.clean_and_normalize_text()
            
            if not pipeline.pipeline_state['text_cleaned']:
                print("❌ Text cleaning failed, skipping this data source")
                continue
            
            # Step 2: Extract tokens and phrases
            print("🔍 Step 2: Extracting tokens and phrases...")
            pipeline.extract_tokens_and_phrases()
            
            if not pipeline.pipeline_state['phrases_constructed']:
                print("❌ Phrase extraction failed, skipping this data source")
                continue
            
            # Step 3: Build global graph
            print("🌐 Step 3: Building global co-occurrence graph...")
            pipeline.build_global_graph()
            
            if not pipeline.pipeline_state['global_graph_built']:
                print("❌ Global graph construction failed, skipping this data source")
                continue
            
            # Step 4: Apply scientific optimization
            print("🔬 Step 4: Applying scientific optimization...")
            pipeline.apply_scientific_optimization()
            
            # Step 5: Activate state subgraphs
            print("🗺️ Step 5: Activating state-based subgraphs...")
            pipeline.activate_state_subgraphs()
            
            if not pipeline.pipeline_state['subgraphs_activated']:
                print("❌ Subgraph activation failed, skipping visualization")
                continue
            
            # Step 6: Generate visualizations
            print("🎨 Step 6: Generating scientific visualizations...")
            pipeline.generate_scientific_visualizations()
            
            # Step 7: Export results
            print("📦 Step 7: Exporting complete results...")
            pipeline.export_complete_results()
            
            print(f"✅ Pipeline completed successfully for {data_type} data!")
            print(f"📁 Results saved to: {output_dir}")
            
            # Show visualization paths
            if hasattr(pipeline, 'visualization_paths') and pipeline.visualization_paths:
                print("\n🎨 Generated visualizations:")
                for viz_name, viz_path in pipeline.visualization_paths.items():
                    print(f"   📊 {viz_name}: {viz_path}")
            
        except Exception as e:
            print(f"❌ Pipeline failed for {data_type} data: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Automated pipeline execution completed!")
    print(f"📁 All results saved to: {output_base_path}")

if __name__ == "__main__":
    run_automated_pipeline()