#!/usr/bin/env python3
"""
Quick pipeline setup with memo data paths

This script sets up the pipeline with the data paths from memo.txt and then
launches the interactive menu, so you don't have to manually configure paths.
"""

import os
import sys
from complete_usage_guide import ResearchPipelineCLI

def setup_and_run():
    """Setup pipeline with memo paths and run interactively"""
    
    print("🚀 QUICK PIPELINE SETUP")
    print("=" * 50)
    
    # Data paths from memo
    chinese_data_path = "/Users/zhangjingsen/Desktop/python/graph4socialscience/semantic-node-refinement-test/data/raw"
    english_toc_path = "/Users/zhangjingsen/Desktop/python/graph4socialscience/toc_doc"
    output_base_path = "/Users/zhangjingsen/Desktop/python/graph4socialscience/hajimi/to/"
    
    # Check which data paths exist
    chinese_exists = os.path.exists(chinese_data_path)
    english_exists = os.path.exists(english_toc_path)
    
    print(f"📂 Chinese data: {'✅' if chinese_exists else '❌'} {chinese_data_path}")
    print(f"📂 English TOC: {'✅' if english_exists else '❌'} {english_toc_path}")
    print(f"📂 Output base: {output_base_path}")
    
    if not (chinese_exists or english_exists):
        print("❌ No data sources found. Please check paths.")
        return
    
    # Choose data source
    if chinese_exists and english_exists:
        print("\n🔍 Choose data source:")
        print("1. Chinese data (semantic-node-refinement-test)")
        print("2. English TOC data (toc_doc)")
        
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == "1":
            input_path = chinese_data_path
            output_dir = os.path.join(output_base_path, "chinese_output")
        elif choice == "2":
            input_path = english_toc_path
            output_dir = os.path.join(output_base_path, "english_toc_output")
        else:
            print("❌ Invalid choice")
            return
    elif chinese_exists:
        input_path = chinese_data_path
        output_dir = os.path.join(output_base_path, "chinese_output")
    else:
        input_path = english_toc_path
        output_dir = os.path.join(output_base_path, "english_toc_output")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 Selected input: {input_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Initialize and configure pipeline
    try:
        pipeline = ResearchPipelineCLI()
        
        # Pre-configure paths
        pipeline.input_directory = input_path
        pipeline.output_dir = output_dir
        
        # Scan for input files
        print("🔍 Scanning for input files...")
        pipeline.input_files = []
        valid_extensions = {'.json', '.txt', '.md'}
        
        for root, dirs, files in os.walk(input_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in valid_extensions:
                    pipeline.input_files.append(file_path)
        
        if pipeline.input_files:
            print(f"✅ Found {len(pipeline.input_files)} files to process")
            pipeline.pipeline_state['data_loaded'] = True
        else:
            print(f"⚠️ No valid files found in {input_path}")
        
        print("\n🎯 Pipeline configured! Launching interactive menu...")
        print("💡 Tip: Data is already loaded, you can start from step 2.1 (Clean & Normalize Text)")
        input("Press Enter to continue...")
        
        # Run interactive pipeline
        pipeline.run()
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    setup_and_run()