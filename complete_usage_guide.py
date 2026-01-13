#!/usr/bin/env python3
"""
Research-Oriented Text-to-Co-occurrence-Graph Pipeline - Interactive Interface

This is a reproducible, research-oriented interactive interface for text-to-co-occurrence-graph analysis.
The pipeline enforces a fixed workflow order to ensure reproducibility and traceability:

FIXED PIPELINE ORDER:
1. Data Input (TOC-segmented documents from directories)
2. Text Cleaning & Normalization (with preview/export capability)
3. Token/Phrase Construction (configurable parameters)
4. Global Co-occurrence Graph Construction (shared node space)
5. Subgraph Activation (by state/document group from global graph)
6. Visualization & Export (deterministic layouts, clear output paths)

REPRODUCIBILITY FEATURES:
- Fixed random seed for deterministic results
- Explicit co-occurrence window definition (one TOC segment = one window)
- Visible parameter configuration for all reproducibility-affecting settings
- Clear distinction between global graph construction and subgraph filtering
- Traceable output file naming with parameters

SCIENTIFIC ENHANCEMENTS:
- NPMI/Salton semantic weighting for true semantic associations
- Adaptive graph sparsification (Disparity Filter + Quantile-based)
- LCC extraction and community pruning for publication-quality visualizations
- K-Core decomposition for rigorous core-periphery identification
- Deterministic layouts with scientific reporting

Author: Semantic Co-word Network Analysis Research Team
Version: 5.0.0 (Scientific Research Pipeline)
Date: 2024年
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

# Import scientific optimizer
from scientific_graph_optimizer import ScientificGraphOptimizer

# 导入管线组件
try:
    from semantic_coword_pipeline.pipeline import SemanticCowordPipeline
    from semantic_coword_pipeline.core.config import Config
    from semantic_coword_pipeline.core.logger import PipelineLogger
    from semantic_coword_pipeline.core.data_models import TOCDocument
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    IMPORT_ERROR = str(e)


class ResearchPipelineCLI:
    """Research-Oriented Text-to-Co-occurrence-Graph Pipeline Interface"""
    
    def __init__(self):
        self.pipeline = None
        self.input_directory = None
        self.input_files = []
        self.output_dir = "research_pipeline_output"
        
        # Pipeline state tracking
        self.pipeline_state = {
            'data_loaded': False,
            'text_cleaned': False,
            'phrases_constructed': False,
            'global_graph_built': False,
            'subgraphs_activated': False,
            'results_exported': False
        }
        
        # Reproducibility controls - all parameters visible and configurable
        self.reproducibility_config = {
            'random_seed': 42,
            'cooccurrence_window': 'toc_segment',  # One TOC segment = one window
            'edge_weight_strategy': 'npmi',  # 'npmi', 'salton', 'pmi', 'frequency_count'
            'phrase_type': 'mixed',  # word, bigram, or mixed
            'stopword_strategy': 'dynamic_tfidf',  # static_list or dynamic_tfidf
            'layout_algorithm': 'spring_deterministic',
            'min_phrase_frequency': 2,
            'language_detection': 'auto'
        }
        
        # Scientific optimization parameters
        # Scientific optimization parameters - LESS AGGRESSIVE
        self.scientific_config = {
            'semantic_weighting': 'npmi',  # 'npmi', 'salton', 'pmi'
            'sparsification_method': 'quantile',  # Use quantile only, not adaptive
            'edge_retention_rate': 0.3,  # Keep top 30% of edges (was 0.05)
            'disparity_alpha': 0.05,  # Significance level for disparity filter
            'min_community_size': 3,  # Smaller communities allowed (was 8)
            'max_legend_communities': 15,  # More communities in legend (was 10)
            'core_method': 'k_core',  # 'k_core', 'pagerank'
            'min_core_nodes': 20,  # Fewer core nodes required (was 50)
            'enable_lcc_extraction': False,  # DISABLE LCC extraction
            'enable_community_pruning': False,  # DISABLE community pruning
        }
        
        # Processing results storage - GRAPH OBJECTS as first-class citizens
        self.cleaned_text_data = None
        self.phrase_data = None
        
        # CORE GRAPH OBJECTS (not just serialized data)
        self.global_graph_object = None  # NetworkX graph with positions and attributes
        self.global_layout_positions = None  # Fixed 2D positions for all nodes
        self.state_subgraph_objects = {}  # NetworkX subgraph views with shared positions
        
        # Scientific optimization objects
        self.scientific_optimizer = None
        self.optimized_global_graph = None
        self.global_communities = None
        self.global_node_roles = None
        self.structural_statistics = None
        
        # Legacy data structures (for export compatibility only)
        self.global_graph = None  # Will be deprecated in favor of graph_object
        self.state_subgraphs = {}  # Will be deprecated in favor of subgraph_objects
        self.visualization_paths = {}
        
        # Initialize pipeline
        self.initialize_pipeline()
    
    def initialize_pipeline(self):
        """Initialize research pipeline with reproducibility controls"""
        print("=" * 80)
        print("Scientific Research-Oriented Text-to-Co-occurrence-Graph Pipeline")
        print("=" * 80)
        print("🔬 REPRODUCIBLE RESEARCH WORKFLOW")
        print("📋 Fixed Pipeline Order: Data Input → Text Cleaning → Phrase Construction")
        print("   → Global Graph → Scientific Optimization → Subgraph Activation → Visualization & Export")
        print("🎯 Reproducibility Controls: Fixed seed, explicit parameters, traceable outputs")
        print("🧪 Scientific Methods: NPMI weighting, adaptive sparsification, LCC extraction")
        print("=" * 80)
        
        if not PIPELINE_AVAILABLE:
            print(f"❌ ERROR: Pipeline components unavailable")
            print(f"   Details: {IMPORT_ERROR}")
            print("   Please ensure all dependencies are correctly installed.")
            return False
        
        try:
            print("🔄 Initializing research pipeline...")
            self.pipeline = SemanticCowordPipeline()
            
            # Initialize scientific optimizer
            print("🔬 Initializing scientific graph optimizer...")
            self.scientific_optimizer = ScientificGraphOptimizer(random_seed=self.reproducibility_config['random_seed'])
            self.scientific_optimizer.config.update(self.scientific_config)
            
            print("✅ Research pipeline initialized successfully!")
            print(f"🌱 Random seed set to: {self.reproducibility_config['random_seed']}")
            print(f"🪟 Co-occurrence window: {self.reproducibility_config['cooccurrence_window']}")
            print(f"⚖️ Semantic weighting: {self.scientific_config['semantic_weighting'].upper()}")
            print(f"🔬 Sparsification method: {self.scientific_config['sparsification_method']}")
            return True
        except Exception as e:
            print(f"❌ Pipeline initialization failed: {e}")
            return False
    
    def print_menu(self):
        """Display research pipeline menu with fixed workflow order"""
        print("\n" + "=" * 80)
        print("RESEARCH PIPELINE MENU - Fixed Workflow Order")
        print("=" * 80)
        
        # Show current pipeline state
        print("📊 PIPELINE STATE:")
        state_indicators = []
        for step, completed in self.pipeline_state.items():
            indicator = "✅" if completed else "⏳"
            state_indicators.append(f"{indicator} {step.replace('_', ' ').title()}")
        print("   " + " → ".join(state_indicators))
        print()
        
        print("🔬 RESEARCH WORKFLOW (Execute in Order):")
        print("=" * 50)
        print("1. DATA INPUT & DIRECTORY PROCESSING")
        print("   1.1 Select Input Directory (batch process all files)")
        print("   1.2 Set Output Directory")
        print("   1.3 View Current Data Settings")
        print()
        print("2. TEXT CLEANING & NORMALIZATION")
        print("   2.1 Clean & Normalize Text (with preview)")
        print("   2.2 Export Cleaned Text Data")
        print("   2.3 View Text Cleaning Results")
        print()
        print("3. TOKEN/PHRASE CONSTRUCTION")
        print("   3.1 Configure Phrase Parameters")
        print("   3.2 Extract Tokens & Phrases")
        print("   3.3 View Phrase Statistics")
        print()
        print("4. GLOBAL CO-OCCURRENCE GRAPH CONSTRUCTION")
        print("   4.1 Build Global Graph (shared node space)")
        print("   4.2 Apply Scientific Optimization (NPMI, sparsification, LCC)")
        print("   4.3 View Global Graph Statistics")
        print("   4.4 Export Global Graph Data")
        print()
        print("5. SUBGRAPH ACTIVATION (from Global Graph)")
        print("   5.1 Activate State-based Subgraphs")
        print("   5.2 View Subgraph Comparisons")
        print("   5.3 Export Subgraph Data")
        print()
        print("6. VISUALIZATION & EXPORT")
        print("   6.1 Generate Scientific Visualizations")
        print("   6.2 View Output Image Paths")
        print("   6.3 Export Complete Results")
        print("   6.4 View Graph Nodes & Data Details")
        print()
        print("🔧 SCIENTIFIC CONTROLS:")
        print("   S.1 Configure Scientific Parameters")
        print("   S.2 View Scientific Statistics")
        print("   S.3 Export Scientific Report")
        print()
        print("🔧 REPRODUCIBILITY CONTROLS:")
        print("   R.1 Configure Reproducibility Parameters")
        print("   R.2 View All Parameter Settings")
        print("   R.3 Export Parameter Configuration")
        print()
        print("🛠️ UTILITIES:")
        print("   U.1 Create Sample Research Data")
        print("   U.2 System & Pipeline Status")
        print("   U.3 Research Workflow Help")
        print()
        print("0. Exit Pipeline")
        print("=" * 80)
    
    def get_user_choice(self, prompt="请选择操作", valid_choices=None):
        """获取用户选择"""
        while True:
            try:
                choice = input(f"\n{prompt}: ").strip()
                if valid_choices and choice not in valid_choices:
                    print(f"❌ 无效选择，请输入: {', '.join(valid_choices)}")
                    continue
                return choice
            except KeyboardInterrupt:
                print("\n\n👋 程序被用户中断，再见！")
                sys.exit(0)
            except EOFError:
                print("\n\n👋 程序结束，再见！")
                sys.exit(0)
    
    def select_input_directory(self):
        """Select input directory for batch processing"""
        print("\n📁 INPUT DIRECTORY SELECTION")
        print("-" * 50)
        print("🔬 Research Pipeline: Directory input supports batch processing")
        print("📂 All valid files in the directory will be automatically processed")
        print()
        
        print("Input options:")
        print("1. Enter directory path (batch process all files)")
        print("2. Create sample research data directory")
        print("3. Return to main menu")
        
        choice = self.get_user_choice("Select input method", ["1", "2", "3"])
        
        if choice == "1":
            dir_path = input("\n📂 Enter directory path: ").strip()
            if not dir_path:
                print("⚠️ No directory specified")
                return
            
            if not os.path.exists(dir_path):
                print(f"❌ Directory does not exist: {dir_path}")
                return
            
            if not os.path.isdir(dir_path):
                print(f"❌ Path is not a directory: {dir_path}")
                return
            
            # Scan directory for valid files
            self.input_directory = dir_path
            self.input_files = []
            
            print(f"\n🔍 Scanning directory: {dir_path}")
            
            # Recursively find valid files
            valid_extensions = {'.json', '.txt', '.md'}
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    if file_ext in valid_extensions:
                        self.input_files.append(file_path)
                        print(f"   ✅ Found: {os.path.relpath(file_path, dir_path)}")
                    else:
                        print(f"   ⏭️ Skipped (unsupported): {os.path.relpath(file_path, dir_path)}")
            
            if self.input_files:
                print(f"\n✅ Directory scan complete: {len(self.input_files)} valid files found")
                self.pipeline_state['data_loaded'] = True
            else:
                print("⚠️ No valid files found in directory")
                print("   Supported formats: .json, .txt, .md")
        
        elif choice == "2":
            self.create_sample_research_data()
            
        elif choice == "3":
            return
    
    def set_output_directory(self):
        """Set output directory with research-oriented structure"""
        print("\n📂 OUTPUT DIRECTORY CONFIGURATION")
        print("-" * 50)
        print(f"Current output directory: {self.output_dir}")
        print("🔬 Research pipeline will create structured subdirectories:")
        print("   📁 cleaned_text/     - Text cleaning results")
        print("   📁 global_graph/     - Global co-occurrence graph")
        print("   📁 subgraphs/        - State-based subgraphs")
        print("   📁 visualizations/   - Deterministic layout images")
        print("   📁 exports/          - Final research outputs")
        print("   📁 parameters/       - Reproducibility configurations")
        print()
        
        new_dir = input("Enter new output directory (press Enter to keep current): ").strip()
        if new_dir:
            self.output_dir = new_dir
            print(f"✅ Output directory set to: {self.output_dir}")
        else:
            print("📁 Keeping current output directory")
        
        # Create directory structure
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Output directory ready: {os.path.abspath(self.output_dir)}")
    
    def show_data_settings(self):
        """Display current data input settings"""
        print("\n📋 CURRENT DATA SETTINGS")
        print("-" * 50)
        print(f"Input directory: {self.input_directory or 'Not set'}")
        print(f"Output directory: {self.output_dir}")
        print(f"Valid files found: {len(self.input_files)}")
        
        if self.input_files:
            print("\nInput files (showing first 10):")
            for i, file_path in enumerate(self.input_files[:10], 1):
                rel_path = os.path.relpath(file_path, self.input_directory) if self.input_directory else file_path
                print(f"   {i:2d}. {rel_path}")
            if len(self.input_files) > 10:
                print(f"   ... and {len(self.input_files) - 10} more files")
        else:
            print("⚠️ No input files selected")
            print("   Use option 1.1 to select input directory")
    
    def clean_and_normalize_text(self):
        """Clean and normalize text with preview capability"""
        if not self.validate_pipeline_step('data_loaded', "Please load input data first (step 1.1)"):
            return
        
        print("\n🧹 TEXT CLEANING & NORMALIZATION")
        print("-" * 50)
        print("🔬 Research Pipeline: Text cleaning with transparency and debugging")
        print(f"🌱 Using random seed: {self.reproducibility_config['random_seed']}")
        print(f"🗣️ Language detection: {self.reproducibility_config['language_detection']}")
        
        try:
            print("⏳ Loading and cleaning text data...")
            input_data = self.load_input_data()
            
            # Simulate text cleaning process (replace with actual pipeline call)
            cleaned_documents = []
            total_tokens = 0
            
            # Add progress bar for text cleaning
            for doc in tqdm(input_data, desc="🧹 Cleaning documents", unit="doc"):
                # Simulate cleaning
                cleaned_text = doc['text'].lower().strip()
                tokens = cleaned_text.split()
                total_tokens += len(tokens)
                
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
            
            self.cleaned_text_data = cleaned_documents
            
            print(f"✅ Text cleaning completed!")
            print(f"📊 Documents processed: {len(cleaned_documents)}")
            print(f"📊 Total tokens: {total_tokens}")
            print(f"📊 Average tokens per document: {total_tokens/len(cleaned_documents):.1f}")
            
            # Show preview
            print("\n📋 CLEANED TEXT PREVIEW (first document):")
            print("-" * 40)
            first_doc = cleaned_documents[0]
            print(f"Document ID: {first_doc['segment_id']}")
            print(f"Title: {first_doc['title']}")
            print(f"Original: {first_doc['original_text'][:100]}...")
            print(f"Cleaned:  {first_doc['cleaned_text'][:100]}...")
            print(f"Tokens:   {first_doc['token_count']} tokens")
            
            self.pipeline_state['text_cleaned'] = True
            
        except Exception as e:
            print(f"❌ Text cleaning failed: {e}")
    
    def export_cleaned_text(self):
        """Export cleaned text data for transparency"""
        if not self.validate_pipeline_step('text_cleaned', "Please clean text data first (step 2.1)"):
            return
        
        print("\n💾 EXPORT CLEANED TEXT DATA")
        print("-" * 50)
        
        try:
            # Create cleaned text directory
            cleaned_dir = os.path.join(self.output_dir, "cleaned_text")
            os.makedirs(cleaned_dir, exist_ok=True)
            
            # Export options
            print("Export formats:")
            print("1. JSON (structured data with metadata)")
            print("2. TXT (plain text tokens)")
            print("3. Both formats")
            
            choice = self.get_user_choice("Select export format", ["1", "2", "3"])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if choice in ["1", "3"]:
                # Export JSON
                json_file = os.path.join(cleaned_dir, f"cleaned_text_data_{timestamp}.json")
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cleaned_text_data, f, indent=2, ensure_ascii=False)
                print(f"✅ JSON exported: {json_file}")
            
            if choice in ["2", "3"]:
                # Export plain text tokens
                txt_file = os.path.join(cleaned_dir, f"cleaned_tokens_{timestamp}.txt")
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write("# Cleaned Text Tokens Export\n")
                    f.write(f"# Generated: {datetime.now().isoformat()}\n")
                    f.write(f"# Random Seed: {self.reproducibility_config['random_seed']}\n\n")
                    
                    for doc in self.cleaned_text_data:
                        f.write(f"## Document: {doc['segment_id']}\n")
                        f.write(f"# Title: {doc['title']}\n")
                        f.write(f"# Tokens: {doc['token_count']}\n")
                        f.write(" ".join(doc['tokens']) + "\n\n")
                
                print(f"✅ TXT exported: {txt_file}")
            
            print(f"📁 Cleaned text data exported to: {cleaned_dir}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def view_text_cleaning_results(self):
        """View detailed text cleaning results"""
        if not self.validate_pipeline_step('text_cleaned', "Please clean text data first (step 2.1)"):
            return
        
        print("\n📊 TEXT CLEANING RESULTS")
        print("-" * 50)
        
        if not self.cleaned_text_data:
            print("⚠️ No cleaned text data available")
            return
        
        # Statistics
        total_docs = len(self.cleaned_text_data)
        total_tokens = sum(doc['token_count'] for doc in self.cleaned_text_data)
        avg_tokens = total_tokens / total_docs
        
        print(f"📊 CLEANING STATISTICS:")
        print(f"   Documents processed: {total_docs}")
        print(f"   Total tokens: {total_tokens}")
        print(f"   Average tokens per document: {avg_tokens:.1f}")
        print(f"   Random seed used: {self.reproducibility_config['random_seed']}")
        
        # Language distribution
        languages = {}
        for doc in self.cleaned_text_data:
            lang = doc['language']
            languages[lang] = languages.get(lang, 0) + 1
        
        print(f"\n🗣️ LANGUAGE DISTRIBUTION:")
        for lang, count in languages.items():
            print(f"   {lang}: {count} documents")
        
        # State distribution
        states = {}
        for doc in self.cleaned_text_data:
            state = doc['state']
            states[state] = states.get(state, 0) + 1
        
        print(f"\n🗺️ STATE DISTRIBUTION:")
        for state, count in states.items():
            print(f"   {state}: {count} documents")
        
        # Show sample
        print(f"\n📋 SAMPLE CLEANED DOCUMENTS:")
        for i, doc in enumerate(self.cleaned_text_data[:3], 1):
            print(f"\n   {i}. {doc['segment_id']} ({doc['language']}, {doc['state']})")
            print(f"      Title: {doc['title']}")
            print(f"      Tokens: {doc['token_count']}")
            print(f"      Sample: {' '.join(doc['tokens'][:10])}...")
    
    def configure_reproducibility_parameters(self):
        """Configure all reproducibility-affecting parameters"""
        print("\n🔬 REPRODUCIBILITY PARAMETER CONFIGURATION")
        print("-" * 60)
        print("🎯 All parameters that affect reproducibility are configurable here")
        print("📋 Current settings will be saved for complete traceability")
        print()
        
        # Random seed
        print(f"🌱 Random Seed: {self.reproducibility_config['random_seed']}")
        print("   Controls: layout algorithms, sampling, any randomized processes")
        new_seed = input("Enter new random seed (press Enter to keep current): ").strip()
        if new_seed:
            try:
                self.reproducibility_config['random_seed'] = int(new_seed)
                print(f"✅ Random seed set to: {new_seed}")
            except ValueError:
                print("❌ Invalid seed, keeping current value")
        
        # Co-occurrence window definition
        print(f"\n🪟 Co-occurrence Window: {self.reproducibility_config['cooccurrence_window']}")
        print("   Options: toc_segment (one TOC segment = one window), sentence, paragraph")
        window_options = ['toc_segment', 'sentence', 'paragraph']
        print(f"   Available: {', '.join(window_options)}")
        new_window = input("Enter co-occurrence window type (press Enter to keep current): ").strip()
        if new_window and new_window in window_options:
            self.reproducibility_config['cooccurrence_window'] = new_window
            print(f"✅ Co-occurrence window set to: {new_window}")
        
        # Edge weight strategy
        print(f"\n⚖️ Edge Weight Strategy: {self.reproducibility_config['edge_weight_strategy']}")
        print("   Options: frequency_count, pmi, tfidf_weighted")
        weight_options = ['frequency_count', 'pmi', 'tfidf_weighted']
        print(f"   Available: {', '.join(weight_options)}")
        new_weight = input("Enter edge weight strategy (press Enter to keep current): ").strip()
        if new_weight and new_weight in weight_options:
            self.reproducibility_config['edge_weight_strategy'] = new_weight
            print(f"✅ Edge weight strategy set to: {new_weight}")
        
        # Phrase type
        print(f"\n📝 Phrase Type: {self.reproducibility_config['phrase_type']}")
        print("   Options: word (unigrams), bigram (2-grams), mixed (both)")
        phrase_options = ['word', 'bigram', 'mixed']
        print(f"   Available: {', '.join(phrase_options)}")
        new_phrase = input("Enter phrase type (press Enter to keep current): ").strip()
        if new_phrase and new_phrase in phrase_options:
            self.reproducibility_config['phrase_type'] = new_phrase
            print(f"✅ Phrase type set to: {new_phrase}")
        
        # Stopword strategy
        print(f"\n🚫 Stopword Strategy: {self.reproducibility_config['stopword_strategy']}")
        print("   Options: static_list (predefined), dynamic_tfidf (TF-IDF based)")
        stopword_options = ['static_list', 'dynamic_tfidf']
        print(f"   Available: {', '.join(stopword_options)}")
        new_stopword = input("Enter stopword strategy (press Enter to keep current): ").strip()
        if new_stopword and new_stopword in stopword_options:
            self.reproducibility_config['stopword_strategy'] = new_stopword
            print(f"✅ Stopword strategy set to: {new_stopword}")
        
        # Minimum phrase frequency
        print(f"\n📊 Minimum Phrase Frequency: {self.reproducibility_config['min_phrase_frequency']}")
        print("   Controls: phrase filtering threshold")
        new_freq = input("Enter minimum phrase frequency (press Enter to keep current): ").strip()
        if new_freq:
            try:
                freq = int(new_freq)
                if freq > 0:
                    self.reproducibility_config['min_phrase_frequency'] = freq
                    print(f"✅ Minimum phrase frequency set to: {freq}")
                else:
                    print("❌ Frequency must be positive")
            except ValueError:
                print("❌ Invalid frequency, keeping current value")
        
        print(f"\n✅ Reproducibility parameters updated!")
        print("💾 Use option R.3 to export these settings for documentation")
    
    def view_all_parameters(self):
        """Display all reproducibility parameters"""
        print("\n📋 ALL REPRODUCIBILITY PARAMETERS")
        print("-" * 60)
        print("🔬 These parameters affect reproducibility and are fully traceable:")
        print()
        
        for key, value in self.reproducibility_config.items():
            param_name = key.replace('_', ' ').title()
            print(f"   {param_name:25}: {value}")
        
        print(f"\n📁 Output Directory: {self.output_dir}")
        print(f"📂 Input Directory:  {self.input_directory or 'Not set'}")
        print(f"📊 Input Files:      {len(self.input_files)} files")
    
    def configure_scientific_parameters(self):
        """Configure scientific optimization parameters"""
        print("\n🔬 SCIENTIFIC PARAMETER CONFIGURATION")
        print("-" * 60)
        print("🧪 Configure rigorous network science methods")
        print()
        
        # Semantic weighting method
        print(f"⚖️ Semantic Weighting: {self.scientific_config['semantic_weighting']}")
        print("   Options: npmi (Normalized PMI), salton (Salton's Cosine), pmi (Standard PMI)")
        weighting_options = ['npmi', 'salton', 'pmi']
        new_weighting = input("Enter semantic weighting method (press Enter to keep current): ").strip()
        if new_weighting and new_weighting in weighting_options:
            self.scientific_config['semantic_weighting'] = new_weighting
            self.reproducibility_config['edge_weight_strategy'] = new_weighting
            print(f"✅ Semantic weighting set to: {new_weighting}")
        
        # Sparsification method
        print(f"\n🔍 Sparsification Method: {self.scientific_config['sparsification_method']}")
        print("   Options: quantile (top %), disparity (backbone), adaptive (best of both)")
        sparsification_options = ['quantile', 'disparity', 'adaptive']
        new_sparsification = input("Enter sparsification method (press Enter to keep current): ").strip()
        if new_sparsification and new_sparsification in sparsification_options:
            self.scientific_config['sparsification_method'] = new_sparsification
            print(f"✅ Sparsification method set to: {new_sparsification}")
        
        # Edge retention rate (for quantile method)
        print(f"\n📊 Edge Retention Rate: {self.scientific_config['edge_retention_rate']*100:.1f}%")
        print("   Controls: percentage of edges to retain in quantile sparsification")
        new_retention = input("Enter edge retention rate (0.01-0.20, press Enter to keep current): ").strip()
        if new_retention:
            try:
                retention = float(new_retention)
                if 0.01 <= retention <= 0.20:
                    self.scientific_config['edge_retention_rate'] = retention
                    print(f"✅ Edge retention rate set to: {retention*100:.1f}%")
                else:
                    print("❌ Retention rate must be between 1% and 20%")
            except ValueError:
                print("❌ Invalid retention rate")
        
        # Core identification method
        print(f"\n🎯 Core Identification: {self.scientific_config['core_method']}")
        print("   Options: k_core (K-Core decomposition), pagerank (PageRank-based)")
        core_options = ['k_core', 'pagerank']
        new_core = input("Enter core identification method (press Enter to keep current): ").strip()
        if new_core and new_core in core_options:
            self.scientific_config['core_method'] = new_core
            print(f"✅ Core identification set to: {new_core}")
        
        # Minimum community size
        print(f"\n🏘️ Minimum Community Size: {self.scientific_config['min_community_size']}")
        print("   Controls: communities smaller than this are collapsed into 'Other'")
        new_min_comm = input("Enter minimum community size (press Enter to keep current): ").strip()
        if new_min_comm:
            try:
                min_comm = int(new_min_comm)
                if min_comm > 0:
                    self.scientific_config['min_community_size'] = min_comm
                    print(f"✅ Minimum community size set to: {min_comm}")
                else:
                    print("❌ Community size must be positive")
            except ValueError:
                print("❌ Invalid community size")
        
        # Update scientific optimizer if available
        if self.scientific_optimizer:
            self.scientific_optimizer.config.update(self.scientific_config)
            print(f"\n✅ Scientific parameters updated!")
        
        print("💾 Use option S.3 to export these settings for documentation")
    
    def view_scientific_statistics(self):
        """View comprehensive scientific statistics"""
        print("\n📊 SCIENTIFIC STATISTICS & ANALYSIS")
        print("-" * 60)
        
        if not hasattr(self, 'structural_statistics') or not self.structural_statistics:
            print("⚠️ No scientific statistics available. Run step 4.2 first.")
            return
        
        stats = self.structural_statistics
        
        print("🔬 STRUCTURAL STATISTICS:")
        print(f"   Nodes: {stats.get('nodes', 0)}")
        print(f"   Edges: {stats.get('edges', 0)}")
        print(f"   Density: {stats.get('density', 0):.6f}")
        print(f"   Connected Components: {stats.get('components', 0)}")
        
        if 'largest_component_size' in stats:
            print(f"   Largest Component: {stats['largest_component_size']} nodes ({stats.get('largest_component_fraction', 0)*100:.1f}%)")
        
        if 'average_clustering' in stats:
            print(f"   Average Clustering: {stats['average_clustering']:.4f}")
            print(f"   Transitivity: {stats['transitivity']:.4f}")
        
        if 'average_path_length' in stats:
            print(f"   Average Path Length: {stats['average_path_length']:.2f}")
            print(f"   Diameter: {stats.get('diameter', 'N/A')}")
        
        if 'average_degree' in stats:
            print(f"   Average Degree: {stats['average_degree']:.2f} ± {stats.get('degree_std', 0):.2f}")
            print(f"   Degree Range: {stats.get('min_degree', 0)} - {stats.get('max_degree', 0)}")
        
        if 'centralization' in stats:
            print(f"   Network Centralization: {stats['centralization']:.4f}")
        
        # Community statistics
        if hasattr(self, 'global_communities') and self.global_communities:
            from collections import Counter
            community_sizes = Counter(self.global_communities.values())
            print(f"\n🏘️ COMMUNITY STRUCTURE:")
            print(f"   Number of Communities: {len(community_sizes)}")
            print(f"   Largest Communities: {dict(sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        # Core-periphery statistics
        if hasattr(self, 'global_node_roles') and self.global_node_roles:
            from collections import Counter
            role_counts = Counter(self.global_node_roles.values())
            print(f"\n🎯 CORE-PERIPHERY STRUCTURE:")
            for role, count in role_counts.items():
                pct = count / len(self.global_node_roles) * 100
                print(f"   {role.title()} nodes: {count} ({pct:.1f}%)")
        
        # Top phrases by weighted degree
        if hasattr(self, 'optimized_global_graph') and self.optimized_global_graph:
            if self.scientific_optimizer:
                top_phrases = self.scientific_optimizer.get_top_phrases_by_weighted_degree(self.optimized_global_graph, 10)
                print(f"\n📈 TOP 10 PHRASES BY WEIGHTED DEGREE:")
                for i, (phrase, w_degree, tfidf) in enumerate(top_phrases, 1):
                    print(f"   {i:2d}. {phrase} (degree: {w_degree:.2f}, TF-IDF: {tfidf:.2f})")
        
        # Scientific method summary
        print(f"\n🔬 SCIENTIFIC METHODS APPLIED:")
        print(f"   Semantic Weighting: {self.scientific_config['semantic_weighting'].upper()}")
        print(f"   Sparsification: {self.scientific_config['sparsification_method']}")
        print(f"   Core Identification: {self.scientific_config['core_method']}")
        print(f"   LCC Extraction: {'Yes' if self.scientific_config.get('enable_lcc_extraction', True) else 'No'}")
        print(f"   Community Pruning: {'Yes' if self.scientific_config.get('enable_community_pruning', True) else 'No'}")
    
    def export_scientific_report(self):
        """Export comprehensive scientific report"""
        print("\n💾 EXPORT SCIENTIFIC REPORT")
        print("-" * 50)
        
        try:
            # Create scientific reports directory
            reports_dir = os.path.join(self.output_dir, "scientific_reports")
            os.makedirs(reports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = os.path.join(reports_dir, f"scientific_analysis_report_{timestamp}.json")
            
            # Prepare comprehensive report data
            report_data = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_version': '5.0.0',
                    'report_type': 'scientific_analysis'
                },
                'scientific_parameters': self.scientific_config.copy(),
                'reproducibility_config': self.reproducibility_config.copy(),
                'structural_statistics': self.structural_statistics if hasattr(self, 'structural_statistics') else {},
                'community_analysis': {},
                'core_periphery_analysis': {},
                'top_phrases': []
            }
            
            # Add community analysis
            if hasattr(self, 'global_communities') and self.global_communities:
                from collections import Counter
                community_sizes = Counter(self.global_communities.values())
                report_data['community_analysis'] = {
                    'total_communities': len(community_sizes),
                    'community_sizes': dict(community_sizes),
                    'largest_communities': dict(sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:10])
                }
            
            # Add core-periphery analysis
            if hasattr(self, 'global_node_roles') and self.global_node_roles:
                from collections import Counter
                role_counts = Counter(self.global_node_roles.values())
                report_data['core_periphery_analysis'] = {
                    'role_distribution': dict(role_counts),
                    'core_percentage': role_counts.get('core', 0) / len(self.global_node_roles) * 100
                }
            
            # Add top phrases analysis
            if hasattr(self, 'optimized_global_graph') and self.optimized_global_graph and self.scientific_optimizer:
                top_phrases = self.scientific_optimizer.get_top_phrases_by_weighted_degree(self.optimized_global_graph, 20)
                report_data['top_phrases'] = [
                    {'phrase': phrase, 'weighted_degree': float(w_degree), 'tfidf_score': float(tfidf)}
                    for phrase, w_degree, tfidf in top_phrases
                ]
            
            # Export report
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Scientific report exported: {report_file}")
            print("📋 Report includes:")
            print("   - Scientific parameters and methods")
            print("   - Structural network statistics")
            print("   - Community structure analysis")
            print("   - Core-periphery identification")
            print("   - Top phrases by semantic importance")
            print("🔬 This report provides complete scientific documentation")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
        """Export parameter configuration for reproducibility"""
        print("\n💾 EXPORT PARAMETER CONFIGURATION")
        print("-" * 50)
        
        try:
            # Create parameters directory
            params_dir = os.path.join(self.output_dir, "parameters")
            os.makedirs(params_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_file = os.path.join(params_dir, f"reproducibility_config_{timestamp}.json")
            
            # Prepare configuration data
            config_data = {
                'export_timestamp': datetime.now().isoformat(),
                'pipeline_version': '4.0.0',
                'reproducibility_parameters': self.reproducibility_config.copy(),
                'input_settings': {
                    'input_directory': self.input_directory,
                    'input_files_count': len(self.input_files),
                    'output_directory': self.output_dir
                },
                'pipeline_state': self.pipeline_state.copy()
            }
            
            # Export configuration
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Configuration exported: {config_file}")
            print("📋 This file contains all parameters needed to reproduce results")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def validate_pipeline_step(self, required_step, error_message):
        """Validate that required pipeline step is completed"""
        if not self.pipeline_state.get(required_step, False):
            print(f"⚠️ WORKFLOW ERROR: {error_message}")
            print("🔬 Research Pipeline: Steps must be executed in order for reproducibility")
            return False
        return True
    
    def configure_phrase_parameters(self):
        """Configure phrase extraction parameters"""
        if not self.validate_pipeline_step('text_cleaned', "Please clean text data first (step 2.1)"):
            return
        
        print("\n📝 PHRASE PARAMETER CONFIGURATION")
        print("-" * 50)
        print("🔬 Configure parameters for token and phrase construction")
        print()
        
        # Show current phrase settings
        print("CURRENT PHRASE SETTINGS:")
        print(f"   Phrase Type: {self.reproducibility_config['phrase_type']}")
        print(f"   Min Frequency: {self.reproducibility_config['min_phrase_frequency']}")
        print(f"   Stopword Strategy: {self.reproducibility_config['stopword_strategy']}")
        print()
        
        print("These parameters are also configurable in Reproducibility Controls (R.1)")
        print("✅ Phrase parameters ready for extraction")
    
    def extract_tokens_and_phrases(self):
        """Extract tokens and phrases from cleaned text"""
        if not self.validate_pipeline_step('text_cleaned', "Please clean text data first (step 2.1)"):
            return
        
        print("\n🔍 TOKEN & PHRASE EXTRACTION")
        print("-" * 50)
        print(f"🌱 Using random seed: {self.reproducibility_config['random_seed']}")
        print(f"📝 Phrase type: {self.reproducibility_config['phrase_type']}")
        print(f"📊 Min frequency: {self.reproducibility_config['min_phrase_frequency']}")
        
        try:
            # Simulate phrase extraction (replace with actual pipeline call)
            print("⏳ Extracting tokens and phrases...")
            
            all_phrases = []
            phrase_counts = {}
            
            for doc in tqdm(self.cleaned_text_data, desc="🔍 Extracting phrases", unit="doc"):
                tokens = doc['tokens']
                
                # Extract unigrams
                if self.reproducibility_config['phrase_type'] in ['word', 'mixed']:
                    for token in tokens:
                        if len(token) > 2:  # Filter short tokens
                            all_phrases.append(token)
                            phrase_counts[token] = phrase_counts.get(token, 0) + 1
                
                # Extract bigrams
                if self.reproducibility_config['phrase_type'] in ['bigram', 'mixed']:
                    for i in range(len(tokens) - 1):
                        bigram = f"{tokens[i]} {tokens[i+1]}"
                        all_phrases.append(bigram)
                        phrase_counts[bigram] = phrase_counts.get(bigram, 0) + 1
            
            # Filter by minimum frequency
            min_freq = self.reproducibility_config['min_phrase_frequency']
            filtered_phrases = {phrase: count for phrase, count in phrase_counts.items() 
                              if count >= min_freq}
            
            self.phrase_data = {
                'all_phrases': all_phrases,
                'phrase_counts': phrase_counts,
                'filtered_phrases': filtered_phrases,
                'extraction_params': self.reproducibility_config.copy()
            }
            
            print(f"✅ Phrase extraction completed!")
            print(f"📊 Total phrase instances: {len(all_phrases)}")
            print(f"📊 Unique phrases: {len(phrase_counts)}")
            print(f"📊 Phrases above threshold: {len(filtered_phrases)}")
            
            self.pipeline_state['phrases_constructed'] = True
            
        except Exception as e:
            print(f"❌ Phrase extraction failed: {e}")
    
    def view_phrase_statistics(self):
        """View phrase extraction statistics"""
        if not self.validate_pipeline_step('phrases_constructed', "Please extract phrases first (step 3.2)"):
            return
        
        print("\n📊 PHRASE STATISTICS")
        print("-" * 50)
        
        if not hasattr(self, 'phrase_data'):
            print("⚠️ No phrase data available")
            return
        
        phrase_counts = self.phrase_data['phrase_counts']
        filtered_phrases = self.phrase_data['filtered_phrases']
        
        print(f"📊 EXTRACTION RESULTS:")
        print(f"   Total phrase instances: {len(self.phrase_data['all_phrases'])}")
        print(f"   Unique phrases: {len(phrase_counts)}")
        print(f"   Phrases above threshold: {len(filtered_phrases)}")
        print(f"   Minimum frequency threshold: {self.phrase_data['extraction_params']['min_phrase_frequency']}")
        
        # Top phrases
        sorted_phrases = sorted(filtered_phrases.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🔝 TOP 10 PHRASES:")
        for i, (phrase, count) in enumerate(sorted_phrases[:10], 1):
            print(f"   {i:2d}. {phrase} (count: {count})")
    
    def build_global_graph(self):
        """Build global co-occurrence graph as a true NetworkX graph object (shared node space)"""
        if not self.validate_pipeline_step('phrases_constructed', "Please extract phrases first (step 3.2)"):
            return
        
        print("\n🌐 GLOBAL CO-OCCURRENCE GRAPH CONSTRUCTION")
        print("-" * 60)
        print("🔬 Building shared node space as NetworkX graph object")
        print(f"🪟 Co-occurrence window: {self.reproducibility_config['cooccurrence_window']}")
        print(f"⚖️ Edge weight strategy: {self.reproducibility_config['edge_weight_strategy']}")
        
        # Add graph construction parameters for structural filtering
        # Add graph construction parameters for structural filtering - LESS AGGRESSIVE
        self.graph_construction_config = {
            'edge_density_reduction': 0.5,  # Keep top 50% of edges by weight (was 0.1)
            'min_edge_weight': 1,  # Lower minimum co-occurrence count (was 2)
            'core_node_percentile': 0.3,  # Top 30% nodes are "core" (was 0.2)
            'community_layout_separation': 2.0,  # Separation factor between communities
            'sliding_window_size': 5,  # Sliding window for co-occurrence
            'min_cooccurrence_threshold': 1,  # Lower minimum global co-occurrence threshold (was 3)
        }
        
        try:
            print("⏳ Constructing global co-occurrence NetworkX graph...")
            
            # Set random seed for deterministic layout
            np.random.seed(self.reproducibility_config['random_seed'])
            
            filtered_phrases = self.phrase_data['filtered_phrases']
            
            # A. STRUCTURAL TOKEN FILTERING - Remove structural tokens before node creation
            print("🔧 Applying structural token filtering...")
            
            import re
            structural_patterns = [
                r'^\d+(\.\d+)*$',  # Pure TOC numbering: 1, 1.1, 1.2.3
                r'^\d+\.$',        # Numbered items: 1., 2., 3.
                r'^\d{4}[-–]\d{4}$',  # Year ranges: 2024-2025, 2024–2025
                r'^\d{4}$',        # Single years: 2024, 2025
            ]
            
            # Compile patterns for efficiency
            compiled_patterns = [re.compile(pattern) for pattern in structural_patterns]
            
            # Stopwords for semantic phrase filtering
            stopwords = {'and', 'or', 'of', 'the', 'are', 'be', 'to', 'for', 'with', 'in', 'on', 'at', 'by', 'from', 'as', 'is', 'was', 'will', 'can', 'may', 'shall', 'should', 'would', 'could'}
            
            semantically_filtered_phrases = {}
            structural_tokens_removed = 0
            semantic_tokens_removed = 0
            
            for phrase, count in filtered_phrases.items():
                # Check for structural patterns
                is_structural = False
                
                # Check pure structural patterns
                for pattern in compiled_patterns:
                    if pattern.match(phrase.strip()):
                        is_structural = True
                        break
                
                # Check for section numbers as suffix/prefix
                if not is_structural:
                    words = phrase.split()
                    for word in words:
                        # Check if word contains section numbers
                        if re.search(r'\d+\.?\d*\.?$', word) or re.search(r'^\d+\.?\d*\.?', word):
                            is_structural = True
                            break
                
                if is_structural:
                    structural_tokens_removed += 1
                    continue
                
                # B. SEMANTIC PHRASE FILTERING - Apply semantic rules
                words = phrase.lower().split()
                
                # Remove phrases starting or ending with stopwords
                if words[0] in stopwords or words[-1] in stopwords:
                    semantic_tokens_removed += 1
                    continue
                
                # For bigrams, check if head is conjunction or auxiliary verb
                if len(words) == 2:
                    conjunctions = {'and', 'or', 'but', 'yet', 'so', 'nor'}
                    auxiliaries = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'can', 'could', 'may', 'might', 'shall', 'should'}
                    
                    if words[0] in conjunctions or words[0] in auxiliaries:
                        semantic_tokens_removed += 1
                        continue
                
                # Keep only phrases with at least one content word (simplified check)
                content_indicators = any(len(word) > 3 for word in words)  # Simple heuristic for content words
                if not content_indicators:
                    semantic_tokens_removed += 1
                    continue
                
                # Phrase passed all filters
                semantically_filtered_phrases[phrase] = count
            
            print(f"   📊 Original phrases: {len(filtered_phrases)}")
            print(f"   📊 Structural tokens removed: {structural_tokens_removed}")
            print(f"   📊 Semantic tokens removed: {semantic_tokens_removed}")
            print(f"   📊 Final phrases: {len(semantically_filtered_phrases)}")
            
            phrase_list = list(semantically_filtered_phrases.keys())
            
            # CREATE NETWORKX GRAPH OBJECT (not just adjacency data)
            self.global_graph_object = nx.Graph()
            
            # Add all phrases as nodes with semantic attributes
            for phrase in phrase_list:
                # Calculate TF-IDF score (simplified)
                frequency = semantically_filtered_phrases[phrase]
                total_docs = len(self.cleaned_text_data)
                doc_frequency = sum(1 for doc in self.cleaned_text_data if phrase in ' '.join(doc['tokens']))
                
                # Simple TF-IDF calculation
                tf = frequency
                idf = np.log(total_docs / (doc_frequency + 1))  # +1 to avoid division by zero
                tfidf_score = tf * idf
                
                self.global_graph_object.add_node(
                    phrase, 
                    raw_phrase=phrase,
                    frequency=frequency,
                    tf_idf_score=tfidf_score,
                    is_structural=False,  # All remaining phrases are non-structural
                    phrase_type='bigram' if ' ' in phrase else 'unigram'
                )
            
            # C. CONSTRAINED CO-OCCURRENCE EDGE CREATION - Use sliding window approach
            print("🔧 Computing co-occurrences with sliding window constraint...")
            cooccurrence_counts = defaultdict(int)
            
            # Process each document for co-occurrences with sliding window
            for doc in tqdm(self.cleaned_text_data, desc="🌐 Building co-occurrences", unit="doc"):
                tokens = doc['tokens']
                
                # Extract valid phrases from this document
                doc_phrases = []
                
                # Extract unigrams
                if self.reproducibility_config['phrase_type'] in ['word', 'mixed']:
                    doc_phrases.extend([token for token in tokens if token in semantically_filtered_phrases])
                
                # Extract bigrams
                if self.reproducibility_config['phrase_type'] in ['bigram', 'mixed']:
                    for i in range(len(tokens) - 1):
                        bigram = f"{tokens[i]} {tokens[i+1]}"
                        if bigram in semantically_filtered_phrases:
                            doc_phrases.append(bigram)
                
                # Apply sliding window for co-occurrence calculation
                window_size = self.graph_construction_config['sliding_window_size']
                
                for i in range(len(doc_phrases)):
                    # Define window boundaries
                    window_start = max(0, i - window_size // 2)
                    window_end = min(len(doc_phrases), i + window_size // 2 + 1)
                    
                    phrase1 = doc_phrases[i]
                    
                    # Count co-occurrences within window
                    for j in range(window_start, window_end):
                        if i != j:
                            phrase2 = doc_phrases[j]
                            if phrase1 != phrase2:
                                edge = tuple(sorted([phrase1, phrase2]))
                                cooccurrence_counts[edge] += 1
            
            # Apply minimum co-occurrence threshold
            min_cooccurrence = self.graph_construction_config['min_cooccurrence_threshold']
            filtered_cooccurrences = {edge: count for edge, count in cooccurrence_counts.items() 
                                    if count >= min_cooccurrence}
            
            print(f"   📊 Raw co-occurrences: {len(cooccurrence_counts)}")
            print(f"   📊 After min threshold ({min_cooccurrence}): {len(filtered_cooccurrences)}")
            
            # STRUCTURAL FILTERING: Apply edge filtering at construction time
            print("🔧 Applying structural filtering to reduce graph density...")
            
            # Filter edges by minimum weight threshold
            min_weight = self.graph_construction_config['min_edge_weight']
            weight_filtered_edges = {edge: weight for edge, weight in filtered_cooccurrences.items() 
                                   if weight >= min_weight}
            
            print(f"   📊 After min weight filter ({min_weight}): {len(weight_filtered_edges)}")
            
            # Apply density reduction: keep only top percentile of edges by weight
            if weight_filtered_edges:
                edge_weights = list(weight_filtered_edges.values())
                density_threshold = np.percentile(edge_weights, 
                                                (1 - self.graph_construction_config['edge_density_reduction']) * 100)
                
                final_edges = {edge: weight for edge, weight in weight_filtered_edges.items() 
                             if weight >= density_threshold}
                
                print(f"   📊 After density reduction ({self.graph_construction_config['edge_density_reduction']*100:.1f}%): {len(final_edges)}")
            else:
                final_edges = {}
            
            # Add filtered edges to NetworkX graph
            for (phrase1, phrase2), weight in final_edges.items():
                self.global_graph_object.add_edge(phrase1, phrase2, weight=weight, raw_weight=weight)
            
            # Store raw co-occurrence counts for reference
            self.raw_cooccurrence_counts = cooccurrence_counts
            
            # COMPUTE NODE IMPORTANCE AND ROLES
            print("📊 Computing node importance and roles...")
            
            with tqdm(total=3, desc="📊 Node importance computation", unit="measure") as pbar:
                pbar.set_description("📊 Computing degree centrality")
                degree_centrality = nx.degree_centrality(self.global_graph_object)
                pbar.update(1)
                
                pbar.set_description("📊 Computing weighted degree")
                weighted_degree = dict(self.global_graph_object.degree(weight='weight'))
                # Normalize weighted degree
                max_weighted_degree = max(weighted_degree.values()) if weighted_degree else 1
                weighted_degree_norm = {node: deg/max_weighted_degree for node, deg in weighted_degree.items()}
                pbar.update(1)
                
                pbar.set_description("📊 Computing PageRank")
                try:
                    pagerank = nx.pagerank(self.global_graph_object, weight='weight')
                except:
                    # Fallback to degree centrality if PageRank fails
                    pagerank = degree_centrality
                pbar.update(1)
            
            # Assign node roles based on importance
            print("🎭 Assigning node roles (core vs periphery)...")
            
            # Combine multiple importance measures
            node_importance = {}
            for node in self.global_graph_object.nodes():
                importance = (
                    0.4 * degree_centrality.get(node, 0) +
                    0.4 * weighted_degree_norm.get(node, 0) +
                    0.2 * pagerank.get(node, 0)
                )
                node_importance[node] = importance
            
            # Determine core nodes (top percentile)
            importance_threshold = np.percentile(list(node_importance.values()), 
                                               (1 - self.graph_construction_config['core_node_percentile']) * 100)
            
            node_roles = {}
            core_nodes = []
            for node, importance in node_importance.items():
                if importance >= importance_threshold:
                    node_roles[node] = 'core'
                    core_nodes.append(node)
                else:
                    node_roles[node] = 'periphery'
            
            print(f"   🎯 Core nodes: {len(core_nodes)} ({len(core_nodes)/len(node_importance)*100:.1f}%)")
            print(f"   🌐 Periphery nodes: {len(node_importance) - len(core_nodes)}")
            
            # Store node attributes
            nx.set_node_attributes(self.global_graph_object, degree_centrality, 'degree_centrality')
            nx.set_node_attributes(self.global_graph_object, weighted_degree_norm, 'weighted_degree')
            nx.set_node_attributes(self.global_graph_object, pagerank, 'pagerank')
            nx.set_node_attributes(self.global_graph_object, node_importance, 'importance')
            nx.set_node_attributes(self.global_graph_object, node_roles, 'role')
            
            # COMPUTE DETERMINISTIC 2D LAYOUT (core requirement)
            print("🎯 Computing deterministic 2D layout...")
            
            # 修复的布局计算 - 分批显示真实进度
            iterations = 50
            batch_size = 10
            with tqdm(total=iterations, desc="🎯 Spring layout进度", unit="iter") as pbar:
                pos = None
                for i in range(0, iterations, batch_size):
                    current_iterations = min(batch_size, iterations - i)
                    
                    if pos is None:
                        pos = nx.spring_layout(
                            self.global_graph_object,
                            k=1.0,
                            iterations=current_iterations,
                            seed=self.reproducibility_config['random_seed']
                        )
                    else:
                        pos = nx.spring_layout(
                            self.global_graph_object,
                            k=1.0,
                            iterations=current_iterations,
                            pos=pos,
                            seed=self.reproducibility_config['random_seed']
                        )
                    
                    pbar.update(current_iterations)
                    time.sleep(0.02)  # 短暂延迟显示进度
                
                self.global_layout_positions = pos
            
            # Store positions as node attributes for persistence
            nx.set_node_attributes(self.global_graph_object, self.global_layout_positions, 'pos')
            
            # COMMUNITY DETECTION ON FILTERED GRAPH
            print("🏘️ Detecting communities on filtered graph...")
            try:
                with tqdm(total=1, desc="🏘️ Community detection", unit="step") as pbar:
                    communities = nx.community.greedy_modularity_communities(self.global_graph_object)
                    pbar.update(1)
                    
                community_map = {}
                with tqdm(communities, desc="🏘️ Assigning communities", unit="community") as pbar:
                    for i, community in enumerate(pbar):
                        for node in community:
                            community_map[node] = i
                
                nx.set_node_attributes(self.global_graph_object, community_map, 'community')
                print(f"   Found {len(communities)} communities")
                
                # COMMUNITY-AWARE LAYOUT REFINEMENT
                print("🎨 Refining layout with community separation...")
                self.global_layout_positions = self._compute_community_aware_layout(
                    self.global_graph_object, communities, self.global_layout_positions
                )
                nx.set_node_attributes(self.global_graph_object, self.global_layout_positions, 'pos')
                
            except:
                # Fallback: assign all nodes to community 0
                with tqdm(total=1, desc="🏘️ Fallback community assignment", unit="step") as pbar:
                    community_map = {node: 0 for node in self.global_graph_object.nodes()}
                    nx.set_node_attributes(self.global_graph_object, community_map, 'community')
                    pbar.update(1)
                print("   Using single community (fallback)")
            
            # Create legacy data structure for backward compatibility
            cooccurrence_matrix = {}
            for phrase1, phrase2, data in self.global_graph_object.edges(data=True):
                pair_key = f"{sorted([phrase1, phrase2])[0]}|||{sorted([phrase1, phrase2])[1]}"
                cooccurrence_matrix[pair_key] = data['weight']
            
            self.global_graph = {
                'nodes': phrase_list,
                'edges': cooccurrence_matrix,
                'node_count': len(phrase_list),
                'edge_count': len(cooccurrence_matrix),
                'construction_params': self.reproducibility_config.copy(),
                'construction_timestamp': datetime.now().isoformat(),
                'structural_params': self.graph_construction_config.copy()
            }
            
            print(f"✅ Global NetworkX graph construction completed!")
            print(f"🌐 Graph nodes: {self.global_graph_object.number_of_nodes()}")
            print(f"🌐 Graph edges: {self.global_graph_object.number_of_edges()}")
            
            if self.global_graph_object.number_of_nodes() > 1:
                density = nx.density(self.global_graph_object)
                print(f"🌐 Graph density: {density * 100:.2f}%")
            
            print(f"🎯 Layout positions computed and stored")
            print(f"🏘️ Community structure detected and stored")
            print(f"🎭 Node roles assigned (core/periphery)")
            
            self.pipeline_state['global_graph_built'] = True
            
        except Exception as e:
            print(f"❌ Global graph construction failed: {e}")
            import traceback
            traceback.print_exc()
    
    def apply_scientific_optimization(self):
        """Apply scientific optimization to the global graph"""
        if not self.validate_pipeline_step('global_graph_built', "Please build global graph first (step 4.1)"):
            return
        
        print("\n🔬 SCIENTIFIC GRAPH OPTIMIZATION")
        print("-" * 60)
        print("🧪 Applying rigorous network science methods")
        print(f"⚖️ Semantic weighting: {self.scientific_config['semantic_weighting'].upper()}")
        print(f"🔍 Sparsification: {self.scientific_config['sparsification_method']}")
        print(f"🎯 Core identification: {self.scientific_config['core_method']}")
        
        try:
            if not self.scientific_optimizer:
                print("❌ Scientific optimizer not initialized")
                return
            
            # Prepare phrase frequencies for semantic weighting
            phrase_frequencies = self.phrase_data['filtered_phrases']
            total_phrases = len(self.phrase_data['all_phrases'])
            
            print(f"📊 Input graph: {self.global_graph_object.number_of_nodes()} nodes, {self.global_graph_object.number_of_edges()} edges")
            
            # Apply scientific optimization
            (self.optimized_global_graph, 
             self.global_communities, 
             self.global_node_roles, 
             self.global_layout_positions, 
             self.structural_statistics) = self.scientific_optimizer.optimize_graph(
                self.global_graph_object, 
                phrase_frequencies, 
                total_phrases
            )
            
            # Update the main graph object with optimized version
            self.global_graph_object = self.optimized_global_graph
            
            print(f"\n✅ Scientific optimization completed!")
            print(f"📊 Optimized graph: {self.optimized_global_graph.number_of_nodes()} nodes, {self.optimized_global_graph.number_of_edges()} edges")
            print(f"📊 Density improvement: {self.structural_statistics.get('density', 0):.6f}")
            print(f"🏘️ Communities detected: {len(set(self.global_communities.values()))}")
            print(f"🎯 Core nodes identified: {sum(1 for role in self.global_node_roles.values() if role == 'core')}")
            
            # Update pipeline state
            self.pipeline_state['scientifically_optimized'] = True
            
        except Exception as e:
            print(f"❌ Scientific optimization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _compute_community_aware_layout(self, graph, communities, initial_positions):
        """Compute community-aware layout with separation between communities"""
        
        # Calculate community centers
        community_centers = {}
        for i, community in enumerate(communities):
            if len(community) > 0:
                # Get average position of nodes in this community
                x_coords = [initial_positions[node][0] for node in community if node in initial_positions]
                y_coords = [initial_positions[node][1] for node in community if node in initial_positions]
                
                if x_coords and y_coords:
                    center_x = np.mean(x_coords)
                    center_y = np.mean(y_coords)
                    community_centers[i] = (center_x, center_y)
        
        # Separate community centers
        separation_factor = self.graph_construction_config['community_layout_separation']
        
        if len(community_centers) > 1:
            # Arrange community centers in a circle for better separation
            n_communities = len(community_centers)
            angle_step = 2 * np.pi / n_communities
            
            new_centers = {}
            for i, comm_id in enumerate(community_centers.keys()):
                angle = i * angle_step
                new_x = separation_factor * np.cos(angle)
                new_y = separation_factor * np.sin(angle)
                new_centers[comm_id] = (new_x, new_y)
            
            # Adjust node positions based on new community centers
            refined_positions = {}
            for i, community in enumerate(communities):
                if i in new_centers and i in community_centers:
                    old_center = community_centers[i]
                    new_center = new_centers[i]
                    
                    # Translate nodes in this community
                    offset_x = new_center[0] - old_center[0]
                    offset_y = new_center[1] - old_center[1]
                    
                    for node in community:
                        if node in initial_positions:
                            old_pos = initial_positions[node]
                            refined_positions[node] = (
                                old_pos[0] + offset_x,
                                old_pos[1] + offset_y
                            )
            
            # Fill in any missing nodes
            for node in graph.nodes():
                if node not in refined_positions:
                    refined_positions[node] = initial_positions.get(node, (0, 0))
            
            return refined_positions
        
        return initial_positions
    
    def view_global_graph_statistics(self):
        """View global graph statistics computed directly from NetworkX graph object"""
        if not self.validate_pipeline_step('global_graph_built', "Please build global graph first (step 4.1)"):
            return
        
        print("\n📊 GLOBAL GRAPH STATISTICS")
        print("-" * 50)
        
        if self.global_graph_object is None:
            print("⚠️ No global graph object available")
            return
        
        G = self.global_graph_object
        
        print(f"🌐 GLOBAL GRAPH STRUCTURE (NetworkX object):")
        print(f"   Nodes (phrases): {G.number_of_nodes()}")
        print(f"   Edges (co-occurrences): {G.number_of_edges()}")
        
        # Show structural filtering impact
        if hasattr(self, 'phrase_data') and 'filtered_phrases' in self.phrase_data:
            original_phrases = len(self.phrase_data['filtered_phrases'])
            current_nodes = G.number_of_nodes()
            structural_removed = original_phrases - current_nodes
            print(f"   Original phrases (before structural filtering): {original_phrases}")
            print(f"   Structural tokens removed: {structural_removed}")
            print(f"   Structural filtering reduction: {structural_removed/original_phrases*100:.1f}%")
        
        # Show edge filtering impact
        if hasattr(self, 'raw_cooccurrence_counts'):
            raw_edge_count = len(self.raw_cooccurrence_counts)
            filtered_edge_count = G.number_of_edges()
            reduction_pct = (1 - filtered_edge_count/raw_edge_count) * 100 if raw_edge_count > 0 else 0
            print(f"   Raw edges (before filtering): {raw_edge_count}")
            print(f"   Filtered edges: {filtered_edge_count}")
            print(f"   Edge reduction: {reduction_pct:.1f}%")
        
        # Density comparison (before vs after filtering)
        if G.number_of_nodes() > 1:
            current_density = nx.density(G)
            print(f"   Current graph density: {current_density * 100:.2f}%")
            
            # Calculate theoretical density before filtering
            if hasattr(self, 'raw_cooccurrence_counts') and hasattr(self, 'phrase_data'):
                original_nodes = len(self.phrase_data.get('filtered_phrases', {}))
                if original_nodes > 1:
                    max_possible_edges = original_nodes * (original_nodes - 1) / 2
                    raw_edges = len(self.raw_cooccurrence_counts)
                    original_density = raw_edges / max_possible_edges
                    print(f"   Density before filtering: {original_density * 100:.2f}%")
                    print(f"   Density reduction: {(original_density - current_density) * 100:.2f} percentage points")
        
        # Connected components analysis
        if G.number_of_nodes() > 0:
            components = list(nx.connected_components(G))
            print(f"   Connected components: {len(components)}")
            if len(components) > 1:
                largest_cc = max(components, key=len)
                print(f"   Largest component size: {len(largest_cc)} nodes")
                
                # Show component size distribution
                component_sizes = sorted([len(comp) for comp in components], reverse=True)
                print(f"   Component sizes: {component_sizes[:5]}{'...' if len(component_sizes) > 5 else ''}")
            
            # Isolated nodes count
            isolated_nodes = list(nx.isolates(G))
            print(f"   Isolated nodes: {len(isolated_nodes)}")
        
        # Community structure
        communities = set(nx.get_node_attributes(G, 'community').values())
        if communities:
            print(f"\n🏘️ COMMUNITY STRUCTURE:")
            print(f"   Number of communities: {len(communities)}")
            
            # Community size distribution
            community_sizes = defaultdict(int)
            for node, community in nx.get_node_attributes(G, 'community').items():
                community_sizes[community] += 1
            
            sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
            print(f"   Largest communities: {dict(sorted_communities[:5])}")
        
        # Node roles analysis
        node_roles = nx.get_node_attributes(G, 'role')
        if node_roles:
            role_counts = defaultdict(int)
            for role in node_roles.values():
                role_counts[role] += 1
            
            print(f"\n🎭 NODE ROLES:")
            for role, count in role_counts.items():
                pct = count / len(node_roles) * 100
                print(f"   {role.title()} nodes: {count} ({pct:.1f}%)")
        
        # Semantic attributes analysis
        tf_idf_scores = nx.get_node_attributes(G, 'tf_idf_score')
        if tf_idf_scores:
            print(f"\n📊 SEMANTIC ATTRIBUTES:")
            print(f"   Nodes with TF-IDF scores: {len(tf_idf_scores)}")
            print(f"   Average TF-IDF score: {np.mean(list(tf_idf_scores.values())):.3f}")
            print(f"   Max TF-IDF score: {max(tf_idf_scores.values()):.3f}")
            
            # Top nodes by TF-IDF
            top_tfidf = sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"   Top 5 by TF-IDF:")
            for i, (node, score) in enumerate(top_tfidf, 1):
                print(f"     {i}. {node} ({score:.3f})")
        
        # Edge weight distribution
        if G.number_of_edges() > 0:
            edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
            print(f"\n⚖️ EDGE WEIGHT DISTRIBUTION:")
            print(f"   Min weight: {min(edge_weights)}")
            print(f"   Max weight: {max(edge_weights)}")
            print(f"   Average weight: {sum(edge_weights) / len(edge_weights):.2f}")
            print(f"   Median weight: {np.median(edge_weights):.2f}")
        
        # Top important nodes by different measures
        importance_scores = nx.get_node_attributes(G, 'importance')
        if importance_scores:
            top_important_nodes = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n📊 TOP 5 IMPORTANT NODES (combined score):")
            for i, (node, importance) in enumerate(top_important_nodes, 1):
                role = node_roles.get(node, 'unknown')
                tfidf = tf_idf_scores.get(node, 0) if tf_idf_scores else 0
                print(f"   {i}. {node} (importance: {importance:.3f}, TF-IDF: {tfidf:.3f}, role: {role})")
        
        # Top co-occurring pairs (from NetworkX edges)
        if G.number_of_edges() > 0:
            sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
            print(f"\n🔗 TOP 10 CO-OCCURRING PHRASE PAIRS:")
            for i, (phrase1, phrase2, data) in enumerate(sorted_edges[:10], 1):
                print(f"   {i:2d}. {phrase1} ↔ {phrase2} (weight: {data['weight']})")
        
        # Layout information
        if self.global_layout_positions:
            print(f"\n🎯 LAYOUT INFORMATION:")
            print(f"   2D positions computed: {len(self.global_layout_positions)} nodes")
            print(f"   Layout algorithm: {self.reproducibility_config['layout_algorithm']}")
            print(f"   Random seed: {self.reproducibility_config['random_seed']}")
            
        # Filtering configuration summary
        if hasattr(self, 'graph_construction_config'):
            print(f"\n🔧 FILTERING CONFIGURATION:")
            config = self.graph_construction_config
            print(f"   Sliding window size: {config.get('sliding_window_size', 'N/A')}")
            print(f"   Min co-occurrence threshold: {config.get('min_cooccurrence_threshold', 'N/A')}")
            print(f"   Min edge weight: {config.get('min_edge_weight', 'N/A')}")
            print(f"   Edge density reduction: {config.get('edge_density_reduction', 'N/A')}")
            print(f"   Core node percentile: {config.get('core_node_percentile', 'N/A')}")
            print(f"   Positions stored as node attributes: ✅")
            print(f"   Community-aware layout: ✅")
    
    def export_global_graph_data(self):
        """Export global graph data from NetworkX object (secondary to graph object)"""
        if not self.validate_pipeline_step('global_graph_built', "Please build global graph first (step 4.1)"):
            return
        
        print("\n💾 EXPORT GLOBAL GRAPH DATA")
        print("-" * 50)
        print("📋 Exporting from NetworkX graph object (graph object remains primary)")
        
        try:
            # Create global graph directory
            graph_dir = os.path.join(self.output_dir, "global_graph")
            os.makedirs(graph_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.global_graph_object is None:
                print("⚠️ No NetworkX graph object available")
                return
            
            G = self.global_graph_object
            
            # Export NetworkX graph in multiple formats
            base_name = f"global_graph_{timestamp}"
            
            # 1. Export as GraphML (preserves all attributes) - Convert numpy arrays to lists
            graphml_file = os.path.join(graph_dir, f"{base_name}.graphml")
            
            # Create a copy of the graph with numpy arrays converted to separate x,y attributes for GraphML compatibility
            G_copy = G.copy()
            for node in G_copy.nodes():
                if 'pos' in G_copy.nodes[node]:
                    pos = G_copy.nodes[node]['pos']
                    if isinstance(pos, np.ndarray):
                        G_copy.nodes[node]['pos_x'] = float(pos[0])
                        G_copy.nodes[node]['pos_y'] = float(pos[1])
                        del G_copy.nodes[node]['pos']  # Remove the array/list attribute
                    elif isinstance(pos, (list, tuple)) and len(pos) == 2:
                        G_copy.nodes[node]['pos_x'] = float(pos[0])
                        G_copy.nodes[node]['pos_y'] = float(pos[1])
                        del G_copy.nodes[node]['pos']  # Remove the array/list attribute
            
            nx.write_graphml(G_copy, graphml_file)
            print(f"✅ GraphML exported: {graphml_file}")
            
            # 2. Export as JSON with full structure
            json_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                    'density': nx.density(G) if G.number_of_nodes() > 1 else 0,
                    'construction_params': self.reproducibility_config.copy()
                },
                'nodes': [
                    {
                        'id': node,
                        'frequency': G.nodes[node].get('frequency', 0),
                        'phrase_type': G.nodes[node].get('phrase_type', 'unknown'),
                        'community': G.nodes[node].get('community', 0),
                        'degree_centrality': G.nodes[node].get('degree_centrality', 0),
                        'betweenness_centrality': G.nodes[node].get('betweenness_centrality', 0),
                        'position': G.nodes[node].get('pos', [0, 0]).tolist() if isinstance(G.nodes[node].get('pos', [0, 0]), np.ndarray) else list(G.nodes[node].get('pos', [0, 0]))
                    }
                    for node in G.nodes()
                ],
                'edges': [
                    {
                        'source': u,
                        'target': v,
                        'weight': data['weight']
                    }
                    for u, v, data in G.edges(data=True)
                ]
            }
            
            json_file = os.path.join(graph_dir, f"{base_name}.json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"✅ JSON exported: {json_file}")
            
            # 3. Export edge list for external tools
            edge_list_file = os.path.join(graph_dir, f"{base_name}_edges.csv")
            with open(edge_list_file, 'w', encoding='utf-8') as f:
                f.write("source,target,weight\n")
                for u, v, data in G.edges(data=True):
                    f.write(f'"{u}","{v}",{data["weight"]}\n')
            print(f"✅ Edge list exported: {edge_list_file}")
            
            # 4. Export node attributes
            node_attrs_file = os.path.join(graph_dir, f"{base_name}_nodes.csv")
            with open(node_attrs_file, 'w', encoding='utf-8') as f:
                f.write("node,frequency,phrase_type,community,degree_centrality,betweenness_centrality,pos_x,pos_y\n")
                for node in G.nodes():
                    attrs = G.nodes[node]
                    pos = attrs.get('pos', [0, 0])
                    f.write(f'"{node}",{attrs.get("frequency", 0)},"{attrs.get("phrase_type", "unknown")}",'
                           f'{attrs.get("community", 0)},{attrs.get("degree_centrality", 0)},'
                           f'{attrs.get("betweenness_centrality", 0)},{pos[0]},{pos[1]}\n')
            print(f"✅ Node attributes exported: {node_attrs_file}")
            
            print("📋 All exports are secondary representations of the primary NetworkX graph object")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            import traceback
            traceback.print_exc()
    
    def activate_state_subgraphs(self):
        """Activate state-based subgraphs as NetworkX subgraph views (NOT rebuilds)"""
        if not self.validate_pipeline_step('global_graph_built', "Please build global graph first (step 4.1)"):
            return
        
        print("\n🗺️ STATE-BASED SUBGRAPH ACTIVATION")
        print("-" * 60)
        print("🔬 Creating NetworkX subgraph views from global graph (NOT rebuilding)")
        print("🌐 Subgraphs share the same node space and positions as the global graph")
        print("⚖️ Using re-weighting approach to preserve global structure")
        
        try:
            print("⏳ Activating state-based NetworkX subgraphs...")
            
            # Group documents by state
            state_documents = {}
            for doc in self.cleaned_text_data:
                state = doc['state']
                if state not in state_documents:
                    state_documents[state] = []
                state_documents[state].append(doc)
            
            self.state_subgraph_objects = {}
            
            # For each state, create a subgraph view from the global NetworkX graph
            for state, docs in tqdm(state_documents.items(), desc="🗺️ Activating subgraphs", unit="state"):
                print(f"   🗺️ Processing state: {state} ({len(docs)} documents)")
                
                # B. FIXED SUBGRAPH ACTIVATION LOGIC:
                # 1. Use global graph as base (NOT rebuild from scratch)
                # 2. Select edges whose co-occurrence windows belong to target state
                # 3. Re-weight edges instead of re-creating nodes
                # 4. Preserve global node positions
                # 5. Allow isolated nodes to remain
                
                # Get phrases that appear in this state's documents
                state_phrases = set()
                state_cooccurrences = defaultdict(int)
                
                # Calculate state-specific co-occurrences for re-weighting
                for doc in docs:
                    tokens = doc['tokens']
                    doc_phrases = []
                    
                    # Extract phrases from this state's documents
                    if self.reproducibility_config['phrase_type'] in ['word', 'mixed']:
                        doc_phrases.extend([token for token in tokens if token in self.phrase_data['filtered_phrases']])
                    
                    if self.reproducibility_config['phrase_type'] in ['bigram', 'mixed']:
                        for i in range(len(tokens) - 1):
                            bigram = f"{tokens[i]} {tokens[i+1]}"
                            if bigram in self.phrase_data['filtered_phrases']:
                                doc_phrases.append(bigram)
                    
                    # Add to state phrases
                    state_phrases.update(doc_phrases)
                    
                    # Calculate state-specific co-occurrences using sliding window
                    window_size = self.graph_construction_config.get('sliding_window_size', 5)
                    
                    for i in range(len(doc_phrases)):
                        window_start = max(0, i - window_size // 2)
                        window_end = min(len(doc_phrases), i + window_size // 2 + 1)
                        
                        phrase1 = doc_phrases[i]
                        
                        for j in range(window_start, window_end):
                            if i != j:
                                phrase2 = doc_phrases[j]
                                if phrase1 != phrase2:
                                    edge = tuple(sorted([phrase1, phrase2]))
                                    state_cooccurrences[edge] += 1
                
                # Create NetworkX subgraph view with re-weighted edges
                # Start with all nodes that appear in this state (including isolated ones)
                state_nodes = [node for node in self.global_graph_object.nodes() if node in state_phrases]
                
                if state_nodes:
                    # Create induced subgraph (preserves structure from global graph)
                    base_subgraph = self.global_graph_object.subgraph(state_nodes)
                    
                    # Create a copy to allow edge re-weighting
                    state_subgraph = base_subgraph.copy()
                    
                    # Re-weight edges based on state-specific co-occurrences
                    edges_to_remove = []
                    for u, v, data in state_subgraph.edges(data=True):
                        edge_key = tuple(sorted([u, v]))
                        state_weight = state_cooccurrences.get(edge_key, 0)
                        
                        if state_weight > 0:
                            # Re-weight edge based on state-specific co-occurrence
                            state_subgraph[u][v]['weight'] = state_weight
                            state_subgraph[u][v]['state_weight'] = state_weight
                            state_subgraph[u][v]['global_weight'] = data.get('weight', 0)
                        else:
                            # Mark edge for removal if no state-specific co-occurrence
                            edges_to_remove.append((u, v))
                    
                    # Remove edges with no state-specific support
                    state_subgraph.remove_edges_from(edges_to_remove)
                    
                    # Preserve global node positions (for visualization consistency)
                    global_positions = nx.get_node_attributes(self.global_graph_object, 'pos')
                    nx.set_node_attributes(state_subgraph, global_positions, 'pos')
                    
                    # Copy other node attributes from global graph
                    for attr in ['importance', 'role', 'community', 'tf_idf_score', 'frequency']:
                        global_attr = nx.get_node_attributes(self.global_graph_object, attr)
                        state_attr = {node: global_attr.get(node, 0) for node in state_subgraph.nodes()}
                        nx.set_node_attributes(state_subgraph, state_attr, attr)
                    
                    # Store as NetworkX subgraph object
                    self.state_subgraph_objects[state] = state_subgraph
                    
                    # Count isolated nodes (explicitly allowed to remain)
                    isolated_count = len(list(nx.isolates(state_subgraph)))
                    
                    print(f"      ✅ {state}: {state_subgraph.number_of_nodes()} nodes, {state_subgraph.number_of_edges()} edges")
                    if isolated_count > 0:
                        print(f"         Isolated nodes: {isolated_count} (preserved)")
                    
                    # Create legacy data structure for backward compatibility
                    state_edges = {}
                    for phrase1, phrase2, data in state_subgraph.edges(data=True):
                        pair_key = f"{sorted([phrase1, phrase2])[0]}|||{sorted([phrase1, phrase2])[1]}"
                        state_edges[pair_key] = data['weight']
                    
                    self.state_subgraphs[state] = {
                        'state': state,
                        'nodes': list(state_nodes),
                        'edges': state_edges,
                        'node_count': state_subgraph.number_of_nodes(),
                        'edge_count': state_subgraph.number_of_edges(),
                        'isolated_nodes': isolated_count,
                        'document_count': len(docs),
                        'activation_method': 'reweight',
                        'activation_timestamp': datetime.now().isoformat(),
                        'source_global_graph': True
                    }
                else:
                    print(f"      ⚠️ {state}: No valid phrases found")
            
            print(f"\n✅ State subgraph activation completed!")
            print(f"🗺️ Activated {len(self.state_subgraph_objects)} NetworkX state subgraphs")
            print(f"🎯 All subgraphs share positions from global layout")
            print(f"⚖️ Edges re-weighted based on state-specific co-occurrences")
            print(f"🔗 Isolated nodes preserved in subgraphs")
            
            self.pipeline_state['subgraphs_activated'] = True
            
        except Exception as e:
            print(f"❌ Subgraph activation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def view_subgraph_comparisons(self):
        """View subgraph comparison statistics computed from NetworkX objects"""
        if not self.validate_pipeline_step('subgraphs_activated', "Please activate subgraphs first (step 5.1)"):
            return
        
        print("\n📊 SUBGRAPH COMPARISON ANALYSIS")
        print("-" * 60)
        
        if not hasattr(self, 'state_subgraph_objects') or not self.state_subgraph_objects:
            print("⚠️ No state subgraph objects available")
            return
        
        print(f"🗺️ SUBGRAPH OVERVIEW (NetworkX objects):")
        print(f"   Total states: {len(self.state_subgraph_objects)}")
        print(f"   Source: Global graph subviews (shared node space and positions)")
        print(f"   Activation method: Re-weighting (preserves global structure)")
        
        # Enhanced comparison table with new metrics
        print(f"\n📋 STATE COMPARISON TABLE:")
        print(f"{'State':<12} {'Docs':<6} {'Nodes':<8} {'Edges':<8} {'Isolated':<10} {'Density':<10} {'LCC Size':<10} {'Density Δ':<12}")
        print("-" * 100)
        
        global_density = nx.density(self.global_graph_object) if self.global_graph_object.number_of_nodes() > 1 else 0
        
        for state, subgraph in self.state_subgraph_objects.items():
            doc_count = len([doc for doc in self.cleaned_text_data if doc['state'] == state])
            node_count = subgraph.number_of_nodes()
            edge_count = subgraph.number_of_edges()
            
            # Isolated node count
            isolated_count = len(list(nx.isolates(subgraph)))
            
            # Density calculation
            if node_count > 1:
                density = nx.density(subgraph) * 100
                density_diff = (nx.density(subgraph) - global_density) * 100
            else:
                density = 0.0
                density_diff = 0.0
            
            # Largest connected component size
            if node_count > 0:
                components = list(nx.connected_components(subgraph))
                lcc_size = len(max(components, key=len)) if components else 0
            else:
                lcc_size = 0
            
            print(f"{state:<12} {doc_count:<6} {node_count:<8} {edge_count:<8} {isolated_count:<10} {density:<10.2f} {lcc_size:<10} {density_diff:<12.2f}")
        
        # Global vs subgraph statistics comparison
        print(f"\n📊 GLOBAL VS SUBGRAPH STATISTICS:")
        print(f"   Global graph density: {global_density * 100:.2f}%")
        
        # Calculate average subgraph metrics
        subgraph_densities = []
        subgraph_isolated_counts = []
        subgraph_component_counts = []
        
        for subgraph in self.state_subgraph_objects.values():
            if subgraph.number_of_nodes() > 1:
                subgraph_densities.append(nx.density(subgraph))
            
            isolated_count = len(list(nx.isolates(subgraph)))
            subgraph_isolated_counts.append(isolated_count)
            
            components = list(nx.connected_components(subgraph))
            subgraph_component_counts.append(len(components))
        
        if subgraph_densities:
            avg_subgraph_density = np.mean(subgraph_densities) * 100
            print(f"   Average subgraph density: {avg_subgraph_density:.2f}%")
            print(f"   Density difference: {avg_subgraph_density - global_density * 100:.2f} percentage points")
        
        if subgraph_isolated_counts:
            total_isolated = sum(subgraph_isolated_counts)
            avg_isolated = np.mean(subgraph_isolated_counts)
            print(f"   Total isolated nodes across subgraphs: {total_isolated}")
            print(f"   Average isolated nodes per subgraph: {avg_isolated:.1f}")
        
        if subgraph_component_counts:
            avg_components = np.mean(subgraph_component_counts)
            print(f"   Average connected components per subgraph: {avg_components:.1f}")
        
        # Top states by different metrics
        print(f"\n🏆 TOP STATES BY METRICS:")
        
        # By node count
        states_by_nodes = sorted(self.state_subgraph_objects.items(), 
                               key=lambda x: x[1].number_of_nodes(), reverse=True)[:5]
        print(f"   Largest by nodes: {[(state, sg.number_of_nodes()) for state, sg in states_by_nodes]}")
        
        # By edge count
        states_by_edges = sorted(self.state_subgraph_objects.items(), 
                               key=lambda x: x[1].number_of_edges(), reverse=True)[:5]
        print(f"   Largest by edges: {[(state, sg.number_of_edges()) for state, sg in states_by_edges]}")
        
        # By density (for states with >1 node)
        states_by_density = [(state, nx.density(sg)) for state, sg in self.state_subgraph_objects.items() 
                           if sg.number_of_nodes() > 1]
        states_by_density.sort(key=lambda x: x[1], reverse=True)
        print(f"   Highest density: {[(state, f'{density*100:.1f}%') for state, density in states_by_density[:5]]}")
        
        # Community preservation analysis
        print(f"\n🏘️ COMMUNITY PRESERVATION ANALYSIS:")
        global_communities = nx.get_node_attributes(self.global_graph_object, 'community')
        
        if global_communities:
            for state, subgraph in list(self.state_subgraph_objects.items())[:5]:  # Show first 5 states
                state_communities = set()
                for node in subgraph.nodes():
                    if node in global_communities:
                        state_communities.add(global_communities[node])
                
                print(f"   {state}: {len(state_communities)} communities preserved from global graph")
        
        # Edge re-weighting analysis
        print(f"\n⚖️ EDGE RE-WEIGHTING ANALYSIS:")
        total_reweighted_edges = 0
        total_preserved_edges = 0
        
        for state, subgraph in self.state_subgraph_objects.items():
            reweighted = 0
            preserved = 0
            
            for u, v, data in subgraph.edges(data=True):
                if 'state_weight' in data and 'global_weight' in data:
                    if data['state_weight'] != data['global_weight']:
                        reweighted += 1
                    else:
                        preserved += 1
            
            total_reweighted_edges += reweighted
            total_preserved_edges += preserved
        
        total_edges = total_reweighted_edges + total_preserved_edges
        if total_edges > 0:
            reweight_pct = (total_reweighted_edges / total_edges) * 100
            print(f"   Total edges across subgraphs: {total_edges}")
            print(f"   Re-weighted edges: {total_reweighted_edges} ({reweight_pct:.1f}%)")
            print(f"   Preserved edges: {total_preserved_edges} ({100-reweight_pct:.1f}%)")
    
    def export_subgraph_data(self):
        """Export subgraph data from NetworkX objects (secondary to graph objects)"""
        if not self.validate_pipeline_step('subgraphs_activated', "Please activate subgraphs first (step 5.1)"):
            return
        
        print("\n💾 EXPORT SUBGRAPH DATA")
        print("-" * 50)
        print("📋 Exporting from NetworkX subgraph objects (graph objects remain primary)")
        
        try:
            # Create subgraphs directory
            subgraph_dir = os.path.join(self.output_dir, "subgraphs")
            os.makedirs(subgraph_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export each subgraph from NetworkX objects
            subgraph_items = list(self.state_subgraph_objects.items())
            
            for state, subgraph in tqdm(subgraph_items, desc="💾 Exporting subgraphs", unit="subgraph"):
                print(f"   📊 Exporting {state} subgraph...")
                
                # Progress for individual subgraph export steps
                with tqdm(total=4, desc=f"💾 {state} export steps", unit="step", leave=False) as step_pbar:
                    
                    # Step 1: Prepare GraphML export
                    step_pbar.set_description(f"💾 {state}: Preparing GraphML")
                    graphml_file = os.path.join(subgraph_dir, f"subgraph_{state}_{timestamp}.graphml")
                    
                    # Create a copy of the subgraph with numpy arrays converted to separate x,y attributes
                    subgraph_copy = subgraph.copy()
                    for node in subgraph_copy.nodes():
                        # Get position from global graph and convert if needed
                        if node in self.global_graph_object.nodes:
                            pos = self.global_graph_object.nodes[node].get('pos')
                            if pos is not None:
                                if isinstance(pos, np.ndarray):
                                    subgraph_copy.nodes[node]['pos_x'] = float(pos[0])
                                    subgraph_copy.nodes[node]['pos_y'] = float(pos[1])
                                    if 'pos' in subgraph_copy.nodes[node]:
                                        del subgraph_copy.nodes[node]['pos']  # Remove the array/list attribute
                                elif isinstance(pos, (list, tuple)) and len(pos) == 2:
                                    subgraph_copy.nodes[node]['pos_x'] = float(pos[0])
                                    subgraph_copy.nodes[node]['pos_y'] = float(pos[1])
                                    if 'pos' in subgraph_copy.nodes[node]:
                                        del subgraph_copy.nodes[node]['pos']  # Remove the array/list attribute
                    step_pbar.update(1)
                    
                    # Step 2: Write GraphML file
                    step_pbar.set_description(f"💾 {state}: Writing GraphML")
                    nx.write_graphml(subgraph_copy, graphml_file)
                    step_pbar.update(1)
                    
                    # Step 3: Prepare JSON export
                    step_pbar.set_description(f"💾 {state}: Preparing JSON")
                    json_data = {
                        'metadata': {
                            'state': state,
                            'timestamp': datetime.now().isoformat(),
                            'nodes': subgraph.number_of_nodes(),
                            'edges': subgraph.number_of_edges(),
                            'density': nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0,
                            'document_count': len([doc for doc in self.cleaned_text_data if doc['state'] == state]),
                            'source_global_graph': True
                        },
                        'nodes': [
                            {
                                'id': node,
                                'frequency': self.global_graph_object.nodes[node].get('frequency', 0),
                                'phrase_type': self.global_graph_object.nodes[node].get('phrase_type', 'unknown'),
                                'community': self.global_graph_object.nodes[node].get('community', 0),
                                'degree_centrality': self.global_graph_object.nodes[node].get('degree_centrality', 0),
                                'position': self.global_graph_object.nodes[node].get('pos', [0, 0]).tolist() if isinstance(self.global_graph_object.nodes[node].get('pos', [0, 0]), np.ndarray) else list(self.global_graph_object.nodes[node].get('pos', [0, 0]))
                            }
                            for node in subgraph.nodes()
                        ],
                        'edges': [
                            {
                                'source': u,
                                'target': v,
                                'weight': data['weight']
                            }
                            for u, v, data in subgraph.edges(data=True)
                        ]
                    }
                    step_pbar.update(1)
                    
                    # Step 4: Write JSON file
                    step_pbar.set_description(f"💾 {state}: Writing JSON")
                    json_file = os.path.join(subgraph_dir, f"subgraph_{state}_{timestamp}.json")
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    step_pbar.update(1)
                
                print(f"      ✅ GraphML: subgraph_{state}_{timestamp}.graphml")
                print(f"      ✅ JSON: subgraph_{state}_{timestamp}.json")
            
            # Export combined summary
            summary_data = {
                'export_timestamp': datetime.now().isoformat(),
                'total_states': len(self.state_subgraph_objects),
                'subgraphs': {
                    state: {
                        'nodes': subgraph.number_of_nodes(),
                        'edges': subgraph.number_of_edges(),
                        'density': nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0
                    }
                    for state, subgraph in self.state_subgraph_objects.items()
                }
            }
            
            summary_file = os.path.join(subgraph_dir, f"subgraph_summary_{timestamp}.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Summary exported: subgraph_summary_{timestamp}.json")
            print("📋 All exports are secondary representations of the primary NetworkX subgraph objects")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_scientific_visualizations(self):
        """Generate publication-quality scientific visualizations"""
        if not self.validate_pipeline_step('subgraphs_activated', "Please activate subgraphs first (step 5.1)"):
            return
        
        print("\n🎨 SCIENTIFIC VISUALIZATION GENERATION")
        print("-" * 60)
        print("🔬 Generating publication-quality network visualizations")
        print(f"🌱 Random seed: {self.reproducibility_config['random_seed']}")
        print(f"🎯 Layout algorithm: {self.reproducibility_config['layout_algorithm']}")
        
        try:
            # Create visualizations directory
            viz_dir = os.path.join(self.output_dir, "scientific_visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed = self.reproducibility_config['random_seed']
            
            self.visualization_paths = {}
            
            # Use optimized graph if available, otherwise fall back to original
            graph_to_visualize = self.optimized_global_graph if self.optimized_global_graph else self.global_graph_object
            communities = self.global_communities if self.global_communities else {}
            node_roles = self.global_node_roles if self.global_node_roles else {}
            positions = self.global_layout_positions if self.global_layout_positions else {}
            
            if not communities:
                # Fallback community detection
                print("🏘️ Applying fallback community detection...")
                try:
                    import community as community_louvain
                    communities = community_louvain.best_partition(graph_to_visualize, weight='weight', random_state=seed)
                except:
                    communities = {node: 0 for node in graph_to_visualize.nodes()}
            
            if not node_roles:
                # Fallback core-periphery identification
                print("🎯 Applying fallback core-periphery identification...")
                degrees = dict(graph_to_visualize.degree())
                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                n_core = min(50, len(sorted_nodes) // 5)  # Top 20% or 50 nodes
                core_nodes = set(node for node, _ in sorted_nodes[:n_core])
                node_roles = {node: 'core' if node in core_nodes else 'periphery' for node in graph_to_visualize.nodes()}
            
            if not positions:
                # Fallback layout computation
                print("🎯 Computing fallback layout...")
                positions = nx.spring_layout(graph_to_visualize, seed=seed, k=1.0/np.sqrt(graph_to_visualize.number_of_nodes()))
            
            # 1. GLOBAL SCIENTIFIC VISUALIZATION
            if graph_to_visualize and graph_to_visualize.number_of_nodes() > 0:
                global_viz_name = f"scientific_global_network_seed{seed}_{timestamp}.png"
                global_viz_path = os.path.join(viz_dir, global_viz_name)
                
                if self.scientific_optimizer:
                    self.scientific_optimizer.generate_scientific_visualization(
                        graph_to_visualize, communities, node_roles, positions,
                        global_viz_path, "Scientific Global Co-occurrence Network"
                    )
                else:
                    # Fallback visualization
                    self._generate_fallback_visualization(
                        graph_to_visualize, communities, node_roles, positions,
                        global_viz_path, "Global Co-occurrence Network"
                    )
                
                self.visualization_paths['scientific_global'] = global_viz_path
                print(f"      ✅ Global visualization: {global_viz_name}")
            
            # 2. STATE SUBGRAPH SCIENTIFIC VISUALIZATIONS
            if hasattr(self, 'state_subgraph_objects') and self.state_subgraph_objects:
                subgraph_items = list(self.state_subgraph_objects.items())
                
                for state, subgraph in tqdm(subgraph_items, desc="🎨 Scientific state networks", unit="subgraph"):
                    if subgraph.number_of_nodes() > 0:
                        # Use positions from global graph for consistency
                        subgraph_positions = {node: positions[node] for node in subgraph.nodes() 
                                            if node in positions}
                        
                        # Get communities and roles for subgraph nodes
                        subgraph_communities = {node: communities.get(node, 0) for node in subgraph.nodes()}
                        subgraph_roles = {node: node_roles.get(node, 'periphery') for node in subgraph.nodes()}
                        
                        state_viz_name = f"scientific_state_{state}_network_seed{seed}_{timestamp}.png"
                        state_viz_path = os.path.join(viz_dir, state_viz_name)
                        
                        if self.scientific_optimizer:
                            self.scientific_optimizer.generate_scientific_visualization(
                                subgraph, subgraph_communities, subgraph_roles, subgraph_positions,
                                state_viz_path, f"Scientific State {state} Network"
                            )
                        else:
                            # Fallback visualization
                            self._generate_fallback_visualization(
                                subgraph, subgraph_communities, subgraph_roles, subgraph_positions,
                                state_viz_path, f"State {state} Network"
                            )
                        
                        self.visualization_paths[f'scientific_subgraph_{state}'] = state_viz_path
                        print(f"      ✅ State {state} visualization: {state_viz_name}")
            
            print(f"\n✅ Scientific visualization generation completed!")
            print(f"🎨 Generated {len(self.visualization_paths)} publication-quality visualizations")
            print(f"📁 Output directory: {viz_dir}")
            print(f"🔬 Scientific methods applied: semantic weighting, sparsification, community pruning")
            print(f"🎯 Deterministic layouts with fixed seed: {seed}")
            
            self.pipeline_state['results_exported'] = True
            
        except Exception as e:
            print(f"❌ Scientific visualization generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_fallback_visualization(self, graph, communities, node_roles, positions, output_path, title):
        """Fallback visualization method when scientific optimizer is not available"""
        plt.figure(figsize=(16, 12))
        
        # Simple visualization
        node_colors = [communities.get(node, 0) for node in graph.nodes()]
        node_sizes = [300 if node_roles.get(node, 'periphery') == 'core' else 100 for node in graph.nodes()]
        
        nx.draw(graph, positions, 
                node_color=node_colors, 
                node_size=node_sizes,
                with_labels=False,
                edge_color='lightgray',
                alpha=0.7)
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        """Generate deterministic visualizations directly from NetworkX graph objects"""
        if not self.validate_pipeline_step('subgraphs_activated', "Please activate subgraphs first (step 5.1)"):
            return
        
        print("\n🎨 DETERMINISTIC VISUALIZATION GENERATION")
        print("-" * 60)
        print("🔬 Generating readable thematic network visualizations")
        print(f"🌱 Random seed: {self.reproducibility_config['random_seed']}")
        print(f"🎯 Layout algorithm: {self.reproducibility_config['layout_algorithm']}")
        
        # C. FIXED VISUALIZATION CONFIGURATION for semantic reference style
        self.viz_config = {
            # Deterministic layout
            'fixed_random_seed': self.reproducibility_config['random_seed'],
            'cache_positions': True,
            
            # Visual encoding
            'edge_alpha_light': 0.3,  # Intra-community edges
            'edge_alpha_inter': 0.05,  # Inter-community edges  
            'edge_color': 'lightgray',
            'edge_weight_threshold': 2,  # Hide edges below this weight
            
            # Node shapes by role
            'core_node_shape': '^',  # Triangle for core nodes
            'periphery_node_shape': 'o',  # Circle for periphery nodes
            
            # Node size scaling by semantic importance (TF-IDF, not raw frequency)
            'min_node_size': 50,
            'max_node_size': 800,
            'size_by_tfidf': True,
            
            # Selective labeling
            'label_core_only': True,
            'label_importance_threshold': 0.7,  # Top 30% important nodes
            'max_labels_per_community': 3,
            'never_label_structural': True,
            
            # High-resolution output
            'output_dpi': 300,
            'figure_size': (16, 12),
            'export_format': 'PNG',
        }
        
        try:
            # Create visualizations directory
            viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed = self.reproducibility_config['random_seed']
            
            self.visualization_paths = {}
            
            print("⏳ Generating readable thematic network visualizations...")
            
            # Set matplotlib parameters for consistent, high-resolution output
            plt.rcParams['figure.dpi'] = self.viz_config['output_dpi']
            plt.rcParams['savefig.dpi'] = self.viz_config['output_dpi']
            plt.rcParams['font.size'] = 10
            plt.rcParams['font.family'] = 'sans-serif'
            
            # 1. GLOBAL GRAPH VISUALIZATION - SEMANTIC THEMATIC NETWORK
            if self.global_graph_object and self.global_layout_positions:
                with tqdm(total=8, desc="🌐 Global thematic network", unit="step") as pbar:
                    pbar.set_description("🌐 Setting up figure")
                    fig, ax = plt.subplots(1, 1, figsize=self.viz_config['figure_size'])
                    G = self.global_graph_object
                    
                    # Use cached positions from global graph (deterministic)
                    pos = self.global_layout_positions
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Preparing node attributes")
                    # Get node attributes for visual encoding
                    communities = nx.get_node_attributes(G, 'community')
                    importance_scores = nx.get_node_attributes(G, 'importance')
                    node_roles = nx.get_node_attributes(G, 'role')
                    tf_idf_scores = nx.get_node_attributes(G, 'tf_idf_score')
                    is_structural = nx.get_node_attributes(G, 'is_structural')
                    
                    # Create distinct color map for communities
                    unique_communities = sorted(set(communities.values())) if communities else [0]
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
                    community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Computing visual attributes")
                    # Node visual attributes based on semantic importance
                    node_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in G.nodes()]
                    
                    # Node sizes based on TF-IDF scores (semantic importance), NOT raw frequency
                    node_sizes = []
                    node_shapes_core = []
                    node_shapes_periphery = []
                    
                    for node in G.nodes():
                        # Use TF-IDF for size scaling if available, fallback to importance
                        if self.viz_config['size_by_tfidf'] and tf_idf_scores:
                            semantic_score = tf_idf_scores.get(node, 0)
                            max_score = max(tf_idf_scores.values()) if tf_idf_scores.values() else 1
                            normalized_score = semantic_score / max_score if max_score > 0 else 0
                        else:
                            normalized_score = importance_scores.get(node, 0)
                        
                        size = self.viz_config['min_node_size'] + (self.viz_config['max_node_size'] - self.viz_config['min_node_size']) * normalized_score
                        node_sizes.append(size)
                        
                        # Separate nodes by role for different shapes
                        role = node_roles.get(node, 'periphery')
                        if role == 'core':
                            node_shapes_core.append(node)
                        else:
                            node_shapes_periphery.append(node)
                    
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Drawing edges with community-aware filtering")
                    # Edge rendering with community-aware alpha and weight threshold
                    edges_to_draw = []
                    edge_colors = []
                    edge_alphas = []
                    
                    for u, v, data in G.edges(data=True):
                        weight = data['weight']
                        
                        # Apply weight threshold to avoid hairball effect
                        if weight < self.viz_config['edge_weight_threshold']:
                            continue
                        
                        u_community = communities.get(u, 0)
                        v_community = communities.get(v, 0)
                        
                        # Community-aware edge rendering
                        if u_community == v_community:
                            # Intra-community edges: higher alpha
                            alpha = self.viz_config['edge_alpha_light']
                        else:
                            # Inter-community edges: lower alpha
                            alpha = self.viz_config['edge_alpha_inter']
                        
                        edges_to_draw.append((u, v))
                        edge_colors.append(self.viz_config['edge_color'])
                        edge_alphas.append(alpha)
                    
                    # Draw edges in batches to avoid performance issues
                    if edges_to_draw:
                        # Separate intra and inter community edges for different rendering
                        intra_edges = []
                        inter_edges = []
                        
                        for i, (u, v) in enumerate(edges_to_draw):
                            u_community = communities.get(u, 0)
                            v_community = communities.get(v, 0)
                            
                            if u_community == v_community:
                                intra_edges.append((u, v))
                            else:
                                inter_edges.append((u, v))
                        
                        # Draw inter-community edges first (lower layer)
                        if inter_edges:
                            nx.draw_networkx_edges(G, pos, edgelist=inter_edges,
                                                 width=0.5, alpha=self.viz_config['edge_alpha_inter'], 
                                                 edge_color=self.viz_config['edge_color'], ax=ax)
                        
                        # Draw intra-community edges on top
                        if intra_edges:
                            nx.draw_networkx_edges(G, pos, edgelist=intra_edges,
                                                 width=1.0, alpha=self.viz_config['edge_alpha_light'], 
                                                 edge_color=self.viz_config['edge_color'], ax=ax)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Drawing nodes by role")
                    # Draw nodes by role with different shapes
                    if node_shapes_core:
                        core_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_core]
                        core_sizes = [node_sizes[list(G.nodes()).index(node)] for node in node_shapes_core]
                        nx.draw_networkx_nodes(G, pos, nodelist=node_shapes_core,
                                             node_color=core_colors, node_size=core_sizes,
                                             node_shape=self.viz_config['core_node_shape'],
                                             alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax)
                    
                    if node_shapes_periphery:
                        periphery_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_periphery]
                        periphery_sizes = [node_sizes[list(G.nodes()).index(node)] for node in node_shapes_periphery]
                        nx.draw_networkx_nodes(G, pos, nodelist=node_shapes_periphery,
                                             node_color=periphery_colors, node_size=periphery_sizes,
                                             node_shape=self.viz_config['periphery_node_shape'],
                                             alpha=0.8, edgecolors='gray', linewidths=0.5, ax=ax)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Adding selective semantic labels")
                    # SELECTIVE LABELING - Only label core nodes, never structural tokens
                    labels_to_draw = {}
                    
                    if self.viz_config['label_core_only']:
                        # Only label core nodes
                        candidate_nodes = node_shapes_core
                    else:
                        # Label all nodes above importance threshold
                        importance_threshold = np.percentile(list(importance_scores.values()), 
                                                           self.viz_config['label_importance_threshold'] * 100)
                        candidate_nodes = [node for node in G.nodes() 
                                         if importance_scores.get(node, 0) >= importance_threshold]
                    
                    # Never label structural tokens
                    if self.viz_config['never_label_structural']:
                        candidate_nodes = [node for node in candidate_nodes 
                                         if not is_structural.get(node, False)]
                    
                    # Group nodes by community for balanced labeling
                    community_nodes = defaultdict(list)
                    for node in candidate_nodes:
                        community = communities.get(node, 0)
                        importance = importance_scores.get(node, 0)
                        community_nodes[community].append((node, importance))
                    
                    # Select top nodes per community (max 3 per community)
                    for community, nodes in community_nodes.items():
                        # Sort by importance and take top N
                        top_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)[:self.viz_config['max_labels_per_community']]
                        for node, _ in top_nodes:
                            # Truncate long labels for readability
                            label = node[:15] + "..." if len(node) > 15 else node
                            labels_to_draw[node] = label
                    
                    if labels_to_draw:
                        nx.draw_networkx_labels(G, pos, labels_to_draw, 
                                              font_size=9, font_weight='bold', 
                                              font_color='black', ax=ax)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Adding enhanced legends")
                    # Enhanced title with semantic filtering info
                    structural_removed = len(self.phrase_data.get('filtered_phrases', {})) - G.number_of_nodes()
                    ax.set_title(f'Global Semantic Co-occurrence Network\n'
                               f'{G.number_of_nodes()} nodes ({structural_removed} structural tokens removed), '
                               f'{G.number_of_edges()} edges, {len(unique_communities)} communities\n'
                               f'Seed: {seed} | Density: {nx.density(G)*100:.2f}% | TF-IDF weighted', 
                               fontsize=14, fontweight='bold', pad=20)
                    
                    # Enhanced community legend
                    legend_elements = []
                    for comm in sorted(unique_communities)[:8]:  # Show first 8 communities
                        color = community_colors[comm]
                        legend_elements.append(patches.Patch(color=color, label=f'Community {comm}'))
                    
                    if len(unique_communities) > 8:
                        legend_elements.append(patches.Patch(color='lightgray', label=f'... +{len(unique_communities)-8} more'))
                    
                    # Role and semantic legend
                    legend_elements.append(patches.Patch(color='white', label=''))  # Spacer
                    legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                                    markerfacecolor='gray', markersize=10, label='Core nodes (triangles)'))
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor='gray', markersize=8, label='Periphery nodes (circles)'))
                    legend_elements.append(patches.Patch(color='white', label=''))  # Spacer
                    legend_elements.append(patches.Patch(color='lightgray', label='Node size: TF-IDF score'))
                    legend_elements.append(patches.Patch(color='lightgray', label='Edge alpha: Community relationship'))
                    
                    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                            frameon=True, fancybox=True, shadow=True)
                    
                    ax.axis('off')
                    plt.tight_layout()
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Saving high-resolution visualization")
                    global_viz_name = f"global_thematic_network_seed{seed}_{timestamp}.png"
                    global_viz_path = os.path.join(viz_dir, global_viz_name)
                    
                    # Always export physical image file (PNG) with high resolution
                    plt.savefig(global_viz_path, bbox_inches='tight', facecolor='white', 
                              dpi=self.viz_config['output_dpi'], format=self.viz_config['export_format'])
                    plt.close()
                    
                    # Print absolute output image path after generation
                    print(f"      ✅ Saved: {os.path.basename(global_viz_path)}")
                    print(f"      📁 Full path: {os.path.abspath(global_viz_path)}")
                    
                    self.visualization_paths['global'] = os.path.abspath(global_viz_path)
                    
                    self.visualization_paths['global_graph'] = global_viz_path
                    pbar.update(1)
                
                print(f"      ✅ Saved: {global_viz_name}")
            
            # 2. STATE SUBGRAPH VISUALIZATIONS - HIGHLIGHTED SUBSETS
            subgraph_items = list(self.state_subgraph_objects.items())
            
            for state, subgraph in tqdm(subgraph_items, desc="🎨 Generating state thematic networks", unit="subgraph"):
                if subgraph.number_of_nodes() > 0:
                    with tqdm(total=8, desc=f"🎨 {state} thematic network", unit="step", leave=False) as step_pbar:
                        step_pbar.set_description(f"🎨 {state}: Setting up figure")
                        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
                        
                        # Use same positions as global graph for consistency
                        subgraph_pos = {node: self.global_layout_positions[node] for node in subgraph.nodes() 
                                      if node in self.global_layout_positions}
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Preparing attributes")
                        # Get node attributes from global graph (maintain consistency)
                        communities = {node: self.global_graph_object.nodes[node].get('community', 0) 
                                     for node in subgraph.nodes()}
                        importance_scores = {node: self.global_graph_object.nodes[node].get('importance', 0) 
                                           for node in subgraph.nodes()}
                        node_roles = {node: self.global_graph_object.nodes[node].get('role', 'periphery') 
                                    for node in subgraph.nodes()}
                        
                        # Use same color scheme as global graph
                        unique_communities = sorted(set(communities.values()))
                        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_communities)))
                        community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Computing visual attributes")
                        # Node visual attributes (consistent with global)
                        node_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in subgraph.nodes()]
                        
                        node_sizes = []
                        node_shapes_core = []
                        node_shapes_periphery = []
                        
                        for node in subgraph.nodes():
                            importance = importance_scores.get(node, 0)
                            size = self.viz_config['min_node_size'] + (self.viz_config['max_node_size'] - self.viz_config['min_node_size']) * importance
                            node_sizes.append(size)
                            
                            role = node_roles.get(node, 'periphery')
                            if role == 'core':
                                node_shapes_core.append(node)
                            else:
                                node_shapes_periphery.append(node)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Drawing edges")
                        # 简化边绘制避免卡住
                        if subgraph.number_of_edges() > 0:
                            # 限制边数并简化绘制
                            edge_list = list(subgraph.edges(data=True))[:30]  # 最多30条边
                            if edge_list:
                                nx.draw_networkx_edges(subgraph, subgraph_pos, 
                                                     edgelist=[(u, v) for u, v, _ in edge_list],
                                                     width=1.0, alpha=0.3, edge_color='gray', ax=ax)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Drawing nodes")
                        # Draw nodes by role
                        if node_shapes_core:
                            core_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_core]
                            core_sizes = [node_sizes[list(subgraph.nodes()).index(node)] for node in node_shapes_core]
                            nx.draw_networkx_nodes(subgraph, subgraph_pos, nodelist=node_shapes_core,
                                                 node_color=core_colors, node_size=core_sizes,
                                                 node_shape=self.viz_config['core_node_shape'],
                                                 alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax)
                        
                        if node_shapes_periphery:
                            periphery_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_periphery]
                            periphery_sizes = [node_sizes[list(subgraph.nodes()).index(node)] for node in node_shapes_periphery]
                            nx.draw_networkx_nodes(subgraph, subgraph_pos, nodelist=node_shapes_periphery,
                                                 node_color=periphery_colors, node_size=periphery_sizes,
                                                 node_shape=self.viz_config['periphery_node_shape'],
                                                 alpha=0.8, edgecolors='gray', linewidths=0.5, ax=ax)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Adding labels")
                        # Selective labeling for subgraph
                        labels_to_draw = {}
                        if importance_scores:
                            importance_threshold = np.percentile(list(importance_scores.values()), 70)
                            
                            community_nodes = defaultdict(list)
                            for node in subgraph.nodes():
                                community = communities.get(node, 0)
                                importance = importance_scores.get(node, 0)
                                if importance >= importance_threshold:
                                    community_nodes[community].append((node, importance))
                            
                            for community, nodes in community_nodes.items():
                                top_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)[:2]  # Fewer labels for subgraphs
                                for node, _ in top_nodes:
                                    labels_to_draw[node] = node
                        
                        if labels_to_draw:
                            nx.draw_networkx_labels(subgraph, subgraph_pos, labels_to_draw,
                                                  font_size=9, font_weight='bold',
                                                  font_color='black', ax=ax)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Finalizing")
                        doc_count = len([doc for doc in self.cleaned_text_data if doc['state'] == state])
                        core_count = len(node_shapes_core)
                        
                        ax.set_title(f'State {state} Thematic Network\n'
                                   f'{subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges, '
                                   f'{len(unique_communities)} communities\n'
                                   f'{doc_count} documents, {core_count} core nodes | Seed: {seed}', 
                                   fontsize=12, fontweight='bold', pad=15)
                        
                        # Add legend (simplified for subgraphs)
                        legend_elements = []
                        for comm in sorted(unique_communities):
                            color = community_colors[comm]
                            legend_elements.append(patches.Patch(color=color, label=f'Community {comm}'))
                        
                        if len(legend_elements) > 0:
                            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
                        
                        ax.axis('off')
                        plt.tight_layout()
                        
                        state_viz_name = f"state_{state}_thematic_network_seed{seed}_{timestamp}.png"
                        state_viz_path = os.path.join(viz_dir, state_viz_name)
                        plt.savefig(state_viz_path, bbox_inches='tight', facecolor='white', dpi=300)
                        plt.close()
                        
                        self.visualization_paths[f'subgraph_{state}'] = state_viz_path
                        step_pbar.update(1)
                    
                    print(f"      ✅ Saved: {state_viz_name}")
            
            print(f"\n✅ Thematic network visualization generation completed!")
            print(f"🎨 Generated {len(self.visualization_paths)} readable visualizations")
            print(f"📁 Output directory: {viz_dir}")
            print(f"🎯 All visualizations use consistent community-aware layout")
            print(f"🔍 Edge filtering applied for readability")
            print(f"🎭 Node roles visualized (core=triangles, periphery=circles)")
            
            self.pipeline_state['results_exported'] = True
            
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    
    def generate_deterministic_visualizations(self):
        """Generate deterministic visualizations directly from NetworkX graph objects"""
        if not self.validate_pipeline_step('subgraphs_activated', "Please activate subgraphs first (step 5.1)"):
            return
        
        print("\n🎨 DETERMINISTIC VISUALIZATION GENERATION")
        print("-" * 60)
        print("🔬 Generating readable thematic network visualizations")
        print(f"🌱 Random seed: {self.reproducibility_config['random_seed']}")
        print(f"🎯 Layout algorithm: {self.reproducibility_config['layout_algorithm']}")
        
        # C. FIXED VISUALIZATION CONFIGURATION for semantic reference style
        self.viz_config = {
            # Deterministic layout
            'fixed_random_seed': self.reproducibility_config['random_seed'],
            'cache_positions': True,
            
            # Visual encoding
            'edge_alpha_light': 0.3,  # Intra-community edges
            'edge_alpha_inter': 0.05,  # Inter-community edges  
            'edge_color': 'lightgray',
            'edge_weight_threshold': 2,  # Hide edges below this weight
            
            # Node shapes by role
            'core_node_shape': '^',  # Triangle for core nodes
            'periphery_node_shape': 'o',  # Circle for periphery nodes
            
            # Node size scaling by semantic importance (TF-IDF, not raw frequency)
            'min_node_size': 50,
            'max_node_size': 800,
            'size_by_tfidf': True,
            
            # Selective labeling
            'label_core_only': True,
            'label_importance_threshold': 0.7,  # Top 30% important nodes
            'max_labels_per_community': 3,
            'never_label_structural': True,
            
            # High-resolution output
            'output_dpi': 300,
            'figure_size': (16, 12),
            'export_format': 'PNG',
        }
        
        try:
            # Create visualizations directory
            viz_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            seed = self.reproducibility_config['random_seed']
            
            self.visualization_paths = {}
            
            print("⏳ Generating readable thematic network visualizations...")
            
            # Set matplotlib parameters for consistent, high-resolution output
            plt.rcParams['figure.dpi'] = self.viz_config['output_dpi']
            plt.rcParams['savefig.dpi'] = self.viz_config['output_dpi']
            plt.rcParams['font.size'] = 10
            plt.rcParams['font.family'] = 'sans-serif'
            
            # 1. GLOBAL GRAPH VISUALIZATION - SEMANTIC THEMATIC NETWORK
            if self.global_graph_object and self.global_layout_positions:
                with tqdm(total=8, desc="🌐 Global thematic network", unit="step") as pbar:
                    pbar.set_description("🌐 Setting up figure")
                    fig, ax = plt.subplots(1, 1, figsize=self.viz_config['figure_size'])
                    G = self.global_graph_object
                    
                    # Use cached positions from global graph (deterministic)
                    pos = self.global_layout_positions
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Preparing node attributes")
                    # Get node attributes for visual encoding
                    communities = nx.get_node_attributes(G, 'community')
                    importance_scores = nx.get_node_attributes(G, 'importance')
                    node_roles = nx.get_node_attributes(G, 'role')
                    tf_idf_scores = nx.get_node_attributes(G, 'tf_idf_score')
                    is_structural = nx.get_node_attributes(G, 'is_structural')
                    
                    # Create distinct color map for communities
                    unique_communities = sorted(set(communities.values())) if communities else [0]
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
                    community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Computing visual attributes")
                    # Node visual attributes based on semantic importance
                    node_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in G.nodes()]
                    
                    # Node sizes based on TF-IDF scores (semantic importance), NOT raw frequency
                    node_sizes = []
                    node_shapes_core = []
                    node_shapes_periphery = []
                    
                    for node in G.nodes():
                        # Use TF-IDF for size scaling if available, fallback to importance
                        if self.viz_config['size_by_tfidf'] and tf_idf_scores:
                            semantic_score = tf_idf_scores.get(node, 0)
                            max_score = max(tf_idf_scores.values()) if tf_idf_scores.values() else 1
                            normalized_score = semantic_score / max_score if max_score > 0 else 0
                        else:
                            normalized_score = importance_scores.get(node, 0)
                        
                        size = self.viz_config['min_node_size'] + (self.viz_config['max_node_size'] - self.viz_config['min_node_size']) * normalized_score
                        node_sizes.append(size)
                        
                        # Separate nodes by role for different shapes
                        role = node_roles.get(node, 'periphery')
                        if role == 'core':
                            node_shapes_core.append(node)
                        else:
                            node_shapes_periphery.append(node)
                    
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Drawing edges with community-aware filtering")
                    # Edge rendering with community-aware alpha and weight threshold
                    edges_to_draw = []
                    edge_colors = []
                    edge_alphas = []
                    
                    for u, v, data in G.edges(data=True):
                        weight = data['weight']
                        
                        # Apply weight threshold to avoid hairball effect
                        if weight < self.viz_config['edge_weight_threshold']:
                            continue
                        
                        u_community = communities.get(u, 0)
                        v_community = communities.get(v, 0)
                        
                        # Community-aware edge rendering
                        if u_community == v_community:
                            # Intra-community edges: higher alpha
                            alpha = self.viz_config['edge_alpha_light']
                        else:
                            # Inter-community edges: lower alpha
                            alpha = self.viz_config['edge_alpha_inter']
                        
                        edges_to_draw.append((u, v))
                        edge_colors.append(self.viz_config['edge_color'])
                        edge_alphas.append(alpha)
                    
                    # Draw edges in batches to avoid performance issues
                    if edges_to_draw:
                        # Separate intra and inter community edges for different rendering
                        intra_edges = []
                        inter_edges = []
                        
                        for i, (u, v) in enumerate(edges_to_draw):
                            u_community = communities.get(u, 0)
                            v_community = communities.get(v, 0)
                            
                            if u_community == v_community:
                                intra_edges.append((u, v))
                            else:
                                inter_edges.append((u, v))
                        
                        # Draw inter-community edges first (lower layer)
                        if inter_edges:
                            nx.draw_networkx_edges(G, pos, edgelist=inter_edges,
                                                 width=0.5, alpha=self.viz_config['edge_alpha_inter'], 
                                                 edge_color=self.viz_config['edge_color'], ax=ax)
                        
                        # Draw intra-community edges on top
                        if intra_edges:
                            nx.draw_networkx_edges(G, pos, edgelist=intra_edges,
                                                 width=1.0, alpha=self.viz_config['edge_alpha_light'], 
                                                 edge_color=self.viz_config['edge_color'], ax=ax)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Drawing nodes by role")
                    # Draw nodes by role with different shapes
                    if node_shapes_core:
                        core_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_core]
                        core_sizes = [node_sizes[list(G.nodes()).index(node)] for node in node_shapes_core]
                        nx.draw_networkx_nodes(G, pos, nodelist=node_shapes_core,
                                             node_color=core_colors, node_size=core_sizes,
                                             node_shape=self.viz_config['core_node_shape'],
                                             alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax)
                    
                    if node_shapes_periphery:
                        periphery_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_periphery]
                        periphery_sizes = [node_sizes[list(G.nodes()).index(node)] for node in node_shapes_periphery]
                        nx.draw_networkx_nodes(G, pos, nodelist=node_shapes_periphery,
                                             node_color=periphery_colors, node_size=periphery_sizes,
                                             node_shape=self.viz_config['periphery_node_shape'],
                                             alpha=0.8, edgecolors='gray', linewidths=0.5, ax=ax)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Adding selective semantic labels")
                    # SELECTIVE LABELING - Only label core nodes, never structural tokens
                    labels_to_draw = {}
                    
                    if self.viz_config['label_core_only']:
                        # Only label core nodes
                        candidate_nodes = node_shapes_core
                    else:
                        # Label all nodes above importance threshold
                        importance_threshold = np.percentile(list(importance_scores.values()), 
                                                           self.viz_config['label_importance_threshold'] * 100)
                        candidate_nodes = [node for node in G.nodes() 
                                         if importance_scores.get(node, 0) >= importance_threshold]
                    
                    # Never label structural tokens
                    if self.viz_config['never_label_structural']:
                        candidate_nodes = [node for node in candidate_nodes 
                                         if not is_structural.get(node, False)]
                    
                    # Group nodes by community for balanced labeling
                    community_nodes = defaultdict(list)
                    for node in candidate_nodes:
                        community = communities.get(node, 0)
                        importance = importance_scores.get(node, 0)
                        community_nodes[community].append((node, importance))
                    
                    # Select top nodes per community (max 3 per community)
                    for community, nodes in community_nodes.items():
                        # Sort by importance and take top N
                        top_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)[:self.viz_config['max_labels_per_community']]
                        for node, _ in top_nodes:
                            # Truncate long labels for readability
                            label = node[:15] + "..." if len(node) > 15 else node
                            labels_to_draw[node] = label
                    
                    if labels_to_draw:
                        nx.draw_networkx_labels(G, pos, labels_to_draw, 
                                              font_size=9, font_weight='bold', 
                                              font_color='black', ax=ax)
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Adding enhanced legends")
                    # Enhanced title with semantic filtering info
                    structural_removed = len(self.phrase_data.get('filtered_phrases', {})) - G.number_of_nodes()
                    ax.set_title(f'Global Semantic Co-occurrence Network\n'
                               f'{G.number_of_nodes()} nodes ({structural_removed} structural tokens removed), '
                               f'{G.number_of_edges()} edges, {len(unique_communities)} communities\n'
                               f'Seed: {seed} | Density: {nx.density(G)*100:.2f}% | TF-IDF weighted', 
                               fontsize=14, fontweight='bold', pad=20)
                    
                    # Enhanced community legend
                    legend_elements = []
                    for comm in sorted(unique_communities)[:8]:  # Show first 8 communities
                        color = community_colors[comm]
                        legend_elements.append(patches.Patch(color=color, label=f'Community {comm}'))
                    
                    if len(unique_communities) > 8:
                        legend_elements.append(patches.Patch(color='lightgray', label=f'... +{len(unique_communities)-8} more'))
                    
                    # Role and semantic legend
                    legend_elements.append(patches.Patch(color='white', label=''))  # Spacer
                    legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                                    markerfacecolor='gray', markersize=10, label='Core nodes (triangles)'))
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor='gray', markersize=8, label='Periphery nodes (circles)'))
                    legend_elements.append(patches.Patch(color='white', label=''))  # Spacer
                    legend_elements.append(patches.Patch(color='lightgray', label='Node size: TF-IDF score'))
                    legend_elements.append(patches.Patch(color='lightgray', label='Edge alpha: Community relationship'))
                    
                    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                            frameon=True, fancybox=True, shadow=True)
                    
                    ax.axis('off')
                    plt.tight_layout()
                    pbar.update(1)
                    
                    pbar.set_description("🌐 Saving high-resolution visualization")
                    global_viz_name = f"global_thematic_network_seed{seed}_{timestamp}.png"
                    global_viz_path = os.path.join(viz_dir, global_viz_name)
                    
                    # Always export physical image file (PNG) with high resolution
                    plt.savefig(global_viz_path, bbox_inches='tight', facecolor='white', 
                              dpi=self.viz_config['output_dpi'], format=self.viz_config['export_format'])
                    plt.close()
                    
                    # Print absolute output image path after generation
                    print(f"      ✅ Saved: {os.path.basename(global_viz_path)}")
                    print(f"      📁 Full path: {os.path.abspath(global_viz_path)}")
                    
                    self.visualization_paths['global'] = os.path.abspath(global_viz_path)
                    
                    self.visualization_paths['global_graph'] = global_viz_path
                    pbar.update(1)
                
                print(f"      ✅ Saved: {global_viz_name}")
            
            # 2. STATE SUBGRAPH VISUALIZATIONS - HIGHLIGHTED SUBSETS
            subgraph_items = list(self.state_subgraph_objects.items())
            
            for state, subgraph in tqdm(subgraph_items, desc="🎨 Generating state thematic networks", unit="subgraph"):
                if subgraph.number_of_nodes() > 0:
                    with tqdm(total=8, desc=f"🎨 {state} thematic network", unit="step", leave=False) as step_pbar:
                        step_pbar.set_description(f"🎨 {state}: Setting up figure")
                        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
                        
                        # Use same positions as global graph for consistency
                        subgraph_pos = {node: self.global_layout_positions[node] for node in subgraph.nodes() 
                                      if node in self.global_layout_positions}
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Preparing attributes")
                        # Get node attributes from global graph (maintain consistency)
                        communities = {node: self.global_graph_object.nodes[node].get('community', 0) 
                                     for node in subgraph.nodes()}
                        importance_scores = {node: self.global_graph_object.nodes[node].get('importance', 0) 
                                           for node in subgraph.nodes()}
                        node_roles = {node: self.global_graph_object.nodes[node].get('role', 'periphery') 
                                    for node in subgraph.nodes()}
                        
                        # Use same color scheme as global graph
                        unique_communities = sorted(set(communities.values()))
                        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_communities)))
                        community_colors = {comm: colors[i % len(colors)] for i, comm in enumerate(unique_communities)}
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Computing visual attributes")
                        # Node visual attributes (consistent with global)
                        node_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in subgraph.nodes()]
                        
                        node_sizes = []
                        node_shapes_core = []
                        node_shapes_periphery = []
                        
                        for node in subgraph.nodes():
                            importance = importance_scores.get(node, 0)
                            size = self.viz_config['min_node_size'] + (self.viz_config['max_node_size'] - self.viz_config['min_node_size']) * importance
                            node_sizes.append(size)
                            
                            role = node_roles.get(node, 'periphery')
                            if role == 'core':
                                node_shapes_core.append(node)
                            else:
                                node_shapes_periphery.append(node)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Drawing edges")
                        # 简化边绘制避免卡住
                        if subgraph.number_of_edges() > 0:
                            # 限制边数并简化绘制
                            edge_list = list(subgraph.edges(data=True))[:30]  # 最多30条边
                            if edge_list:
                                nx.draw_networkx_edges(subgraph, subgraph_pos, 
                                                     edgelist=[(u, v) for u, v, _ in edge_list],
                                                     width=1.0, alpha=0.3, edge_color='gray', ax=ax)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Drawing nodes")
                        # Draw nodes by role
                        if node_shapes_core:
                            core_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_core]
                            core_sizes = [node_sizes[list(subgraph.nodes()).index(node)] for node in node_shapes_core]
                            nx.draw_networkx_nodes(subgraph, subgraph_pos, nodelist=node_shapes_core,
                                                 node_color=core_colors, node_size=core_sizes,
                                                 node_shape=self.viz_config['core_node_shape'],
                                                 alpha=0.9, edgecolors='black', linewidths=1.5, ax=ax)
                        
                        if node_shapes_periphery:
                            periphery_colors = [community_colors.get(communities.get(node, 0), 'lightblue') for node in node_shapes_periphery]
                            periphery_sizes = [node_sizes[list(subgraph.nodes()).index(node)] for node in node_shapes_periphery]
                            nx.draw_networkx_nodes(subgraph, subgraph_pos, nodelist=node_shapes_periphery,
                                                 node_color=periphery_colors, node_size=periphery_sizes,
                                                 node_shape=self.viz_config['periphery_node_shape'],
                                                 alpha=0.8, edgecolors='gray', linewidths=0.5, ax=ax)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Adding labels")
                        # Selective labeling for subgraph
                        labels_to_draw = {}
                        if importance_scores:
                            importance_threshold = np.percentile(list(importance_scores.values()), 70)
                            
                            community_nodes = defaultdict(list)
                            for node in subgraph.nodes():
                                community = communities.get(node, 0)
                                importance = importance_scores.get(node, 0)
                                if importance >= importance_threshold:
                                    community_nodes[community].append((node, importance))
                            
                            for community, nodes in community_nodes.items():
                                top_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)[:2]  # Fewer labels for subgraphs
                                for node, _ in top_nodes:
                                    labels_to_draw[node] = node
                        
                        if labels_to_draw:
                            nx.draw_networkx_labels(subgraph, subgraph_pos, labels_to_draw,
                                                  font_size=9, font_weight='bold',
                                                  font_color='black', ax=ax)
                        step_pbar.update(1)
                        
                        step_pbar.set_description(f"🎨 {state}: Finalizing")
                        doc_count = len([doc for doc in self.cleaned_text_data if doc['state'] == state])
                        core_count = len(node_shapes_core)
                        
                        ax.set_title(f'State {state} Thematic Network\n'
                                   f'{subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges, '
                                   f'{len(unique_communities)} communities\n'
                                   f'{doc_count} documents, {core_count} core nodes | Seed: {seed}', 
                                   fontsize=12, fontweight='bold', pad=15)
                        
                        # Add legend (simplified for subgraphs)
                        legend_elements = []
                        for comm in sorted(unique_communities):
                            color = community_colors[comm]
                            legend_elements.append(patches.Patch(color=color, label=f'Community {comm}'))
                        
                        if len(legend_elements) > 0:
                            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
                        
                        ax.axis('off')
                        plt.tight_layout()
                        
                        state_viz_name = f"state_{state}_thematic_network_seed{seed}_{timestamp}.png"
                        state_viz_path = os.path.join(viz_dir, state_viz_name)
                        plt.savefig(state_viz_path, bbox_inches='tight', facecolor='white', dpi=300)
                        plt.close()
                        
                        self.visualization_paths[f'subgraph_{state}'] = state_viz_path
                        step_pbar.update(1)
                    
                    print(f"      ✅ Saved: {state_viz_name}")
            
            print(f"\n✅ Thematic network visualization generation completed!")
            print(f"🎨 Generated {len(self.visualization_paths)} readable visualizations")
            print(f"📁 Output directory: {viz_dir}")
            print(f"🎯 All visualizations use consistent community-aware layout")
            print(f"🔍 Edge filtering applied for readability")
            print(f"🎭 Node roles visualized (core=triangles, periphery=circles)")
            
            self.pipeline_state['results_exported'] = True
            
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
            import traceback
            traceback.print_exc()

    def view_output_image_paths(self):
        """View output image paths clearly"""
        if not hasattr(self, 'visualization_paths') or not self.visualization_paths:
            print("⚠️ No visualizations generated yet. Use step 6.1 first.")
            return
        
        print("\n📁 OUTPUT IMAGE PATHS")
        print("-" * 60)
        print("🎨 All visualization files with complete paths:")
        print()
        
        for viz_name, viz_path in self.visualization_paths.items():
            abs_path = os.path.abspath(viz_path)
            print(f"📊 {viz_name}:")
            print(f"   Path: {abs_path}")
            print(f"   Directory: {os.path.dirname(abs_path)}")
            print(f"   Filename: {os.path.basename(abs_path)}")
            print()
        
        print(f"📁 Base visualization directory: {os.path.abspath(os.path.join(self.output_dir, 'visualizations'))}")
    
    def export_complete_results(self):
        """Export complete research results"""
        print("\n📦 EXPORT COMPLETE RESEARCH RESULTS")
        print("-" * 60)
        
        try:
            # Create exports directory
            export_dir = os.path.join(self.output_dir, "exports")
            os.makedirs(export_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create comprehensive results package
            results_package = {
                'export_info': {
                    'timestamp': datetime.now().isoformat(),
                    'pipeline_version': '4.0.0',
                    'export_type': 'complete_research_results'
                },
                'reproducibility_config': self.reproducibility_config.copy(),
                'pipeline_state': self.pipeline_state.copy(),
                'input_summary': {
                    'input_directory': self.input_directory,
                    'file_count': len(self.input_files),
                    'document_count': len(self.cleaned_text_data) if hasattr(self, 'cleaned_text_data') else 0
                },
                'processing_results': {}
            }
            
            # Add processing results if available
            if hasattr(self, 'cleaned_text_data'):
                results_package['processing_results']['text_cleaning'] = {
                    'document_count': len(self.cleaned_text_data),
                    'total_tokens': sum(doc['token_count'] for doc in self.cleaned_text_data)
                }
            
            if hasattr(self, 'phrase_data'):
                results_package['processing_results']['phrase_extraction'] = {
                    'total_phrases': len(self.phrase_data['all_phrases']),
                    'unique_phrases': len(self.phrase_data['phrase_counts']),
                    'filtered_phrases': len(self.phrase_data['filtered_phrases'])
                }
            
            if hasattr(self, 'global_graph_object') and self.global_graph_object:
                results_package['processing_results']['global_graph'] = {
                    'node_count': self.global_graph_object.number_of_nodes(),
                    'edge_count': self.global_graph_object.number_of_edges(),
                    'density': nx.density(self.global_graph_object) if self.global_graph_object.number_of_nodes() > 1 else 0,
                    'connected_components': nx.number_connected_components(self.global_graph_object),
                    'layout_computed': self.global_layout_positions is not None,
                    'communities_detected': len(set(nx.get_node_attributes(self.global_graph_object, 'community').values())),
                    'graph_object_type': 'NetworkX Graph'
                }
            
            if hasattr(self, 'state_subgraph_objects') and self.state_subgraph_objects:
                results_package['processing_results']['state_subgraphs'] = {
                    state: {
                        'node_count': subgraph.number_of_nodes(),
                        'edge_count': subgraph.number_of_edges(),
                        'density': nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0,
                        'document_count': len([doc for doc in self.cleaned_text_data if doc['state'] == state]),
                        'graph_object_type': 'NetworkX SubGraph View'
                    }
                    for state, subgraph in self.state_subgraph_objects.items()
                }
            
            if hasattr(self, 'visualization_paths'):
                results_package['processing_results']['visualizations'] = {
                    'generated_count': len(self.visualization_paths),
                    'output_paths': self.visualization_paths.copy()
                }
            
            # Export results package
            results_file = os.path.join(export_dir, f"complete_results_{timestamp}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_package, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Complete results exported: {results_file}")
            print("📋 Package includes:")
            print("   - Reproducibility configuration")
            print("   - Pipeline state and processing statistics")
            print("   - Input/output summaries")
            print("   - Visualization paths")
            print("🔬 This file provides complete traceability for research reproducibility")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def view_graph_nodes_and_data(self):
        """Export detailed graph node and data information to document"""
        print("\n📊 EXPORT GRAPH NODES & DATA DETAILS")
        print("-" * 60)
        
        # Check prerequisites
        if not hasattr(self, 'visualization_paths') or not self.visualization_paths:
            print("⚠️ No visualizations generated yet. Please run step 6.1 first.")
            return
        
        if not hasattr(self, 'global_graph_object') or not self.global_graph_object:
            print("⚠️ No global graph available. Please run step 4.1 first.")
            return
        
        # Show available graphs
        print("📈 Available graphs for analysis:")
        print("0. Global Graph (complete network)")
        
        available_subgraphs = []
        if hasattr(self, 'state_subgraph_objects') and self.state_subgraph_objects:
            for i, (state, subgraph) in enumerate(self.state_subgraph_objects.items(), 1):
                print(f"{i}. State {state} Subgraph ({subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges)")
                available_subgraphs.append((state, subgraph))
        
        print("A. All graphs (global + 3 random subgraphs)")
        
        # Get user selection
        choice = self.get_user_choice("Select graph to analyze", 
                                    ["0", "A"] + [str(i) for i in range(1, len(available_subgraphs) + 1)])
        
        # Create output directory
        output_dir = os.path.join(self.output_dir, "graph_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if choice == "0":
            # Export global graph
            self._export_single_graph_data(self.global_graph_object, "global", output_dir, timestamp)
            
        elif choice == "A":
            # Export global graph + 3 random subgraphs
            self._export_single_graph_data(self.global_graph_object, "global", output_dir, timestamp)
            
            # Select 3 random subgraphs
            import random
            selected_subgraphs = random.sample(available_subgraphs, min(3, len(available_subgraphs)))
            
            for state, subgraph in selected_subgraphs:
                self._export_single_graph_data(subgraph, f"state_{state}", output_dir, timestamp)
            
            print(f"✅ Exported global graph + {len(selected_subgraphs)} random subgraphs")
            
        else:
            # Export specific subgraph
            idx = int(choice) - 1
            if 0 <= idx < len(available_subgraphs):
                state, subgraph = available_subgraphs[idx]
                self._export_single_graph_data(subgraph, f"state_{state}", output_dir, timestamp)
        
        print(f"📁 All analysis files saved to: {os.path.abspath(output_dir)}")
    
    def _export_single_graph_data(self, graph, graph_name, output_dir, timestamp):
        """Export detailed data for a single graph to document"""
        
        # Prepare filename
        filename = f"graph_data_{graph_name}_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Minimal header with essential info
            f.write(f"# Graph Data Export: {graph_name.upper()}\n")
            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"# Random_Seed: {self.reproducibility_config['random_seed']}\n\n")
            
            # Graph structure - raw data only
            f.write("# GRAPH_STRUCTURE\n")
            f.write(f"nodes={graph.number_of_nodes()}\n")
            f.write(f"edges={graph.number_of_edges()}\n")
            f.write(f"density={nx.density(graph):.6f}\n")
            f.write(f"connected_components={nx.number_connected_components(graph)}\n")
            f.write(f"isolated_nodes={len(list(nx.isolates(graph)))}\n\n")
            
            # All node data - complete attribute table
            f.write("# NODE_DATA\n")
            all_attributes = set()
            for node in graph.nodes():
                all_attributes.update(graph.nodes[node].keys())
            
            # Header row
            f.write("node_id")
            for attr in sorted(all_attributes):
                f.write(f"\t{attr}")
            f.write("\n")
            
            # Data rows
            for node in sorted(graph.nodes()):
                f.write(f"{node}")
                for attr in sorted(all_attributes):
                    value = graph.nodes[node].get(attr, "")
                    if isinstance(value, (list, tuple, np.ndarray)):
                        if len(value) == 2:  # Position coordinates
                            value = f"{value[0]:.6f},{value[1]:.6f}"
                        else:
                            value = str(value).replace('\t', ' ')
                    elif isinstance(value, float):
                        value = f"{value:.6f}"
                    f.write(f"\t{value}")
                f.write("\n")
            f.write("\n")
            
            # All edge data - complete attribute table
            f.write("# EDGE_DATA\n")
            if graph.number_of_edges() > 0:
                edge_attributes = set()
                for u, v, data in graph.edges(data=True):
                    edge_attributes.update(data.keys())
                
                # Header row
                f.write("source\ttarget")
                for attr in sorted(edge_attributes):
                    f.write(f"\t{attr}")
                f.write("\n")
                
                # Data rows
                for u, v, data in graph.edges(data=True):
                    f.write(f"{u}\t{v}")
                    for attr in sorted(edge_attributes):
                        value = data.get(attr, "")
                        if isinstance(value, float):
                            value = f"{value:.6f}"
                        f.write(f"\t{value}")
                    f.write("\n")
            else:
                f.write("# No edges in graph\n")
            f.write("\n")
            
            # Processing parameters - all variables used
            f.write("# PROCESSING_PARAMETERS\n")
            for key, value in sorted(self.reproducibility_config.items()):
                f.write(f"{key}={value}\n")
            f.write("\n")
            
            # Graph construction config if available
            if hasattr(self, 'graph_construction_config'):
                f.write("# GRAPH_CONSTRUCTION_CONFIG\n")
                for key, value in sorted(self.graph_construction_config.items()):
                    f.write(f"{key}={value}\n")
                f.write("\n")
            
            # Source data metrics
            if hasattr(self, 'cleaned_text_data') and self.cleaned_text_data:
                f.write("# SOURCE_DATA_METRICS\n")
                total_docs = len(self.cleaned_text_data)
                total_tokens = sum(doc['token_count'] for doc in self.cleaned_text_data)
                f.write(f"total_documents={total_docs}\n")
                f.write(f"total_tokens={total_tokens}\n")
                f.write(f"avg_tokens_per_doc={total_tokens/total_docs:.2f}\n")

                # State distribution
                state_stats = defaultdict(lambda: {'docs': 0, 'tokens': 0})
                for doc in self.cleaned_text_data:
                    state = doc['state']
                    state_stats[state]['docs'] += 1
                    state_stats[state]['tokens'] += doc['token_count']
                
                f.write("# STATE_DISTRIBUTION\n")
                for state, stats in sorted(state_stats.items()):
                    f.write(f"{state}_docs={stats['docs']}\n")
                    f.write(f"{state}_tokens={stats['tokens']}\n")
                f.write("\n")
            
            # Phrase extraction metrics
            if hasattr(self, 'phrase_data') and self.phrase_data:
                f.write("# PHRASE_EXTRACTION_METRICS\n")
                f.write(f"total_phrase_instances={len(self.phrase_data['all_phrases'])}\n")
                f.write(f"unique_phrases={len(self.phrase_data['phrase_counts'])}\n")
                f.write(f"filtered_phrases={len(self.phrase_data['filtered_phrases'])}\n")
                f.write(f"min_frequency_threshold={self.reproducibility_config['min_phrase_frequency']}\n")
                f.write(f"phrase_type={self.reproducibility_config['phrase_type']}\n")
                
                # Top phrases only (first 50)
                f.write("# TOP_PHRASES\n")
                sorted_phrases = sorted(self.phrase_data['filtered_phrases'].items(), 
                                      key=lambda x: x[1], reverse=True)
                for phrase, freq in sorted_phrases[:50]:
                    f.write(f"{phrase}={freq}\n")
                f.write("\n")
            
            # Layout algorithm parameters
            if hasattr(self, 'global_layout_positions') and self.global_layout_positions:
                f.write("# LAYOUT_ALGORITHM_PARAMS\n")
                f.write(f"algorithm=spring_layout\n")
                f.write(f"iterations=50\n")
                f.write(f"k_parameter=1.0\n")
                f.write(f"random_seed={self.reproducibility_config['random_seed']}\n")
                f.write(f"positions_computed={len(self.global_layout_positions)}\n")
                f.write("\n")
        
        print(f"✅ Exported {graph_name} data: {filename}")
        return filepath
    
    def create_sample_research_data(self):
        """Create sample research data directory"""
        print("\n📊 CREATE SAMPLE RESEARCH DATA")
        print("-" * 50)
        
        # Create sample directory
        sample_dir = "sample_research_data"
        os.makedirs(sample_dir, exist_ok=True)
        
        # Research-oriented sample data with TOC structure
        research_documents = [
            {
                "segment_id": "research_001",
                "title": "Machine Learning Fundamentals",
                "level": 1,
                "order": 1,
                "text": "Machine learning algorithms enable computers to learn patterns from data without explicit programming. Supervised learning uses labeled datasets to train models for prediction tasks. Unsupervised learning discovers hidden patterns in unlabeled data through clustering and dimensionality reduction techniques.",
                "state": "CA",
                "language": "english"
            },
            {
                "segment_id": "research_002", 
                "title": "Deep Learning Applications",
                "level": 2,
                "order": 2,
                "text": "Deep neural networks have revolutionized computer vision and natural language processing. Convolutional neural networks excel at image recognition tasks. Recurrent neural networks and transformers process sequential data for language modeling and machine translation applications.",
                "state": "CA",
                "language": "english"
            },
            {
                "segment_id": "research_003",
                "title": "Natural Language Processing Methods",
                "level": 2,
                "order": 3,
                "text": "Natural language processing combines computational linguistics with machine learning. Text preprocessing includes tokenization, stemming, and stopword removal. Named entity recognition identifies people, organizations, and locations in text documents.",
                "state": "NY",
                "language": "english"
            },
            {
                "segment_id": "research_004",
                "title": "数据科学研究方法",
                "level": 1,
                "order": 4,
                "text": "数据科学结合统计学、计算机科学和领域专业知识来分析复杂数据。机器学习算法用于预测建模和模式识别。数据可视化技术帮助研究人员理解数据分布和趋势。",
                "state": "NY",
                "language": "chinese"
            },
            {
                "segment_id": "research_005",
                "title": "人工智能伦理考量",
                "level": 2,
                "order": 5,
                "text": "人工智能系统的公平性和透明度是重要的研究议题。算法偏见可能导致不公平的决策结果。可解释人工智能技术帮助理解模型的决策过程和推理逻辑。",
                "state": "TX",
                "language": "chinese"
            },
            {
                "segment_id": "research_006",
                "title": "Graph Neural Networks",
                "level": 1,
                "order": 6,
                "text": "Graph neural networks extend deep learning to graph-structured data. Message passing algorithms aggregate information from neighboring nodes. Graph convolutional networks learn node representations for tasks like node classification and link prediction.",
                "state": "TX",
                "language": "english"
            },
            {
                "segment_id": "research_007",
                "title": "Network Analysis Techniques",
                "level": 2,
                "order": 7,
                "text": "Social network analysis studies relationships between entities using graph theory. Centrality measures identify important nodes in networks. Community detection algorithms discover clusters of densely connected nodes in large networks.",
                "state": "FL",
                "language": "english"
            },
            {
                "segment_id": "research_008",
                "title": "Text Mining and Information Extraction",
                "level": 2,
                "order": 8,
                "text": "Text mining extracts valuable information from unstructured text documents. Topic modeling techniques like Latent Dirichlet Allocation discover thematic structures. Sentiment analysis determines emotional polarity in text using machine learning classifiers.",
                "state": "FL",
                "language": "english"
            }
        ]
        
        # Save documents to separate files
        self.input_files = []
        for i, doc in enumerate(research_documents):
            file_path = os.path.join(sample_dir, f"research_doc_{i+1:02d}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            self.input_files.append(file_path)
        
        # Set directory settings
        self.input_directory = sample_dir
        self.pipeline_state['data_loaded'] = True
        
        print(f"✅ Created {len(research_documents)} research documents in: {sample_dir}")
        print("📊 Sample data includes:")
        print("   - Multi-language content (English & Chinese)")
        print("   - Multiple research states (CA, NY, TX, FL)")
        print("   - TOC-structured segments with hierarchical levels")
        print("   - Research topics: ML, NLP, AI Ethics, Graph Networks")
        print("🔬 Ready for research pipeline processing!")
    
    def run(self):
        """Run research pipeline interface"""
        while True:
            self.print_menu()
            choice = self.get_user_choice()
            
            if choice == "0":
                print("\n👋 Research pipeline session ended. Thank you!")
                break
            
            # Data Input & Directory Processing
            elif choice == "1.1":
                self.select_input_directory()
            elif choice == "1.2":
                self.set_output_directory()
            elif choice == "1.3":
                self.show_data_settings()
            
            # Text Cleaning & Normalization
            elif choice == "2.1":
                self.clean_and_normalize_text()
            elif choice == "2.2":
                self.export_cleaned_text()
            elif choice == "2.3":
                self.view_text_cleaning_results()
            
            # Token/Phrase Construction
            elif choice == "3.1":
                self.configure_phrase_parameters()
            elif choice == "3.2":
                self.extract_tokens_and_phrases()
            elif choice == "3.3":
                self.view_phrase_statistics()
            
            # Global Graph Construction
            elif choice == "4.1":
                self.build_global_graph()
            elif choice == "4.2":
                self.apply_scientific_optimization()
            elif choice == "4.3":
                self.view_global_graph_statistics()
            elif choice == "4.4":
                self.export_global_graph_data()
            
            # Subgraph Activation
            elif choice == "5.1":
                self.activate_state_subgraphs()
            elif choice == "5.2":
                self.view_subgraph_comparisons()
            elif choice == "5.3":
                self.export_subgraph_data()
            
            # Visualization & Export
            elif choice == "6.1":
                self.generate_scientific_visualizations()
            elif choice == "6.2":
                self.view_output_image_paths()
            elif choice == "6.3":
                self.export_complete_results()
            elif choice == "6.4":
                self.view_graph_nodes_and_data()
            
            # Scientific Controls
            elif choice == "S.1":
                self.configure_scientific_parameters()
            elif choice == "S.2":
                self.view_scientific_statistics()
            elif choice == "S.3":
                self.export_scientific_report()
            
            # Reproducibility Controls
            elif choice == "R.1":
                self.configure_reproducibility_parameters()
            elif choice == "R.2":
                self.view_all_parameters()
            elif choice == "R.3":
                self.export_parameter_configuration()
            
            # Utilities
            elif choice == "U.1":
                self.create_sample_research_data()
            elif choice == "U.2":
                self.show_system_status()
            elif choice == "U.3":
                self.show_research_workflow_help()
            
            else:
                print("❌ Invalid choice. Please select a valid option.")
            
            # Wait for user to continue (except for exit)
            if choice != "0":
                input("\n🔬 Press Enter to continue research workflow...")
    
    def show_system_status(self):
        """Show system and pipeline status"""
        print("\n💻 SYSTEM & PIPELINE STATUS")
        print("-" * 50)
        print(f"Python version: {sys.version}")
        print(f"Operating system: {os.name}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Pipeline availability: {'✅ Available' if PIPELINE_AVAILABLE else '❌ Unavailable'}")
        if not PIPELINE_AVAILABLE:
            print(f"Import error: {IMPORT_ERROR}")
        
        print(f"\n🔬 RESEARCH PIPELINE STATE:")
        for step, completed in self.pipeline_state.items():
            status = "✅ Completed" if completed else "⏳ Pending"
            print(f"   {step.replace('_', ' ').title()}: {status}")
    
    def show_research_workflow_help(self):
        """Show research workflow help"""
        print("\n📖 RESEARCH WORKFLOW HELP")
        print("-" * 50)
        print("🔬 This is a research-oriented text-to-co-occurrence-graph pipeline.")
        print("📋 Follow the fixed workflow order for reproducible results:")
        print()
        print("WORKFLOW STEPS:")
        print("1️⃣ Data Input: Select directory with text documents")
        print("2️⃣ Text Cleaning: Clean and normalize with preview/export")
        print("3️⃣ Phrase Construction: Extract tokens/phrases with parameters")
        print("4️⃣ Global Graph: Build shared co-occurrence node space")
        print("5️⃣ Subgraph Activation: Filter global graph by state/group")
        print("6️⃣ Visualization: Generate deterministic layouts and export")
        print()
        print("REPRODUCIBILITY FEATURES:")
        print("🌱 Fixed random seed for deterministic results")
        print("🪟 Explicit co-occurrence window (TOC segment = one window)")
        print("⚖️ Configurable edge weight and phrase strategies")
        print("📊 Parameter export for complete traceability")
        print("🎯 Clear distinction between global graph and subgraphs")
        print()
        print("SUPPORTED FORMATS:")
        print("📁 Input: JSON, TXT, MD files (batch directory processing)")
        print("💾 Output: JSON data, PNG visualizations, parameter configs")
        print()
        print("For issues, check system status (U.2) and ensure workflow order.")
    
    def load_input_data(self):
        """Load input data from selected files"""
        if not self.input_files:
            print("📝 No input files selected")
            return []
        
        all_data = []
        for file_path in self.input_files:
            try:
                print(f"📖 Loading: {os.path.relpath(file_path, self.input_directory) if self.input_directory else file_path}")
                
                # Extract state from folder path
                state = "Unknown"
                if self.input_directory:
                    rel_path = os.path.relpath(file_path, self.input_directory)
                    path_parts = rel_path.split(os.sep)
                    if len(path_parts) > 1:
                        # Use the immediate parent folder as state
                        state = path_parts[-2]
                    else:
                        # Use the base directory name as state
                        state = os.path.basename(self.input_directory)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.endswith('.json'):
                        data = json.load(f)
                        if isinstance(data, list):
                            # Update state for each document in the list
                            for doc in data:
                                if 'state' not in doc or doc['state'] == 'Unknown':
                                    doc['state'] = state
                            all_data.extend(data)
                        else:
                            # Update state for single document
                            if 'state' not in data or data['state'] == 'Unknown':
                                data['state'] = state
                            all_data.append(data)
                    else:
                        # Handle text files
                        content = f.read()
                        doc_data = {
                            "segment_id": f"text_{len(all_data)+1}",
                            "title": os.path.basename(file_path),
                            "level": 1,
                            "order": len(all_data)+1,
                            "text": content,
                            "state": state,  # Use extracted state instead of "Unknown"
                            "language": self.reproducibility_config['language_detection'] if self.reproducibility_config['language_detection'] != "auto" else "english"
                        }
                        all_data.append(doc_data)
                        
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")
        
        print(f"✅ Loaded {len(all_data)} documents total")
        return all_data
    

    


def main():
    """Main function for research pipeline"""
    try:
        app = ResearchPipelineCLI()
        app.run()
    except KeyboardInterrupt:
        print("\n\n👋 Research pipeline interrupted by user. Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Research pipeline startup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())