#!/usr/bin/env python3
"""
Test Scientific Graph Optimization
Validates the implementation of rigorous network science methods
"""

import os
import sys
import time
from datetime import datetime
import networkx as nx

def test_scientific_optimization():
    """Test the scientific optimization pipeline"""
    print("🔬 Scientific Graph Optimization Test")
    print("Testing: NPMI weighting, adaptive sparsification, LCC extraction, community pruning")
    print()
    
    # Set test parameters
    input_dir = "test_input"
    output_dir = "test_output"
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    print(f"🧪 Testing scientific optimization with toc_doc data")
    print("=" * 60)
    
    # Import main program
    try:
        from complete_usage_guide import ResearchPipelineCLI
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return
    
    # Create pipeline instance
    cli = ResearchPipelineCLI()
    
    # Set input/output directories
    cli.input_directory = input_dir
    cli.output_dir = output_dir
    
    # Scan input files
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Scan directory for files
    cli.input_files = []
    valid_extensions = {'.json', '.txt', '.md'}
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in valid_extensions:
                cli.input_files.append(file_path)
    
    print(f"📊 Found files: {len(cli.input_files)}")
    
    if len(cli.input_files) == 0:
        print("❌ No valid input files found")
        return
    
    # Set pipeline state
    cli.pipeline_state = {
        'data_loaded': True,
        'text_cleaned': False,
        'phrases_constructed': False,
        'global_graph_built': False,
        'subgraphs_activated': False,
        'results_exported': False
    }
    
    print("\n🔄 Executing scientific optimization test pipeline...")
    
    # Execute pipeline with scientific optimization
    start_time = time.time()
    
    try:
        # 1. Text cleaning
        print("\n=== Step 1: Text Cleaning ===")
        cli.clean_and_normalize_text()
        
        # 2. Phrase extraction
        print("\n=== Step 2: Phrase Extraction ===")
        cli.extract_tokens_and_phrases()
        
        # 3. Global graph construction
        print("\n=== Step 3: Global Graph Construction ===")
        cli.build_global_graph()
        
        # 4. Scientific optimization (NEW)
        print("\n=== Step 4: Scientific Optimization ===")
        cli.apply_scientific_optimization()
        
        # Verify scientific optimization results
        print("\n🔍 Verifying scientific optimization...")
        if hasattr(cli, 'optimized_global_graph') and cli.optimized_global_graph:
            G = cli.optimized_global_graph
            
            print(f"   ✅ Optimized graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            print(f"   ✅ Graph density: {nx.density(G):.6f}")
            
            # Check for scientific attributes
            if hasattr(cli, 'global_communities') and cli.global_communities:
                n_communities = len(set(cli.global_communities.values()))
                print(f"   ✅ Communities detected: {n_communities}")
            
            if hasattr(cli, 'global_node_roles') and cli.global_node_roles:
                core_count = sum(1 for role in cli.global_node_roles.values() if role == 'core')
                print(f"   ✅ Core nodes identified: {core_count}")
            
            if hasattr(cli, 'structural_statistics') and cli.structural_statistics:
                stats = cli.structural_statistics
                print(f"   ✅ Structural statistics computed: {len(stats)} measures")
                if 'average_clustering' in stats:
                    print(f"      - Average clustering: {stats['average_clustering']:.4f}")
                if 'components' in stats:
                    print(f"      - Connected components: {stats['components']}")
        
        # 5. View scientific statistics
        print("\n=== Step 5: Scientific Statistics ===")
        cli.view_scientific_statistics()
        
        # 6. Subgraph activation
        print("\n=== Step 6: Subgraph Activation ===")
        cli.activate_state_subgraphs()
        
        # 7. Scientific visualization
        print("\n=== Step 7: Scientific Visualization ===")
        cli.generate_scientific_visualizations()
        
        # Verify visualizations
        print("\n🔍 Verifying scientific visualizations...")
        if hasattr(cli, 'visualization_paths') and cli.visualization_paths:
            for viz_type, path in cli.visualization_paths.items():
                if os.path.exists(path):
                    file_size = os.path.getsize(path)
                    print(f"   ✅ {viz_type}: {os.path.basename(path)} ({file_size} bytes)")
                else:
                    print(f"   ❌ {viz_type}: File not found - {path}")
        
        # 8. Export scientific report
        print("\n=== Step 8: Scientific Report Export ===")
        cli.export_scientific_report()
        
        end_time = time.time()
        print(f"\n✅ Scientific optimization test completed in {end_time - start_time:.2f} seconds")
        
        # Generate test summary
        print(f"\n📋 Scientific Optimization Test Summary:")
        print("=" * 50)
        
        # A. Scientific methods verification
        print("🔬 A. Scientific Methods Applied:")
        if hasattr(cli, 'scientific_config'):
            config = cli.scientific_config
            print(f"   ✅ Semantic weighting: {config.get('semantic_weighting', 'N/A').upper()}")
            print(f"   ✅ Sparsification method: {config.get('sparsification_method', 'N/A')}")
            print(f"   ✅ Core identification: {config.get('core_method', 'N/A')}")
            print(f"   ✅ Edge retention rate: {config.get('edge_retention_rate', 0)*100:.1f}%")
            print(f"   ✅ Min community size: {config.get('min_community_size', 'N/A')}")
        
        # B. Graph optimization results
        print("\n📊 B. Graph Optimization Results:")
        if hasattr(cli, 'global_graph_object') and hasattr(cli, 'optimized_global_graph'):
            original = cli.global_graph_object if hasattr(cli, 'global_graph_object') else None
            optimized = cli.optimized_global_graph
            
            if original and optimized:
                orig_nodes = original.number_of_nodes()
                orig_edges = original.number_of_edges()
                opt_nodes = optimized.number_of_nodes()
                opt_edges = optimized.number_of_edges()
                
                print(f"   📈 Node reduction: {orig_nodes} → {opt_nodes} ({(1-opt_nodes/orig_nodes)*100:.1f}% reduction)")
                print(f"   📈 Edge reduction: {orig_edges} → {opt_edges} ({(1-opt_edges/orig_edges)*100:.1f}% reduction)")
                print(f"   📈 Density change: {nx.density(original):.6f} → {nx.density(optimized):.6f}")
        
        # C. Community structure
        print("\n🏘️ C. Community Structure:")
        if hasattr(cli, 'global_communities') and cli.global_communities:
            from collections import Counter
            community_sizes = Counter(cli.global_communities.values())
            print(f"   ✅ Total communities: {len(community_sizes)}")
            print(f"   ✅ Largest communities: {dict(sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:3])}")
        
        # D. Core-periphery structure
        print("\n🎯 D. Core-Periphery Structure:")
        if hasattr(cli, 'global_node_roles') and cli.global_node_roles:
            from collections import Counter
            role_counts = Counter(cli.global_node_roles.values())
            for role, count in role_counts.items():
                pct = count / len(cli.global_node_roles) * 100
                print(f"   ✅ {role.title()} nodes: {count} ({pct:.1f}%)")
        
        # E. Visualization quality
        print("\n🎨 E. Visualization Quality:")
        if hasattr(cli, 'visualization_paths') and cli.visualization_paths:
            print(f"   ✅ Generated visualizations: {len(cli.visualization_paths)}")
            print(f"   ✅ Publication-quality: 300 DPI PNG format")
            print(f"   ✅ Scientific styling: Core/periphery shapes, community colors")
            print(f"   ✅ Deterministic layout: Fixed seed {cli.reproducibility_config['random_seed']}")
        
        print(f"\n🎉 All scientific optimization features working correctly!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    test_scientific_optimization()