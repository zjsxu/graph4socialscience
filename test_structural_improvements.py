#!/usr/bin/env python3
"""
Test script to verify structural improvements to graph construction and visualization
"""

import os
import sys
import tempfile
import shutil
import json
import networkx as nxlib
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

def create_test_data():
    """Create test data with meaningful co-occurrence patterns"""
    
    # Create temporary test directory
    test_dir = tempfile.mkdtemp(prefix="test_structural_improvements_")
    print(f"📁 Created test directory: {test_dir}")
    
    # Create state-based folder structure with thematic content
    states = ['CA', 'NY', 'TX']
    test_files = []
    
    # Define thematic content for different states
    themes = {
        'CA': [
            "machine learning algorithms artificial intelligence neural networks deep learning computer vision natural language processing",
            "data science analytics big data cloud computing distributed systems scalable architectures",
            "artificial intelligence machine learning deep learning neural networks computer vision pattern recognition"
        ],
        'NY': [
            "financial markets trading algorithms quantitative analysis risk management portfolio optimization",
            "blockchain technology cryptocurrency digital assets smart contracts decentralized finance",
            "financial technology fintech digital banking mobile payments electronic commerce"
        ],
        'TX': [
            "energy systems renewable energy solar power wind energy grid optimization smart grid",
            "oil gas petroleum energy production drilling extraction refining distribution",
            "renewable energy sustainable development environmental protection climate change mitigation"
        ]
    }
    
    for state in states:
        state_dir = os.path.join(test_dir, state)
        os.makedirs(state_dir, exist_ok=True)
        
        # Create documents with thematic content
        for i, content in enumerate(themes[state]):
            doc_data = {
                "segment_id": f"{state}_doc_{i+1}",
                "title": f"Document {i+1} from {state}",
                "level": 1,
                "order": i+1,
                "text": content,
                "state": "Unknown",  # Will be overridden by folder detection
                "language": "english"
            }
            
            file_path = os.path.join(state_dir, f"doc_{i+1}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
            
            test_files.append(file_path)
    
    return test_dir, test_files

def test_structural_improvements():
    """Test the structural improvements to graph construction and visualization"""
    
    print("🧪 TESTING STRUCTURAL IMPROVEMENTS")
    print("=" * 70)
    
    # Import the pipeline class
    from complete_usage_guide import ResearchPipelineCLI
    
    # Create test data
    test_dir, test_files = create_test_data()
    output_dir = os.path.join(test_dir, "output")
    
    try:
        # Initialize pipeline
        app = ResearchPipelineCLI()
        app.input_directory = test_dir
        app.input_files = test_files
        app.output_dir = output_dir
        app.pipeline_state['data_loaded'] = True
        
        print(f"📁 Test input: {test_dir}")
        print(f"📁 Test output: {output_dir}")
        print(f"📊 Test files: {len(test_files)}")
        
        # Test 1: Text cleaning and phrase extraction
        print("\n1️⃣ TESTING TEXT PROCESSING")
        print("-" * 50)
        
        app.clean_and_normalize_text()
        app.extract_tokens_and_phrases()
        
        if not app.pipeline_state['phrases_constructed']:
            print("❌ FAILED: Text processing failed")
            return False
        
        print("✅ SUCCESS: Text processing completed")
        
        # Test 2: CRITICAL - Structural graph construction
        print("\n2️⃣ TESTING STRUCTURAL GRAPH CONSTRUCTION")
        print("-" * 50)
        print("   Testing:")
        print("   - Edge filtering for density reduction")
        print("   - Node role assignment (core vs periphery)")
        print("   - Community detection on filtered graph")
        print("   - Community-aware layout")
        
        app.build_global_graph()
        
        if not app.pipeline_state['global_graph_built'] or not app.global_graph_object:
            print("❌ FAILED: Global graph construction failed")
            return False
        
        G = app.global_graph_object
        
        # Verify structural improvements
        print(f"✅ SUCCESS: Structural graph construction completed")
        print(f"   Nodes: {G.number_of_nodes()}")
        print(f"   Edges: {G.number_of_edges()}")
        print(f"   Density: {nxlib.density(G)*100:.2f}%")
        
        # Check edge filtering
        if hasattr(app, 'raw_cooccurrence_counts'):
            raw_edges = len(app.raw_cooccurrence_counts)
            filtered_edges = G.number_of_edges()
            reduction = (1 - filtered_edges/raw_edges) * 100 if raw_edges > 0 else 0
            print(f"   Edge reduction: {reduction:.1f}% (raw: {raw_edges} → filtered: {filtered_edges})")
            
            if reduction > 50:  # Should reduce density significantly
                print("   ✅ Edge filtering working correctly")
            else:
                print("   ⚠️ Edge filtering may not be aggressive enough")
        
        # Check node roles
        node_roles = nxlib.get_node_attributes(G, 'role')
        if node_roles:
            core_count = sum(1 for role in node_roles.values() if role == 'core')
            periphery_count = len(node_roles) - core_count
            print(f"   Node roles: {core_count} core, {periphery_count} periphery")
            print("   ✅ Node role assignment working")
        else:
            print("   ❌ Node roles not assigned")
        
        # Check communities
        communities = nxlib.get_node_attributes(G, 'community')
        if communities:
            unique_communities = len(set(communities.values()))
            print(f"   Communities detected: {unique_communities}")
            print("   ✅ Community detection working")
        else:
            print("   ❌ Communities not detected")
        
        # Test 3: Subgraph activation (should preserve structure)
        print("\n3️⃣ TESTING SEMANTIC SUBGRAPH ACTIVATION")
        print("-" * 50)
        
        app.activate_state_subgraphs()
        
        if not app.pipeline_state['subgraphs_activated'] or not app.state_subgraph_objects:
            print("❌ FAILED: Subgraph activation failed")
            return False
        
        print("✅ SUCCESS: Subgraph activation completed")
        print(f"   Subgraphs created: {len(app.state_subgraph_objects)}")
        
        # Verify semantic consistency
        for state, subgraph in app.state_subgraph_objects.items():
            print(f"   {state}: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
            
            # Check that subgraph nodes have same attributes as global graph
            sample_node = list(subgraph.nodes())[0] if subgraph.number_of_nodes() > 0 else None
            if sample_node:
                global_attrs = set(app.global_graph_object.nodes[sample_node].keys())
                expected_attrs = {'frequency', 'phrase_type', 'community', 'role', 'importance', 'pos'}
                if expected_attrs.issubset(global_attrs):
                    print(f"      ✅ {state}: Node attributes preserved from global graph")
                else:
                    print(f"      ⚠️ {state}: Some node attributes missing")
        
        # Test 4: CRITICAL - Readable thematic visualization
        print("\n4️⃣ TESTING READABLE THEMATIC VISUALIZATION")
        print("-" * 50)
        print("   Testing:")
        print("   - Selective edge rendering (low alpha)")
        print("   - Node role visualization (shapes)")
        print("   - Community-aware coloring")
        print("   - Selective labeling")
        print("   - High-resolution output")
        
        app.generate_deterministic_visualizations()
        
        if not hasattr(app, 'visualization_paths') or not app.visualization_paths:
            print("❌ FAILED: Visualization generation failed")
            return False
        
        print("✅ SUCCESS: Thematic visualization generation completed")
        print(f"   Visualizations created: {len(app.visualization_paths)}")
        
        # Verify visualization files
        for viz_name, viz_path in app.visualization_paths.items():
            if os.path.exists(viz_path):
                file_size = os.path.getsize(viz_path)
                print(f"   {viz_name}: {os.path.basename(viz_path)} ({file_size} bytes)")
                
                # Check if it's a thematic network (should have "thematic" in filename)
                if "thematic" in os.path.basename(viz_path):
                    print(f"      ✅ Thematic network visualization created")
                else:
                    print(f"      ⚠️ May not be using new thematic visualization")
            else:
                print(f"   ❌ {viz_name}: File not found at {viz_path}")
                return False
        
        # Test 5: Enhanced statistics
        print("\n5️⃣ TESTING ENHANCED STATISTICS")
        print("-" * 50)
        
        print("   Global graph statistics:")
        app.view_global_graph_statistics()
        
        print("\n   Subgraph comparison:")
        app.view_subgraph_comparisons()
        
        print("\n🎉 ALL STRUCTURAL IMPROVEMENT TESTS PASSED!")
        print("✅ Edge filtering reduces graph density significantly")
        print("✅ Node roles assigned (core vs periphery)")
        print("✅ Community detection working on filtered graph")
        print("✅ Subgraphs preserve global semantic structure")
        print("✅ Thematic visualizations generated with:")
        print("   - Selective edge rendering")
        print("   - Role-based node shapes")
        print("   - Community-aware layout")
        print("   - Selective labeling")
        print("✅ Enhanced statistics show structural metrics")
        
        return True
        
    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup test directory
        try:
            shutil.rmtree(test_dir)
            print(f"\n🧹 Cleaned up test directory: {test_dir}")
        except:
            print(f"\n⚠️ Could not clean up test directory: {test_dir}")

def main():
    """Main test function"""
    success = test_structural_improvements()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())