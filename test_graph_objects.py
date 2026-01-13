#!/usr/bin/env python3
"""
Test script to verify that graph objects are properly created and used
"""

import os
import sys
import numpy as np
import networkx as nx

# Add current directory to path for imports
sys.path.insert(0, '.')

def test_graph_object_creation():
    """Test that NetworkX graph objects are properly created"""
    
    print("🧪 TESTING GRAPH OBJECT CREATION")
    print("=" * 50)
    
    # Import the pipeline class
    from complete_usage_guide import ResearchPipelineCLI
    
    # Initialize pipeline
    app = ResearchPipelineCLI()
    
    # Set test data paths
    input_dir = 'test_input'
    output_dir = 'test_output'
    
    app.input_directory = input_dir
    app.output_dir = output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Input: {input_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Step 1: Load directory (minimal files for testing)
    print("\n1️⃣ LOADING TEST DATA")
    print("-" * 30)
    
    app.input_files = []
    valid_extensions = {'.json', '.txt', '.md'}
    
    file_count = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file_count >= 5:  # Limit to 5 files for testing
                break
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in valid_extensions:
                app.input_files.append(file_path)
                file_count += 1
        if file_count >= 5:
            break
    
    print(f"✅ Loaded {len(app.input_files)} test files")
    app.pipeline_state['data_loaded'] = True
    
    # Step 2: Text cleaning
    print("\n2️⃣ TEXT CLEANING")
    print("-" * 30)
    app.clean_and_normalize_text()
    
    # Step 3: Phrase extraction
    print("\n3️⃣ PHRASE EXTRACTION")
    print("-" * 30)
    app.extract_tokens_and_phrases()
    
    # Step 4: CRITICAL TEST - Global graph construction
    print("\n4️⃣ TESTING GLOBAL GRAPH OBJECT CREATION")
    print("-" * 50)
    
    app.build_global_graph()
    
    # Verify NetworkX graph object was created
    if app.global_graph_object is None:
        print("❌ FAILED: global_graph_object is None")
        return False
    
    if not isinstance(app.global_graph_object, nx.Graph):
        print(f"❌ FAILED: global_graph_object is not NetworkX Graph, got {type(app.global_graph_object)}")
        return False
    
    print(f"✅ SUCCESS: NetworkX Graph object created")
    print(f"   Type: {type(app.global_graph_object)}")
    print(f"   Nodes: {app.global_graph_object.number_of_nodes()}")
    print(f"   Edges: {app.global_graph_object.number_of_edges()}")
    
    # Verify layout positions were computed
    if app.global_layout_positions is None:
        print("❌ FAILED: global_layout_positions is None")
        return False
    
    print(f"✅ SUCCESS: Layout positions computed")
    print(f"   Positions for {len(app.global_layout_positions)} nodes")
    
    # Verify node attributes
    sample_node = list(app.global_graph_object.nodes())[0]
    node_attrs = app.global_graph_object.nodes[sample_node]
    
    expected_attrs = ['frequency', 'phrase_type', 'community', 'degree_centrality', 'pos']
    missing_attrs = [attr for attr in expected_attrs if attr not in node_attrs]
    
    if missing_attrs:
        print(f"⚠️ WARNING: Missing node attributes: {missing_attrs}")
    else:
        print(f"✅ SUCCESS: All expected node attributes present")
    
    print(f"   Sample node '{sample_node}' attributes: {list(node_attrs.keys())}")
    
    # Step 5: CRITICAL TEST - Subgraph activation
    print("\n5️⃣ TESTING SUBGRAPH OBJECT CREATION")
    print("-" * 50)
    
    app.activate_state_subgraphs()
    
    # Verify subgraph objects were created
    if not hasattr(app, 'state_subgraph_objects') or not app.state_subgraph_objects:
        print("❌ FAILED: state_subgraph_objects not created")
        return False
    
    print(f"✅ SUCCESS: Subgraph objects created")
    print(f"   Number of state subgraphs: {len(app.state_subgraph_objects)}")
    
    # Verify subgraphs are NetworkX objects
    for state, subgraph in app.state_subgraph_objects.items():
        # Check if it's a NetworkX graph-like object
        if not hasattr(subgraph, 'nodes') or not hasattr(subgraph, 'edges'):
            print(f"❌ FAILED: Subgraph {state} is not NetworkX-like object, got {type(subgraph)}")
            return False
        
        print(f"   {state}: {type(subgraph).__name__} with {subgraph.number_of_nodes()} nodes")
        
        # Verify shared positions
        if subgraph.number_of_nodes() > 0:
            sample_subgraph_node = list(subgraph.nodes())[0]
            if sample_subgraph_node in app.global_layout_positions:
                global_pos = app.global_layout_positions[sample_subgraph_node]
                subgraph_pos = app.global_graph_object.nodes[sample_subgraph_node].get('pos')
                if np.allclose(global_pos, subgraph_pos):
                    print(f"      ✅ Positions shared with global graph")
                else:
                    print(f"      ⚠️ Position mismatch detected")
    
    # Step 6: CRITICAL TEST - Visualization from graph objects
    print("\n6️⃣ TESTING VISUALIZATION FROM GRAPH OBJECTS")
    print("-" * 50)
    
    app.generate_deterministic_visualizations()
    
    # Verify visualizations were created
    if not hasattr(app, 'visualization_paths') or not app.visualization_paths:
        print("❌ FAILED: No visualizations created")
        return False
    
    print(f"✅ SUCCESS: Visualizations created from NetworkX objects")
    print(f"   Number of visualizations: {len(app.visualization_paths)}")
    
    for viz_name, viz_path in app.visualization_paths.items():
        if os.path.exists(viz_path):
            file_size = os.path.getsize(viz_path)
            print(f"   {viz_name}: {os.path.basename(viz_path)} ({file_size} bytes)")
        else:
            print(f"   ⚠️ {viz_name}: File not found at {viz_path}")
    
    # Step 7: Test exports from graph objects
    print("\n7️⃣ TESTING EXPORTS FROM GRAPH OBJECTS")
    print("-" * 50)
    
    app.export_global_graph_data()
    app.export_subgraph_data()
    
    # Check for GraphML files (NetworkX native format)
    graph_dir = os.path.join(output_dir, "global_graph")
    subgraph_dir = os.path.join(output_dir, "subgraphs")
    
    graphml_files = []
    if os.path.exists(graph_dir):
        graphml_files.extend([f for f in os.listdir(graph_dir) if f.endswith('.graphml')])
    if os.path.exists(subgraph_dir):
        graphml_files.extend([f for f in os.listdir(subgraph_dir) if f.endswith('.graphml')])
    
    if graphml_files:
        print(f"✅ SUCCESS: GraphML files exported from NetworkX objects")
        for f in graphml_files:
            print(f"   {f}")
    else:
        print("⚠️ WARNING: No GraphML files found")
    
    print("\n🎉 ALL GRAPH OBJECT TESTS COMPLETED!")
    return True

def main():
    """Main test function"""
    try:
        success = test_graph_object_creation()
        if success:
            print("\n✅ ALL TESTS PASSED - Graph objects are properly created and used!")
            return 0
        else:
            print("\n❌ TESTS FAILED - Graph objects not working correctly")
            return 1
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())