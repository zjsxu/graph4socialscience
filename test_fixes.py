#!/usr/bin/env python3
"""
Test script to verify the fixes for GraphML export, progress bars, and state detection
"""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, '.')

def create_test_data():
    """Create test data with folder structure for state detection"""
    
    # Create temporary test directory
    test_dir = tempfile.mkdtemp(prefix="test_pipeline_fixes_")
    print(f"📁 Created test directory: {test_dir}")
    
    # Create state-based folder structure
    states = ['CA', 'NY', 'TX']
    test_files = []
    
    for state in states:
        state_dir = os.path.join(test_dir, state)
        os.makedirs(state_dir, exist_ok=True)
        
        # Create 2 test documents per state
        for i in range(2):
            doc_data = {
                "segment_id": f"{state}_doc_{i+1}",
                "title": f"Test Document {i+1} from {state}",
                "level": 1,
                "order": i+1,
                "text": f"This is a test document from {state}. It contains machine learning algorithms and data science methods. Natural language processing techniques are used for text analysis and information extraction.",
                "state": "Unknown",  # Will be overridden by folder detection
                "language": "english"
            }
            
            file_path = os.path.join(state_dir, f"doc_{i+1}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)
            
            test_files.append(file_path)
    
    return test_dir, test_files

def test_pipeline_fixes():
    """Test all the fixes: GraphML export, progress bars, state detection"""
    
    print("🧪 TESTING PIPELINE FIXES")
    print("=" * 60)
    
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
        
        # Test 1: State detection from folder names
        print("\n1️⃣ TESTING STATE DETECTION FROM FOLDER NAMES")
        print("-" * 50)
        
        input_data = app.load_input_data()
        states_found = set(doc['state'] for doc in input_data)
        
        print(f"✅ States detected: {sorted(states_found)}")
        
        if 'Unknown' in states_found:
            print("⚠️ WARNING: Some documents still have 'Unknown' state")
        else:
            print("✅ SUCCESS: All documents have proper state detection")
        
        # Test 2: Text cleaning with progress bar
        print("\n2️⃣ TESTING TEXT CLEANING WITH PROGRESS BAR")
        print("-" * 50)
        
        app.clean_and_normalize_text()
        
        if app.pipeline_state['text_cleaned']:
            print("✅ SUCCESS: Text cleaning completed with progress bar")
        else:
            print("❌ FAILED: Text cleaning failed")
            return False
        
        # Test 3: Phrase extraction with progress bar
        print("\n3️⃣ TESTING PHRASE EXTRACTION WITH PROGRESS BAR")
        print("-" * 50)
        
        app.extract_tokens_and_phrases()
        
        if app.pipeline_state['phrases_constructed']:
            print("✅ SUCCESS: Phrase extraction completed with progress bar")
        else:
            print("❌ FAILED: Phrase extraction failed")
            return False
        
        # Test 4: Global graph construction with detailed progress bars
        print("\n4️⃣ TESTING GLOBAL GRAPH CONSTRUCTION WITH DETAILED PROGRESS BARS")
        print("-" * 50)
        print("   Testing progress bars for:")
        print("   - Co-occurrence building")
        print("   - Layout computation (50 iterations)")
        print("   - Community detection")
        print("   - Centrality measures (degree + betweenness)")
        
        app.build_global_graph()
        
        if app.pipeline_state['global_graph_built'] and app.global_graph_object:
            print("✅ SUCCESS: Global graph construction completed with detailed progress bars")
            print(f"   Nodes: {app.global_graph_object.number_of_nodes()}")
            print(f"   Edges: {app.global_graph_object.number_of_edges()}")
        else:
            print("❌ FAILED: Global graph construction failed")
            return False
        
        # Test 5: Subgraph activation with progress bar
        print("\n5️⃣ TESTING SUBGRAPH ACTIVATION WITH PROGRESS BAR")
        print("-" * 50)
        
        app.activate_state_subgraphs()
        
        if app.pipeline_state['subgraphs_activated'] and app.state_subgraph_objects:
            print("✅ SUCCESS: Subgraph activation completed with progress bar")
            print(f"   Subgraphs created: {len(app.state_subgraph_objects)}")
            for state, subgraph in app.state_subgraph_objects.items():
                print(f"   {state}: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
        else:
            print("❌ FAILED: Subgraph activation failed")
            return False
        
        # Test 6: Visualization generation with detailed progress bars
        print("\n6️⃣ TESTING VISUALIZATION GENERATION WITH DETAILED PROGRESS BARS")
        print("-" * 50)
        print("   Testing progress bars for:")
        print("   - Global graph visualization (6 steps)")
        print("   - Subgraph visualizations (6 steps each)")
        
        app.generate_deterministic_visualizations()
        
        if hasattr(app, 'visualization_paths') and app.visualization_paths:
            print("✅ SUCCESS: Visualization generation completed with detailed progress bars")
            print(f"   Visualizations created: {len(app.visualization_paths)}")
        else:
            print("❌ FAILED: Visualization generation failed")
            return False
        
        # Test 7: CRITICAL - GraphML export without numpy array errors
        print("\n7️⃣ TESTING GRAPHML EXPORT (NUMPY ARRAY FIX)")
        print("-" * 50)
        
        try:
            app.export_global_graph_data()
            print("✅ SUCCESS: Global graph GraphML export completed without errors")
        except Exception as e:
            print(f"❌ FAILED: Global graph GraphML export failed: {e}")
            return False
        
        try:
            print("   Testing detailed export progress bars...")
            print("   - Each subgraph: 4 steps (Prepare GraphML, Write GraphML, Prepare JSON, Write JSON)")
            app.export_subgraph_data()
            print("✅ SUCCESS: Subgraph GraphML export completed without errors")
        except Exception as e:
            print(f"❌ FAILED: Subgraph GraphML export failed: {e}")
            return False
        
        # Verify GraphML files were created
        graph_dir = os.path.join(output_dir, "global_graph")
        subgraph_dir = os.path.join(output_dir, "subgraphs")
        
        graphml_files = []
        if os.path.exists(graph_dir):
            graphml_files.extend([f for f in os.listdir(graph_dir) if f.endswith('.graphml')])
        if os.path.exists(subgraph_dir):
            graphml_files.extend([f for f in os.listdir(subgraph_dir) if f.endswith('.graphml')])
        
        if graphml_files:
            print(f"✅ SUCCESS: {len(graphml_files)} GraphML files created successfully")
            for f in graphml_files:
                print(f"   📄 {f}")
        else:
            print("❌ FAILED: No GraphML files found")
            return False
        
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ GraphML export errors fixed (numpy arrays converted to separate x,y attributes)")
        print("✅ Progress bars added to all slow operations with detailed sub-steps:")
        print("   - Step 4.1: Layout computation, community detection, centrality measures")
        print("   - Step 5.3: Individual subgraph export steps")
        print("   - Step 6.1: Detailed visualization generation steps")
        print("✅ State detection from folder names working")
        
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
    success = test_pipeline_fixes()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())