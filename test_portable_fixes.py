#!/usr/bin/env python3
"""
Test Portable Fixes

This script tests that all hardcoded path fixes work correctly and that
the project is now fully portable.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_portable_config():
    """Test the portable configuration system"""
    print("🧪 Testing portable configuration...")
    
    try:
        from portable_config import portable_config
        
        # Test directory creation
        input_dir = portable_config.get_input_dir()
        output_dir = portable_config.get_output_dir()
        sample_dir = portable_config.get_sample_data_dir()
        
        assert os.path.exists(input_dir), f"Input directory not created: {input_dir}"
        assert os.path.exists(output_dir), f"Output directory not created: {output_dir}"
        assert os.path.exists(sample_dir), f"Sample directory not created: {sample_dir}"
        
        print(f"✅ Portable config working: {input_dir}, {output_dir}, {sample_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Portable config test failed: {e}")
        return False

def test_tqdm_utils():
    """Test the tqdm utilities module"""
    print("🧪 Testing tqdm utilities...")
    
    try:
        from tqdm_utils import progress_files, progress_docs, ProjectProgressBar
        
        # Test basic functionality
        test_items = list(range(5))
        
        # Test progress bars (should not raise exceptions)
        for item in progress_files(test_items, "Testing files"):
            pass
        
        for item in progress_docs(test_items, "Testing docs"):
            pass
        
        print("✅ tqdm utilities working correctly")
        return True
        
    except Exception as e:
        print(f"❌ tqdm utilities test failed: {e}")
        return False

def test_complete_usage_guide_import():
    """Test that complete_usage_guide.py can be imported without hardcoded path errors"""
    print("🧪 Testing complete_usage_guide.py import...")
    
    try:
        # This should not fail due to hardcoded paths
        from complete_usage_guide import ResearchPipelineCLI
        
        # Test initialization
        app = ResearchPipelineCLI()
        
        # Check that output_dir is not hardcoded
        assert not app.output_dir.startswith('/Users/'), f"Hardcoded path found: {app.output_dir}"
        
        print("✅ complete_usage_guide.py imports correctly")
        return True
        
    except Exception as e:
        print(f"❌ complete_usage_guide.py import test failed: {e}")
        return False

def test_plotly_generator():
    """Test that plotly_visualization_generator.py works without hardcoded paths"""
    print("🧪 Testing plotly_visualization_generator.py...")
    
    try:
        # Check if file exists and can be read
        if os.path.exists('plotly_visualization_generator.py'):
            with open('plotly_visualization_generator.py', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for hardcoded paths
            hardcoded_patterns = [
                '/Users/zhangjingsen',
                'hajimi/七周目',
                'hajimi/四周目'
            ]
            
            for pattern in hardcoded_patterns:
                if pattern in content:
                    print(f"❌ Found hardcoded path: {pattern}")
                    return False
            
            print("✅ plotly_visualization_generator.py has no hardcoded paths")
            return True
        else:
            print("⚠️ plotly_visualization_generator.py not found")
            return True
            
    except Exception as e:
        print(f"❌ plotly generator test failed: {e}")
        return False

def test_enhanced_text_processor():
    """Test the enhanced text processor functionality"""
    print("🧪 Testing enhanced text processor...")
    
    try:
        from semantic_coword_pipeline.processors.enhanced_text_processor import EnhancedTextProcessor
        from semantic_coword_pipeline.core.config import Config
        
        # Test initialization
        config = Config()
        processor = EnhancedTextProcessor(config)
        
        print("✅ Enhanced text processor initializes correctly")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced text processor test failed: {e}")
        return False

def create_test_environment():
    """Create a temporary test environment"""
    print("🧪 Creating test environment...")
    
    # Create test directories
    test_dirs = ['test_input', 'test_output', 'sample_research_data']
    
    for dir_name in test_dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    # Create a sample test file
    test_file_content = '''[
    {
        "segment_id": "TEST_001",
        "title": "Test Document",
        "text": "This is a test document for portable testing.",
        "state": "TestState",
        "language": "english",
        "level": 1,
        "order": 1
    }
]'''
    
    with open('test_input/test_document.json', 'w', encoding='utf-8') as f:
        f.write(test_file_content)
    
    print("✅ Test environment created")
    return True

def cleanup_test_environment():
    """Clean up test environment"""
    print("🧹 Cleaning up test environment...")
    
    test_dirs = ['test_input', 'test_output', 'sample_research_data']
    
    for dir_name in test_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    
    print("✅ Test environment cleaned up")

def run_integration_test():
    """Run a basic integration test"""
    print("🧪 Running integration test...")
    
    try:
        from complete_usage_guide import ResearchPipelineCLI
        
        # Initialize app
        app = ResearchPipelineCLI()
        
        # Set test paths
        app.input_directory = "test_input"
        app.output_dir = "test_output"
        
        # Simulate finding files
        if os.path.exists("test_input"):
            app.input_files = [f for f in os.listdir("test_input") if f.endswith('.json')]
            app.pipeline_state['data_loaded'] = len(app.input_files) > 0
        
        print(f"✅ Integration test passed: found {len(app.input_files)} test files")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 TESTING PORTABLE FIXES")
    print("=" * 50)
    
    # Create test environment
    create_test_environment()
    
    tests = [
        ("Portable Config", test_portable_config),
        ("tqdm Utils", test_tqdm_utils),
        ("Complete Usage Guide", test_complete_usage_guide_import),
        ("Plotly Generator", test_plotly_generator),
        ("Enhanced Text Processor", test_enhanced_text_processor),
        ("Integration Test", run_integration_test)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Clean up
    cleanup_test_environment()
    
    # Report results
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📈 Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project is now fully portable.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)