#!/usr/bin/env python3
"""
Fix Hardcoded Paths Script

This script systematically fixes all hardcoded paths in the project to make it
portable and remove local path dependencies.
"""

import os
import re
import glob
from pathlib import Path
from typing import List, Dict, Tuple

def find_hardcoded_paths(file_path: str) -> List[Tuple[int, str, str]]:
    """Find hardcoded paths in a file"""
    hardcoded_patterns = [
        r'/Users/zhangjingsen/Desktop/python/graph4socialscience/[^"\s\']*',
        r'hajimi/[^"\s\']*',
        r'七周目',
        r'四周目',
        r'haniumoa',
        r'nan/',
        r'to/'
    ]
    
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            for pattern in hardcoded_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    issues.append((line_num, line.strip(), match.group()))
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    
    return issues

def fix_test_files():
    """Fix hardcoded paths in test files"""
    test_files = [
        'comprehensive_pipeline_test.py',
        'test_semantic_corrections.py', 
        'test_scientific_optimization.py',
        'test_restored_visualization.py',
        'test_new_feature_6_4.py',
        'test_graph_objects.py',
        'test_complete_usage_guide.py',
        'test_6_4_simplified_output.py',
        'simple_visualization_test.py',
        'quick_pipeline_setup.py',
        'test_6_4_with_toc_doc.py',
        'test_visualization_fix.py',
        'isolated_visualization_test.py',
        'test_plotly_integration.py'
    ]
    
    fixes = {}
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"Fixing {test_file}...")
            
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace hardcoded paths with relative or configurable paths
            original_content = content
            
            # Replace specific hardcoded paths
            content = re.sub(
                r'/Users/zhangjingsen/Desktop/python/graph4socialscience/toc_doc',
                'test_input',
                content
            )
            
            content = re.sub(
                r'/Users/zhangjingsen/Desktop/python/graph4socialscience/semantic-node-refinement-test/data/raw',
                'sample_research_data',
                content
            )
            
            content = re.sub(
                r'/Users/zhangjingsen/Desktop/python/graph4socialscience/hajimi/[^"\']*',
                'test_output',
                content
            )
            
            content = re.sub(
                r'/Users/zhangjingsen/Desktop/python/graph4socialscience/[^"\']*output[^"\']*',
                'test_output',
                content
            )
            
            if content != original_content:
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixes[test_file] = "Fixed hardcoded paths"
    
    return fixes

def fix_plotly_visualization_generator():
    """Fix hardcoded paths in plotly_visualization_generator.py"""
    file_path = 'plotly_visualization_generator.py'
    
    if not os.path.exists(file_path):
        return {}
    
    print(f"Fixing {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Replace hardcoded test paths
    content = re.sub(
        r'output_dir = "/Users/zhangjingsen/Desktop/python/graph4socialscience/hajimi/七周目"',
        'output_dir = "test_output"',
        content
    )
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {file_path: "Fixed hardcoded paths"}
    
    return {}

def add_tqdm_progress_bars():
    """Add tqdm progress bars to functions that need them"""
    
    # Files that need tqdm improvements
    files_to_enhance = [
        'complete_usage_guide.py',
        'plotly_visualization_generator.py',
        'semantic_coword_pipeline/processors/enhanced_text_processor.py'
    ]
    
    fixes = {}
    
    for file_path in files_to_enhance:
        if os.path.exists(file_path):
            print(f"Checking tqdm usage in {file_path}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if tqdm is imported
            if 'from tqdm import tqdm' not in content and 'import tqdm' not in content:
                # Add tqdm import if missing
                if 'import os' in content:
                    content = content.replace('import os', 'import os\nfrom tqdm import tqdm')
                    fixes[file_path] = "Added tqdm import"
    
    return fixes

def create_portable_config():
    """Create a portable configuration system"""
    
    config_content = '''#!/usr/bin/env python3
"""
Portable Configuration for Graph4SocialScience

This configuration removes hardcoded paths and makes the project portable.
"""

import os
from pathlib import Path

class PortableConfig:
    """Portable configuration that adapts to any environment"""
    
    def __init__(self):
        # Get project root directory
        self.project_root = Path(__file__).parent.absolute()
        
        # Default directories (relative to project root)
        self.default_input_dir = self.project_root / "test_input"
        self.default_output_dir = self.project_root / "output"
        self.sample_data_dir = self.project_root / "sample_research_data"
        
        # Create directories if they don't exist
        self.default_input_dir.mkdir(exist_ok=True)
        self.default_output_dir.mkdir(exist_ok=True)
        self.sample_data_dir.mkdir(exist_ok=True)
    
    def get_input_dir(self, custom_path: str = None) -> str:
        """Get input directory path"""
        if custom_path and os.path.exists(custom_path):
            return custom_path
        return str(self.default_input_dir)
    
    def get_output_dir(self, custom_path: str = None) -> str:
        """Get output directory path"""
        if custom_path:
            return custom_path
        return str(self.default_output_dir)
    
    def get_sample_data_dir(self) -> str:
        """Get sample data directory path"""
        return str(self.sample_data_dir)

# Global configuration instance
portable_config = PortableConfig()
'''
    
    with open('portable_config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    return {'portable_config.py': 'Created portable configuration system'}

def scan_all_files():
    """Scan all Python files for hardcoded paths"""
    
    print("🔍 Scanning all Python files for hardcoded paths...")
    
    python_files = glob.glob("**/*.py", recursive=True)
    issues_found = {}
    
    for file_path in python_files:
        # Skip certain directories
        if any(skip in file_path for skip in ['.git', '__pycache__', '.pytest_cache', 'venv', 'env']):
            continue
        
        issues = find_hardcoded_paths(file_path)
        if issues:
            issues_found[file_path] = issues
    
    return issues_found

def main():
    """Main function to fix all hardcoded paths"""
    
    print("🔧 FIXING HARDCODED PATHS IN GRAPH4SOCIALSCIENCE PROJECT")
    print("=" * 70)
    
    all_fixes = {}
    
    # 1. Fix test files
    print("\n1️⃣ Fixing test files...")
    test_fixes = fix_test_files()
    all_fixes.update(test_fixes)
    
    # 2. Fix plotly visualization generator
    print("\n2️⃣ Fixing plotly visualization generator...")
    plotly_fixes = fix_plotly_visualization_generator()
    all_fixes.update(plotly_fixes)
    
    # 3. Add tqdm progress bars
    print("\n3️⃣ Checking tqdm usage...")
    tqdm_fixes = add_tqdm_progress_bars()
    all_fixes.update(tqdm_fixes)
    
    # 4. Create portable configuration
    print("\n4️⃣ Creating portable configuration...")
    config_fixes = create_portable_config()
    all_fixes.update(config_fixes)
    
    # 5. Scan for remaining issues
    print("\n5️⃣ Scanning for remaining hardcoded paths...")
    remaining_issues = scan_all_files()
    
    # Report results
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    
    if all_fixes:
        print(f"✅ Fixed {len(all_fixes)} files:")
        for file_path, description in all_fixes.items():
            print(f"   📄 {file_path}: {description}")
    else:
        print("ℹ️ No automatic fixes applied")
    
    if remaining_issues:
        print(f"\n⚠️ Found {len(remaining_issues)} files with potential hardcoded paths:")
        for file_path, issues in remaining_issues.items():
            print(f"\n📄 {file_path}:")
            for line_num, line, match in issues[:3]:  # Show first 3 issues
                print(f"   Line {line_num}: {match}")
            if len(issues) > 3:
                print(f"   ... and {len(issues) - 3} more issues")
    else:
        print("\n✅ No remaining hardcoded paths found!")
    
    # Recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("1. Use the new portable_config.py for path management")
    print("2. Test all functionality with the fixed paths")
    print("3. Update documentation to reflect portable setup")
    print("4. Consider using environment variables for custom paths")
    
    return len(all_fixes), len(remaining_issues)

if __name__ == "__main__":
    fixes_count, issues_count = main()
    
    if issues_count == 0:
        print("\n🎉 All hardcoded paths have been fixed!")
        exit(0)
    else:
        print(f"\n⚠️ {issues_count} files still have hardcoded paths that need manual review")
        exit(1)