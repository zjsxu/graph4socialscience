#!/usr/bin/env python3
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
