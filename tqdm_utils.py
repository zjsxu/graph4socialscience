#!/usr/bin/env python3
"""
tqdm Utilities for Graph4SocialScience

Provides consistent progress bar styling and utilities across the project.
"""

from tqdm import tqdm
from typing import Iterable, Any, Optional

class ProjectProgressBar:
    """Consistent progress bar styling for the project"""
    
    @staticmethod
    def create_bar(iterable: Iterable, 
                   desc: str, 
                   unit: str = "item",
                   total: Optional[int] = None,
                   **kwargs) -> tqdm:
        """Create a styled progress bar"""
        
        # Default styling
        default_kwargs = {
            'desc': desc,
            'unit': unit,
            'ncols': 80,
            'bar_format': '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        }
        
        # Override with user kwargs
        default_kwargs.update(kwargs)
        
        if total is not None:
            default_kwargs['total'] = total
        
        return tqdm(iterable, **default_kwargs)
    
    @staticmethod
    def file_processing(files: Iterable, desc: str = "📁 Processing files") -> tqdm:
        """Progress bar for file processing"""
        return ProjectProgressBar.create_bar(files, desc, unit="file")
    
    @staticmethod
    def document_processing(docs: Iterable, desc: str = "📄 Processing documents") -> tqdm:
        """Progress bar for document processing"""
        return ProjectProgressBar.create_bar(docs, desc, unit="doc")
    
    @staticmethod
    def phrase_processing(phrases: Iterable, desc: str = "🔍 Processing phrases") -> tqdm:
        """Progress bar for phrase processing"""
        return ProjectProgressBar.create_bar(phrases, desc, unit="phrase")
    
    @staticmethod
    def graph_processing(items: Iterable, desc: str = "🌐 Building graph") -> tqdm:
        """Progress bar for graph processing"""
        return ProjectProgressBar.create_bar(items, desc, unit="node")
    
    @staticmethod
    def visualization_processing(items: Iterable, desc: str = "🎨 Creating visualizations") -> tqdm:
        """Progress bar for visualization processing"""
        return ProjectProgressBar.create_bar(items, desc, unit="viz")

# Convenience functions
def progress_files(files: Iterable, desc: str = "📁 Processing files") -> tqdm:
    """Convenience function for file processing progress"""
    return ProjectProgressBar.file_processing(files, desc)

def progress_docs(docs: Iterable, desc: str = "📄 Processing documents") -> tqdm:
    """Convenience function for document processing progress"""
    return ProjectProgressBar.document_processing(docs, desc)

def progress_phrases(phrases: Iterable, desc: str = "🔍 Processing phrases") -> tqdm:
    """Convenience function for phrase processing progress"""
    return ProjectProgressBar.phrase_processing(phrases, desc)

def progress_graph(items: Iterable, desc: str = "🌐 Building graph") -> tqdm:
    """Convenience function for graph processing progress"""
    return ProjectProgressBar.graph_processing(items, desc)

def progress_viz(items: Iterable, desc: str = "🎨 Creating visualizations") -> tqdm:
    """Convenience function for visualization processing progress"""
    return ProjectProgressBar.visualization_processing(items, desc)
