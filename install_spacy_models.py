#!/usr/bin/env python3
"""
Install spaCy models for enhanced text processing

This script downloads and installs the required spaCy language models
for English and Chinese text processing.
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_spacy_model(model_name: str) -> bool:
    """Install a spaCy model"""
    try:
        logger.info(f"Installing spaCy model: {model_name}")
        result = subprocess.run([
            sys.executable, "-m", "spacy", "download", model_name
        ], capture_output=True, text=True, check=True)
        
        logger.info(f"Successfully installed {model_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {model_name}: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error installing {model_name}: {e}")
        return False

def verify_spacy_installation():
    """Verify spaCy and models are installed correctly"""
    try:
        import spacy
        logger.info(f"spaCy version: {spacy.__version__}")
        
        # Test English model
        try:
            nlp_en = spacy.load("en_core_web_sm")
            doc = nlp_en("This is a test sentence.")
            logger.info("English model (en_core_web_sm) loaded successfully")
        except OSError:
            logger.warning("English model (en_core_web_sm) not found")
        
        # Test Chinese model
        try:
            nlp_zh = spacy.load("zh_core_web_sm")
            doc = nlp_zh("这是一个测试句子。")
            logger.info("Chinese model (zh_core_web_sm) loaded successfully")
        except OSError:
            logger.warning("Chinese model (zh_core_web_sm) not found")
            
    except ImportError:
        logger.error("spaCy not installed. Install with: pip install spacy")

def main():
    """Main installation function"""
    print("spaCy Model Installation for Enhanced Text Processing")
    print("=" * 60)
    
    # Check if spaCy is installed
    try:
        import spacy
        print(f"✓ spaCy is installed (version {spacy.__version__})")
    except ImportError:
        print("✗ spaCy is not installed")
        print("Please install spaCy first: pip install spacy>=3.4.0")
        return
    
    # Install models
    models_to_install = [
        "en_core_web_sm",  # English
        "zh_core_web_sm"   # Chinese
    ]
    
    success_count = 0
    for model in models_to_install:
        if install_spacy_model(model):
            success_count += 1
    
    print(f"\nInstallation Summary:")
    print(f"Successfully installed: {success_count}/{len(models_to_install)} models")
    
    # Verify installation
    print("\nVerifying installation...")
    verify_spacy_installation()
    
    if success_count == len(models_to_install):
        print("\n✓ All models installed successfully!")
        print("You can now use the enhanced text processor with full linguistic analysis.")
    else:
        print(f"\n⚠ Only {success_count} models installed successfully.")
        print("The enhanced text processor will fall back to basic processing for missing models.")

if __name__ == "__main__":
    main()