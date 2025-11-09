#!/usr/bin/env python3
"""
Test app_logic import to see where the hang occurs
"""
import sys
import os
import traceback

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing app_logic import...")
    
    # Test basic imports first
    print("Testing basic imports...")
    import pandas as pd
    print("✓ pandas imported")
    
    import numpy as np
    print("✓ numpy imported")
    
    import google.generativeai as genai
    print("✓ google.generativeai imported")
    
    from flask import session, flash, current_app
    print("✓ flask imports imported")
    
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("✓ langchain text_splitter imported")
    
    from langchain_community.vectorstores import FAISS
    print("✓ langchain FAISS imported")
    
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    print("✓ langchain document_loaders imported")
    
    from langchain_core.documents import Document
    print("✓ langchain documents imported")
    
    import sqlite3
    print("✓ sqlite3 imported")
    
    import pytz
    from pytz import timezone
    print("✓ pytz imported")
    
    # Now test the problematic import
    print("\nTesting SentenceTransformerEmbeddings import...")
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    print("✓ SentenceTransformerEmbeddings imported")
    
    print("\nAll imports successful! Now testing app_logic module import...")
    from mi_aplicacion import app_logic
    print("✓ app_logic module imported")
    
    print("\nAll app_logic tests passed!")
    
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)