#!/usr/bin/env python3
"""
Test minimal app_logic import to find the issue
"""
import sys
import os
import traceback

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing minimal app_logic import...")
    
    # Test importing just the basic modules that app_logic needs
    print("Step 1: Testing basic imports...")
    import os
    import pandas as pd
    import numpy as np
    import traceback
    import sqlite3
    import io
    import re
    import csv
    import datetime
    import shutil
    print("✓ Basic modules imported")
    
    # Test Flask imports (this might be the issue)
    print("Step 2: Testing Flask imports...")
    try:
        from flask import session, flash, current_app
        print("✓ Flask imports successful")
    except Exception as e:
        print(f"⚠ Flask imports failed: {e}")
        print("This might be expected if Flask is not initialized")
    
    # Test LangChain imports
    print("Step 3: Testing LangChain imports...")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("✓ LangChain text_splitter imported")
    
    from langchain_community.vectorstores import FAISS
    print("✓ LangChain FAISS imported")
    
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    print("✓ LangChain document_loaders imported")
    
    from langchain_core.documents import Document
    print("✓ LangChain documents imported")
    
    # Test the problematic import
    print("Step 4: Testing SentenceTransformerEmbeddings import...")
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    print("✓ SentenceTransformerEmbeddings imported")
    
    # Test pytz
    print("Step 5: Testing pytz...")
    import pytz
    from pytz import timezone
    SANTIAGO_TZ = timezone('America/Santiago')
    print("✓ pytz imported and timezone set")
    
    print("\nAll minimal imports successful!")
    print("The issue might be with Flask current_app access in module-level code")
    
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)