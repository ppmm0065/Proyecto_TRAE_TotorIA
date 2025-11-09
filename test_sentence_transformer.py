#!/usr/bin/env python3
"""
Test SentenceTransformerEmbeddings import specifically
"""
import sys
import os
import traceback

try:
    print("Testing SentenceTransformerEmbeddings import...")
    
    # This is the import that's hanging
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    print("✓ SentenceTransformerEmbeddings imported successfully")
    
    # Try to create an instance
    print("Testing SentenceTransformerEmbeddings instantiation...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✓ SentenceTransformerEmbeddings instantiated successfully")
    
    print("\nAll SentenceTransformer tests passed!")
    
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)