#!/usr/bin/env python3
"""
Test FAISS import specifically
"""
import sys
import os
import traceback
import time

try:
    print("Testing FAISS import...")
    
    # Test basic FAISS import
    print("Step 1: Testing basic FAISS import...")
    start_time = time.time()
    from langchain_community.vectorstores import FAISS
    end_time = time.time()
    print(f"✓ FAISS imported in {end_time - start_time:.2f} seconds")
    
    # Test if we can create a simple FAISS instance
    print("Step 2: Testing FAISS instantiation...")
    start_time = time.time()
    
    # Create some dummy data
    from langchain_core.documents import Document
    docs = [Document(page_content="test", metadata={"source": "test"})]
    
    # Try to create embeddings (this might be the issue)
    print("Step 3: Testing embeddings creation...")
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    print("✓ Embeddings created")
    
    # Try to create FAISS vector store
    print("Step 4: Testing FAISS vector store creation...")
    vector_store = FAISS.from_documents(docs, embeddings)
    print("✓ FAISS vector store created")
    
    print("\nAll FAISS tests passed!")
    
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)