#!/usr/bin/env python3
"""
Test app_logic import step by step
"""
import sys
import os
import traceback

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing app_logic import step by step...")
    
    # Test importing the module itself
    print("Step 1: Importing app_logic module...")
    import mi_aplicacion.app_logic
    print("✓ app_logic module imported")
    
    # Test accessing specific attributes
    print("Step 2: Testing attribute access...")
    print(f"embedding_model_instance: {mi_aplicacion.app_logic.embedding_model_instance}")
    print(f"vector_store: {mi_aplicacion.app_logic.vector_store}")
    print(f"vector_store_followups: {mi_aplicacion.app_logic.vector_store_followups}")
    
    # Test importing specific functions
    print("Step 3: Testing function imports...")
    from mi_aplicacion.app_logic import get_dataframe_from_session_file
    print("✓ get_dataframe_from_session_file imported")
    
    print("\nAll app_logic import tests passed!")
    
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)