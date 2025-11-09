#!/usr/bin/env python3
"""
Simple test to isolate the RAG initialization issue
"""
import sys
import os
import traceback

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing simple Flask app creation without RAG...")
    
    # Try to import just the basic Flask setup
    from flask import Flask
    
    # Create a minimal Flask app
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test_key'
    print("✓ Basic Flask app created successfully")
    
    # Now test the config import
    print("Testing config import...")
    import config
    print("✓ Config imported successfully")
    
    # Test database initialization
    print("Testing database initialization...")
    from mi_aplicacion.app_logic import init_sqlite_db
    test_db_path = "test.db"
    init_sqlite_db(test_db_path)
    print("✓ Database initialization successful")
    
    # Clean up test database
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    print("\nAll basic tests passed!")
    
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)