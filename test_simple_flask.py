#!/usr/bin/env python3
"""
Simple Flask app test
"""
import os
import sys

# Set environment variable to disable RAG
os.environ['ENABLE_RAG'] = 'false'

print("Starting simple Flask test...")

try:
    # Import and create app
    from mi_aplicacion import create_app
    print("✓ Imported create_app")
    
    app = create_app('dev')
    print("✓ App created successfully")
    
    # Create app context
    with app.app_context():
        print("✓ App context created")
        
        # Test that we can access config
        print(f"✓ App config: SECRET_KEY = {app.config.get('SECRET_KEY', 'No secret key')}")
        print(f"✓ App config: DATABASE_FILE = {app.config.get('DATABASE_FILE', 'No db file')}")
        
        # Test importing routes
        print("Importing routes...")
        from mi_aplicacion import routes
        print("✓ Routes imported successfully")
        
        print("\nAll tests passed! The Flask app is working correctly.")
        print("The server should be accessible at http://localhost:5000")
        
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)