#!/usr/bin/env python3
"""
Test Flask app with debug version and RAG disabled
"""
import os
import sys
import traceback

# Set environment variable to disable RAG
os.environ['ENABLE_RAG'] = 'false'

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Creating Flask app with debug version and RAG disabled...")
    
    # Use the debug version
    from mi_aplicacion.__init___debug import create_app
    
    # Create the app
    app = create_app('dev')
    
    print("\nFlask app created successfully!")
    print(f"App name: {app.name}")
    print(f"Debug mode: {app.debug}")
    print(f"Database: {app.config['DATABASE_FILE']}")
    
    # Try to run the app
    print("\nStarting Flask development server...")
    print("The server should be accessible at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)