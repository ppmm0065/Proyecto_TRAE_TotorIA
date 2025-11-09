#!/usr/bin/env python3
"""
Test script to debug Flask application startup
"""
import sys
import os
import traceback

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Starting Flask application test...")
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Try to import the application
    print("Importing mi_aplicacion...")
    from mi_aplicacion import create_app
    print("✓ Successfully imported mi_aplicacion")
    
    # Try to create the app
    print("Creating Flask app...")
    app = create_app('dev')
    print(f"✓ Successfully created app")
    print(f"  - App name: {app.name}")
    print(f"  - Debug mode: {app.debug}")
    print(f"  - Secret key present: {bool(app.config.get('SECRET_KEY'))}")
    print(f"  - Upload folder: {app.config.get('UPLOAD_FOLDER')}")
    
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

print("\nTest completed successfully!")