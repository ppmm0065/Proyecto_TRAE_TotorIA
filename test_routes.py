#!/usr/bin/env python3
"""
Test routes import to see where the hang occurs
"""
import sys
import os
import traceback

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing routes import...")
    
    # Test basic Flask import
    from flask import Flask
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test'
    print("✓ Flask imported and app created")
    
    # Test importing routes
    print("Importing routes module...")
    from mi_aplicacion import routes
    print("✓ Routes module imported")
    
    # Test registering blueprint
    print("Registering blueprint...")
    with app.app_context():
        app.register_blueprint(routes.main_bp)
    print("✓ Blueprint registered")
    
    print("\nAll route tests passed!")
    
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)