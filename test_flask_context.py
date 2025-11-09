#!/usr/bin/env python3
"""
Test Flask app with proper context
"""
import os
import sys

# Set environment variable to disable RAG
os.environ['ENABLE_RAG'] = 'false'

try:
    print("Creating Flask app with proper context...")
    
    # Import and create app
    from mi_aplicacion import create_app
    app = create_app('dev')
    print("✓ App created successfully")
    
    # Create app context
    with app.app_context():
        print("✓ App context created")
        
        # Now try to import routes
        print("Importing routes...")
        from mi_aplicacion import routes
        print("✓ Routes imported successfully")
        
        # Test that we can access config
        print(f"✓ App config loaded: {app.config.get('SECRET_KEY', 'No secret key')}")
        
        # Test a function that uses current_app.config
        print("Testing app_logic function...")
        from mi_aplicacion.app_logic import format_chat_history_for_prompt
        result = format_chat_history_for_prompt([])
        print(f"✓ App logic function works: {result}")
    
    print("\nAll Flask context tests passed!")
    print("The app should be working now!")
    
    # Try to run the server
    print("\nStarting Flask development server...")
    print("The server should be accessible at http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)