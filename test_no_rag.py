#!/usr/bin/env python3
"""
Test Flask app creation without RAG initialization
"""
import sys
import os
import traceback

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Creating Flask app without RAG initialization...")
    
    # Import Flask and create basic app
    from flask import Flask
    app = Flask(__name__)
    
    # Load config manually
    import config
    app.config.from_object(config.config_by_name['dev'])
    app.secret_key = app.config['SECRET_KEY']
    print("✓ Config loaded")
    
    # Set up basic paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    app.config['UPLOAD_FOLDER'] = os.path.join(project_root, app.config.get('UPLOAD_FOLDER', 'uploads'))
    app.config['DATABASE_FILE'] = os.path.join(project_root, app.config.get('DATABASE_FILE', 'seguimiento.db'))
    
    # Create directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    print("✓ Directories created")
    
    # Initialize database only
    from mi_aplicacion.app_logic import init_sqlite_db
    init_sqlite_db(app.config['DATABASE_FILE'])
    print("✓ Database initialized")
    
    # Skip RAG initialization
    print("✓ Skipping RAG initialization")
    
    # Register routes manually
    with app.app_context():
        from mi_aplicacion import routes
        app.register_blueprint(routes.main_bp)
    print("✓ Routes registered")
    
    print("\nFlask app created successfully without RAG!")
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