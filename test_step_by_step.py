#!/usr/bin/env python3
"""
Step by step test to isolate the issue
"""
import os
import sys

# Set environment variable to disable RAG
os.environ['ENABLE_RAG'] = 'false'

print("Step 1: Testing basic imports...")
try:
    from mi_aplicacion import create_app
    print("✓ create_app imported successfully")
except Exception as e:
    print(f"✗ Failed to import create_app: {e}")
    sys.exit(1)

print("\nStep 2: Creating app...")
try:
    app = create_app('dev')
    print("✓ App created successfully")
    print(f"  - App name: {app.name}")
    print(f"  - Secret key exists: {'SECRET_KEY' in app.config}")
except Exception as e:
    print(f"✗ Failed to create app: {e}")
    sys.exit(1)

print("\nStep 3: Creating app context...")
try:
    with app.app_context():
        print("✓ App context created successfully")
        print(f"  - Current app: {app.name}")
        print(f"  - Config keys: {list(app.config.keys())[:5]}...")  # Show first 5 config keys
except Exception as e:
    print(f"✗ Failed to create app context: {e}")
    sys.exit(1)

print("\nStep 4: Testing app_logic import...")
try:
    from mi_aplicacion import app_logic
    print("✓ app_logic imported successfully")
except Exception as e:
    print(f"✗ Failed to import app_logic: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Testing routes import...")
try:
    from mi_aplicacion import routes
    print("✓ routes imported successfully")
except Exception as e:
    print(f"✗ Failed to import routes: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 6: Testing current_app.config access...")
try:
    with app.app_context():
        # Test accessing config through current_app
        nombre_col = app.config.get('NOMBRE_COL', 'default_name')
        print(f"✓ Config access works: NOMBRE_COL = {nombre_col}")
except Exception as e:
    print(f"✗ Failed to access config: {e}")
    sys.exit(1)

print("\n✅ All tests passed!")
print("The Flask application is working correctly.")
print("You can now run: python run.py")