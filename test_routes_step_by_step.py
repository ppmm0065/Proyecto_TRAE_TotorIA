#!/usr/bin/env python3
"""
Test routes import step by step to find the hang
"""
import sys
import os
import traceback

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("Testing routes import step by step...")
    
    # Step 1: Test basic imports that routes.py does
    print("Step 1: Basic imports...")
    import os
    import pandas as pd
    from flask import Blueprint, render_template, request, redirect, url_for, session, flash, jsonify, current_app, make_response, Response
    from werkzeug.utils import secure_filename
    import datetime
    import traceback
    import markdown
    from urllib.parse import unquote, quote 
    import numpy as np 
    import pytz
    from pytz import timezone
    from langchain_community.vectorstores import FAISS 
    print("✓ Basic imports done")
    
    # Step 2: Test app_logic imports
    print("Step 2: Testing app_logic imports...")
    
    # Import only specific functions, not all
    print("Importing specific functions from app_logic...")
    from mi_aplicacion.app_logic import get_dataframe_from_session_file
    print("✓ get_dataframe_from_session_file imported")
    
    from mi_aplicacion.app_logic import load_data_as_string
    print("✓ load_data_as_string imported")
    
    from mi_aplicacion.app_logic import format_chat_history_for_prompt
    print("✓ format_chat_history_for_prompt imported")
    
    print("✓ App logic imports done")
    
    # Step 3: Test creating Blueprint
    print("Step 3: Creating Blueprint...")
    main_bp = Blueprint('main', __name__)
    print("✓ Blueprint created")
    
    print("\nAll step-by-step tests passed!")
    
except Exception as e:
    print(f"\n✗ Error occurred at step: {e}")
    print("Full traceback:")
    traceback.print_exc()
    sys.exit(1)