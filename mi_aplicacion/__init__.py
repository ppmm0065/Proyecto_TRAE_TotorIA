# mi_aplicacion/__init__.py
import os
import datetime
import sys
from flask import Flask
import logging
# INICIO: NUEVAS IMPORTACIONES
from flask_session import Session # Importar la extensión
from werkzeug.utils import secure_filename # Necesario para la ruta de sesión
# FIN: NUEVAS IMPORTACIONES

# Importar las funciones de inicialización y los componentes RAG desde app_logic
from .app_logic import (
    init_sqlite_db,
    initialize_rag_components
)

def create_app(config_name='dev'):
    """
    Application factory para crear y configurar la instancia de la aplicación Flask.
    """
    app = Flask(__name__, instance_relative_config=True) # Cambiado a True para usar la carpeta 'instance'

    # Cargar la configuración.
    try:
        project_root = os.path.abspath(os.path.join(app.root_path, os.pardir))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        import config 
        app.config.from_object(config.config_by_name[config_name])
        print(f"Configuración '{config_name}' cargada desde config.py.")
    except (ImportError, KeyError) as e:
        print(f"Error cargando config.py ('{e}'). Usando defaults.")
        app.config.from_mapping(
            SECRET_KEY='dev_secret_key_fallback',
            DEBUG=True
        )

    # --- INICIO: CONFIGURACIÓN DE SESIONES DEL LADO DEL SERVIDOR ---
    # Asegura que la carpeta 'instance' exista
    os.makedirs(app.instance_path, exist_ok=True)
    
    # Define la carpeta donde se guardarán los archivos de sesión
    session_file_dir = os.path.join(app.instance_path, 'flask_session')
    os.makedirs(session_file_dir, exist_ok=True)
    
    # Configura Flask para usar Flask-Session
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_FILE_DIR"] = session_file_dir
    app.config["SESSION_PERMANENT"] = False
    app.config["SESSION_USE_SIGNER"] = True
    
    # Inicializa la extensión
    Session(app)
    # --- FIN: CONFIGURACIÓN DE SESIONES ---

    app.secret_key = app.config['SECRET_KEY']

    project_root_for_paths = os.path.abspath(os.path.join(app.root_path, os.pardir))

    app.config['UPLOAD_FOLDER'] = os.path.join(project_root_for_paths, app.config.get('UPLOAD_FOLDER', 'uploads_fallback'))
    app.config['CONTEXT_DOCS_FOLDER'] = os.path.join(project_root_for_paths, app.config.get('CONTEXT_DOCS_FOLDER', 'context_docs_fallback'))
    app.config['DATABASE_FILE'] = os.path.join(project_root_for_paths, app.config.get('DATABASE_FILE', 'seguimiento_fallback.db'))
    app.config['FAISS_INDEX_PATH'] = os.path.join(project_root_for_paths, app.config.get('FAISS_INDEX_PATH', 'faiss_index_multi_fallback'))
    app.config['FAISS_FOLLOWUP_INDEX_PATH'] = os.path.join(project_root_for_paths, app.config.get('FAISS_FOLLOWUP_INDEX_PATH', 'faiss_index_followups_fallback'))
    app.config['MODEL_ARTIFACTS_DIR'] = os.path.join(app.instance_path, app.config.get('MODEL_ARTIFACTS_SUBDIR', 'model_artifacts'))

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['CONTEXT_DOCS_FOLDER'], exist_ok=True)
    os.makedirs(os.path.dirname(app.config['FAISS_INDEX_PATH']), exist_ok=True)
    os.makedirs(os.path.dirname(app.config['FAISS_FOLLOWUP_INDEX_PATH']), exist_ok=True)
    os.makedirs(app.config['MODEL_ARTIFACTS_DIR'], exist_ok=True)
    
    print(f"UPLOAD_FOLDER configurado en: {app.config['UPLOAD_FOLDER']}")
    print(f"CONTEXT_DOCS_FOLDER configurado en: {app.config['CONTEXT_DOCS_FOLDER']}")
    print(f"MODEL_ARTIFACTS_DIR configurado en: {app.config['MODEL_ARTIFACTS_DIR']}")

    # Configurar logging según configuración
    level_name = str(app.config.get('LOG_LEVEL', 'INFO')).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s'
    )
    app.logger.setLevel(level)
    app.logger.info(f"Nivel de logging establecido en {level_name}")

    init_sqlite_db(app.config['DATABASE_FILE'])

    with app.app_context():
        initialize_rag_components(app.config) 

    with app.app_context():
        from . import routes
        app.register_blueprint(routes.main_bp)
        # Registrar filtros Jinja personalizados
        try:
            from .filters import nota_un_decimal, ordenar_niveles_educativos, fix_course_characters
            app.jinja_env.filters['nota_un_decimal'] = nota_un_decimal
            app.jinja_env.filters['ordenar_niveles_educativos'] = ordenar_niveles_educativos
            app.jinja_env.filters['fix_course_characters'] = fix_course_characters
        except Exception as e:
            app.logger.warning(f"No se pudieron registrar los filtros Jinja personalizados: {e}")

    @app.context_processor
    def inject_current_year():
        return {'current_year': datetime.datetime.now().year}

    app.logger.info("Aplicación Flask creada y configurada con sesiones del lado del servidor.")
    return app
