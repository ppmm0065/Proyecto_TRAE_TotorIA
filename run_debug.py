#!/usr/bin/env python3
"""
Script para ejecutar la aplicación con mejor manejo de errores y diagnóstico
"""

import sys
import os
import traceback
import logging

# Configurar logging para ver más detalles
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("=== INICIANDO APLICACIÓN FLASK ===")
    
    # Intentar importar la aplicación
    print("Importando mi_aplicacion...")
    from mi_aplicacion import create_app
    print("+ Importación exitosa")
    
    # Crear la aplicación
    print("Creando aplicación...")
    config_name = os.getenv('FLASK_CONFIG', 'dev')
    app = create_app(config_name)
    print(f"+ Aplicación creada con config '{config_name}'")
    
    # Verificar que la aplicación esté bien configurada
    print(f"DEBUG: {app.config.get('DEBUG', 'No encontrado')}")
    print(f"SECRET_KEY existe: {'Sí' if app.config.get('SECRET_KEY') else 'No'}")
    
    # Intentar ejecutar la aplicación
    print("\n=== INICIANDO SERVIDOR FLASK ===")
    print("Servidor iniciándose en http://127.0.0.1:5000")
    print("Presiona Ctrl+C para detener")
    
    app.run(host='127.0.0.1', port=5000, debug=True)
    
except KeyboardInterrupt:
    print("\n\n=== SERVIDOR DETENIDO POR EL USUARIO ===")
    sys.exit(0)
    
except Exception as e:
    print(f"\n\n=== ERROR CRÍTICO ===")
    print(f"Error: {type(e).__name__}: {e}")
    print("\nTraceback completo:")
    traceback.print_exc()
    print("\n=== SUGERENCIAS ===")
    print("1. Verifica que todas las dependencias estén instaladas")
    print("2. Revisa la configuración en config.py")
    print("3. Asegúrate de que los directorios necesarios existan")
    print("4. Verifica los logs anteriores para más detalles")
    sys.exit(1)