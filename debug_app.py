#!/usr/bin/env python3
"""
Script de diagnóstico para identificar problemas en la aplicación Flask
"""

import sys
import os
import traceback

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("=== INICIANDO DIAGNÓSTICO DE LA APLICACIÓN ===")
    
    # 1. Intentar importar la aplicación
    print("1. Intentando importar mi_aplicacion...")
    from mi_aplicacion import create_app
    print("✓ Importación exitosa")
    
    # 2. Crear la aplicación
    print("2. Creando aplicación Flask...")
    app = create_app('dev')
    print(f"✓ Aplicación creada: {type(app)}")
    
    # 3. Verificar configuración básica
    print("3. Verificando configuración...")
    print(f"   - DEBUG: {app.config.get('DEBUG', 'No encontrado')}")
    print(f"   - SECRET_KEY: {'Sí' if app.config.get('SECRET_KEY') else 'No'} existe")
    print(f"   - UPLOAD_FOLDER: {app.config.get('UPLOAD_FOLDER', 'No encontrado')}")
    
    # 4. Verificar blueprints
    print("4. Verificando blueprints...")
    blueprints = list(app.blueprints.keys())
    print(f"   - Blueprints registrados: {blueprints}")
    
    # 5. Verificar rutas
    print("5. Verificando rutas...")
    routes = list(app.url_map.iter_rules())
    print(f"   - Rutas registradas: {len(routes)}")
    for route in routes[:5]:  # Mostrar primeras 5 rutas
        print(f"     * {route.rule} -> {route.endpoint}")
    
    # 6. Probar renderización de plantilla
    print("6. Probando renderización de plantilla...")
    with app.app_context():
        try:
            from flask import render_template
            # Probar renderización básica
            html = render_template('index.html', 
                                 page_title="Test",
                                 student_names_for_select=[],
                                 course_names_for_select=[],
                                 now=None)
            print(f"✓ Plantilla renderizada exitosamente ({len(html)} caracteres)")
        except Exception as e:
            print(f"✗ Error en renderización: {e}")
            traceback.print_exc()
    
    print("\n=== DIAGNÓSTICO COMPLETADO ===")
    print("La aplicación parece estar funcionando correctamente.")
    print("Para ejecutarla: python run.py")
    
except Exception as e:
    print(f"\n✗ ERROR EN DIAGNÓSTICO: {e}")
    traceback.print_exc()
    sys.exit(1)