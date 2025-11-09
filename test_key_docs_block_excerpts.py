#!/usr/bin/env python3
"""Prueba ligera para el bloque de documentos clave con extractos.
Verifica que se incluyan los documentos y sus extractos truncados.
"""
import os
import sys
import sqlite3
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        from mi_aplicacion import create_app, routes
        from mi_aplicacion.app_logic import init_sqlite_db

        app = create_app('dev')
        with app.app_context():
            # DB temporal
            tmp_db = os.path.join(app.instance_path, 'test_key_docs.db')
            if os.path.exists(tmp_db):
                os.remove(tmp_db)
            init_sqlite_db(tmp_db)
            app.config['DATABASE_FILE'] = tmp_db

            tipo = 'alumno'; nombre = 'Juan Pérez'; filename = 'snapshot1.csv'
            # Insertar un Reporte 360 y un Plan de Intervención
            reporte_md = """Linea 1 del reporte\nLinea 2 del reporte\nLinea 3 del reporte\n"""
            plan_md = """Objetivo 1: Mejorar lectura\nAcción: 15 minutos diarios\nIndicador: Comprensión literal\n"""

            with sqlite3.connect(tmp_db) as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO follow_ups(related_filename, related_prompt, related_analysis, follow_up_comment, follow_up_type, related_entity_type, related_entity_name)
                    VALUES (?, ?, ?, ?, 'reporte_360', ?, ?)
                    """,
                    (filename, 'Prueba reporte', 'base', reporte_md, tipo, nombre)
                )
                cur.execute(
                    """
                    INSERT INTO follow_ups(related_filename, related_prompt, related_analysis, follow_up_comment, follow_up_type, related_entity_type, related_entity_name)
                    VALUES (?, ?, ?, ?, 'intervention_plan', ?, ?)
                    """,
                    (filename, 'Prueba plan', 'base', plan_md, tipo, nombre)
                )
                conn.commit()

            # Construir bloque
            block = routes._build_key_docs_block(tipo, nombre, filename)
            assert isinstance(block, str)
            assert 'Documentos clave recientes' in block
            assert 'Reporte 360' in block and 'Plan de Intervención' in block
            assert 'Resumen breve' in block
            assert 'Linea 1 del reporte' in block
            assert 'Objetivo 1: Mejorar lectura' in block

            print('✓ Bloque de documentos clave con extractos funciona')
            return 0
    except Exception as e:
        print(f"\n✗ Error en prueba de documentos clave: {e}")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

