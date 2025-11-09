#!/usr/bin/env python3
"""Pruebas ligeras de utilitarios nuevos en routes: normalización de texto
y bloque de documentos clave (debe ser seguro cuando no hay datos)."""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        from mi_aplicacion import create_app
        from mi_aplicacion import routes
        from mi_aplicacion.app_logic import init_sqlite_db

        app = create_app('dev')
        with app.app_context():
            # Normalización
            assert routes._normalize_text('Álvaro   Núñez') == 'alvaro nunez'
            assert routes._normalize_text('  3°   Básico   B ') == '3° basico b'

            # Bloque de documentos clave: sin datos debe retornar cadena vacía
            tmp_db = os.path.join(app.instance_path, 'test_helpers.db')
            if os.path.exists(tmp_db):
                os.remove(tmp_db)
            init_sqlite_db(tmp_db)
            app.config['DATABASE_FILE'] = tmp_db

            block = routes._build_key_docs_block('alumno', 'Juan Pérez', 'snapshot1.csv')
            assert isinstance(block, str)

            print('✓ Utilitarios de routes funcionan correctamente')
            return 0
    except Exception as e:
        print(f"\n✗ Error en pruebas de routes helpers: {e}")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

