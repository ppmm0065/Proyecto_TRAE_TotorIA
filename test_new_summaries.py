#!/usr/bin/env python3
"""
Tests automáticos para validar que las nuevas rutas lógicas devuelven texto
sin errores: evolución de asistencia y resumen cualitativo por curso.

Se usan datos sintéticos mínimos y una BD SQLite temporal.
"""
import os
import sys
import traceback
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        from mi_aplicacion import create_app
        from mi_aplicacion.app_logic import (
            init_sqlite_db,
            save_data_snapshot_to_db,
            get_attendance_evolution_summary,
            get_course_qualitative_summary,
        )
        import pandas as pd

        # Crear app y contexto
        app = create_app('dev')
        with app.app_context():
            # BD temporal
            tmp_db = os.path.join(app.instance_path, 'test_summaries.db')
            if os.path.exists(tmp_db):
                os.remove(tmp_db)
            init_sqlite_db(tmp_db)
            app.config['DATABASE_FILE'] = tmp_db

            # Construir datos sintéticos (formato largo)
            nombre_col = app.config['NOMBRE_COL']
            curso_col = app.config['CURSO_COL']
            asig_col = app.config['ASIGNATURA_COL']
            nota_col = app.config['NOTA_COL']
            asis_col = app.config['ASISTENCIA_COL']
            obs_col = app.config['OBSERVACIONES_COL']
            fam_col = app.config['FAMILIA_COL']
            intv_col = app.config['ENTREVISTAS_COL']

            # Snapshot 1
            df1 = pd.DataFrame([
                {nombre_col: 'Juan Pérez', curso_col: '3° Básico B', asig_col: 'Matemáticas', nota_col: 4.5, asis_col: 0.90, obs_col: 'Interrumpe en clases', fam_col: 'Se comunica regularmente', intv_col: 'Entrevista con apoderado'},
                {nombre_col: 'Juan Pérez', curso_col: '3° Básico B', asig_col: 'Lenguaje',    nota_col: 5.0, asis_col: 0.90, obs_col: 'Participa activamente', fam_col: 'N/A', intv_col: 'N/A'},
                {nombre_col: 'Ana Díaz',  curso_col: '3° Básico B', asig_col: 'Matemáticas', nota_col: 6.0, asis_col: 0.95, obs_col: 'Respeta normas', fam_col: 'N/A', intv_col: 'N/A'},
                {nombre_col: 'Ana Díaz',  curso_col: '3° Básico B', asig_col: 'Lenguaje',    nota_col: 5.8, asis_col: 0.95, obs_col: 'Conducta destacada', fam_col: 'N/A', intv_col: 'N/A'},
            ])
            ok1, msg1 = save_data_snapshot_to_db(df1, 'snapshot1.csv', app.config['DATABASE_FILE'])
            if not ok1:
                raise RuntimeError(f"Fallo guardando snapshot1: {msg1}")

            # Snapshot 2 (cambios en asistencia y observaciones)
            df2 = pd.DataFrame([
                {nombre_col: 'Juan Pérez', curso_col: '3° Básico B', asig_col: 'Matemáticas', nota_col: 4.2, asis_col: 0.80, obs_col: 'Ofende al profesor', fam_col: 'Contacto esporádico', intv_col: 'Nueva entrevista'},
                {nombre_col: 'Juan Pérez', curso_col: '3° Básico B', asig_col: 'Lenguaje',    nota_col: 4.8, asis_col: 0.80, obs_col: 'Participa poco', fam_col: 'N/A', intv_col: 'N/A'},
                {nombre_col: 'Ana Díaz',  curso_col: '3° Básico B', asig_col: 'Matemáticas', nota_col: 6.2, asis_col: 0.97, obs_col: 'Mantiene respeto', fam_col: 'N/A', intv_col: 'Entrevista de seguimiento'},
                {nombre_col: 'Ana Díaz',  curso_col: '3° Básico B', asig_col: 'Lenguaje',    nota_col: 6.1, asis_col: 0.97, obs_col: 'Excelente participación', fam_col: 'N/A', intv_col: 'N/A'},
            ])
            ok2, msg2 = save_data_snapshot_to_db(df2, 'snapshot2.csv', app.config['DATABASE_FILE'])
            if not ok2:
                raise RuntimeError(f"Fallo guardando snapshot2: {msg2}")

            # Prueba: evolución de asistencia por alumno
            att_text = get_attendance_evolution_summary(app.config['DATABASE_FILE'], entity_name='Juan Pérez')
            assert isinstance(att_text, str) and len(att_text.strip()) > 0, "Texto de asistencia vacío"
            assert 'Evolución' in att_text or 'Asistencia Inicial' in att_text, "No contiene claves esperadas de asistencia"

            # Prueba: resumen cualitativo por curso
            qual_text = get_course_qualitative_summary(app.config['DATABASE_FILE'], '3° Básico B', max_entries=10)
            assert isinstance(qual_text, str) and len(qual_text.strip()) > 0, "Texto cualitativo vacío"
            assert 'Totales por tipo' in qual_text or 'Entradas recientes' in qual_text, "No contiene claves esperadas cualitativas"

            print("✓ Nuevos resúmenes devuelven textos válidos (asistencia y cualitativo curso)")
            return 0

    except Exception as e:
        print(f"\n✗ Error en pruebas de nuevos resúmenes: {e}")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

