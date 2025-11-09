#!/usr/bin/env python3
"""
Pruebas end-to-end de endpoints de chat con el analizador mockeado.
- Verifica /api/detalle_chat con intenciones de asistencia (alumno)
- Verifica /api/submit_advanced_chat con asistencia general y cualitativo por curso

No se realizan llamadas externas. Se usa una BD SQLite temporal y CSV sintético.
"""
import os
import sys
import json
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        from mi_aplicacion import create_app
        from mi_aplicacion import routes as routes_module
        from mi_aplicacion.app_logic import init_sqlite_db, save_data_snapshot_to_db
        import pandas as pd

        # Crea app y cliente de pruebas
        app = create_app('dev')

        # Mock: reemplaza analyze_data_with_gemini por una versión local
        def fake_analyze_data_with_gemini(*args, **kwargs):
            # Devuelve un resultado fijo, suficiente para la UI y las rutas
            return {
                'html_output': '<p>Respuesta simulada</p>',
                'raw_markdown': 'Respuesta simulada',
                'model_name': 'mocked-model',
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0,
                'input_cost': 0.0,
                'output_cost': 0.0,
                'total_cost': 0.0,
                'error': None,
            }

        routes_module.analyze_data_with_gemini = fake_analyze_data_with_gemini

        with app.app_context():
            # BD temporal
            tmp_db = os.path.join(app.instance_path, 'test_endpoints_mocked.db')
            if os.path.exists(tmp_db):
                os.remove(tmp_db)
            init_sqlite_db(tmp_db)
            app.config['DATABASE_FILE'] = tmp_db

            # Columnas desde config
            nombre_col = app.config['NOMBRE_COL']
            curso_col = app.config['CURSO_COL']
            asig_col = app.config['ASIGNATURA_COL']
            nota_col = app.config['NOTA_COL']
            asis_col = app.config['ASISTENCIA_COL']
            obs_col = app.config['OBSERVACIONES_COL']
            fam_col = app.config['FAMILIA_COL']
            intv_col = app.config['ENTREVISTAS_COL']

            # CSV temporal con datos sintéticos mínimos
            tmp_dir = tempfile.mkdtemp()
            csv_path = os.path.join(tmp_dir, 'datos.csv')
            df_csv = pd.DataFrame([
                {nombre_col: 'Juan Pérez', curso_col: '3° Básico B', asig_col: 'Matemáticas', nota_col: 4.5, asis_col: 0.90, obs_col: 'Interrumpe en clases', fam_col: 'Se comunica', intv_col: 'Entrevista 1'},
                {nombre_col: 'Juan Pérez', curso_col: '3° Básico B', asig_col: 'Lenguaje',    nota_col: 5.0, asis_col: 0.90, obs_col: 'Participa',            fam_col: 'N/A',         intv_col: 'N/A'},
                {nombre_col: 'Ana Díaz',  curso_col: '3° Básico B', asig_col: 'Matemáticas', nota_col: 6.0, asis_col: 0.95, obs_col: 'Respeta normas',     fam_col: 'N/A',         intv_col: 'N/A'},
                {nombre_col: 'Ana Díaz',  curso_col: '3° Básico B', asig_col: 'Lenguaje',    nota_col: 5.8, asis_col: 0.95, obs_col: 'Conducta destacada', fam_col: 'N/A',         intv_col: 'N/A'},
            ])
            df_csv.to_csv(csv_path, index=False)

            # Guardar snapshots a BD para habilitar resúmenes históricos
            ok1, msg1 = save_data_snapshot_to_db(df_csv, 'snapshot1.csv', app.config['DATABASE_FILE'])
            if not ok1:
                raise RuntimeError(f"Fallo guardando snapshot1: {msg1}")

            df2 = df_csv.copy()
            df2.loc[df2[nombre_col] == 'Juan Pérez', asis_col] = 0.80
            df2.loc[df2[nombre_col] == 'Juan Pérez', obs_col] = 'Ofende al profesor'
            ok2, msg2 = save_data_snapshot_to_db(df2, 'snapshot2.csv', app.config['DATABASE_FILE'])
            if not ok2:
                raise RuntimeError(f"Fallo guardando snapshot2: {msg2}")

            # Cliente de pruebas Flask
            client = app.test_client()

            # Configurar sesión
            with client.session_transaction() as sess:
                sess['current_file_path'] = csv_path
                sess['uploaded_filename'] = os.path.basename(csv_path)

            # 1) /api/detalle_chat alumno + asistencia
            payload_detalle = {
                'tipo_entidad': 'alumno',
                'nombre_entidad': 'Juan Pérez',
                'prompt': 'Muéstrame la evolución de la asistencia del alumno Juan Perez'
            }
            r1 = client.post('/api/detalle_chat', data=json.dumps(payload_detalle), content_type='application/json')
            assert r1.status_code == 200, f"detalle_chat status: {r1.status_code}, body: {r1.data}"
            data1 = r1.get_json()
            assert data1 and not data1.get('error'), f"detalle_chat error: {data1}"
            assert 'html_output' in data1 and 'raw_markdown' in data1

            # 2) /api/submit_advanced_chat asistencia general (sin entidad)
            payload_adv_att = { 'prompt': 'evolucion de asistencia general del establecimiento' }
            r2 = client.post('/api/submit_advanced_chat', data=json.dumps(payload_adv_att), content_type='application/json')
            assert r2.status_code == 200, f"advanced_chat asistencia status: {r2.status_code}, body: {r2.data}"
            data2 = r2.get_json()
            assert data2 and not data2.get('error')
            assert 'html_output' in data2 and 'raw_markdown' in data2

            # 3) /api/submit_advanced_chat cualitativo por curso
            payload_adv_qual = { 'prompt': 'dame un resumen cualitativo del curso 3° Basico B' }
            r3 = client.post('/api/submit_advanced_chat', data=json.dumps(payload_adv_qual), content_type='application/json')
            assert r3.status_code == 200, f"advanced_chat cualitativo status: {r3.status_code}, body: {r3.data}"
            data3 = r3.get_json()
            assert data3 and not data3.get('error')
            assert 'html_output' in data3 and 'raw_markdown' in data3

            print('✓ Endpoints /api/detalle_chat y /api/submit_advanced_chat responden correctamente (mock)')
            return 0

    except Exception as e:
        print(f"\n✗ Error en pruebas end-to-end con mocks: {e}")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

