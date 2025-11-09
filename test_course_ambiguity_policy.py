#!/usr/bin/env python3
"""
Pruebas de manejo de ambigüedad en detección de curso:
- Política 'ask': devuelve mensaje de ambigüedad con letras disponibles.
- Política 'default': selecciona el paralelo indicado y devuelve resumen cualitativo.
"""
import os
import sys
import json
import tempfile
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    from mi_aplicacion import create_app
    from mi_aplicacion import routes as routes_module
    from mi_aplicacion.app_logic import init_sqlite_db, save_data_snapshot_to_db

    app = create_app('dev')

    # Analizador mock que refleja los bloques para validación
    def echo_analyze_data_with_gemini(*args, **kwargs):
        hist = kwargs.get('historical_data_summary_string', '') or ''
        quali = kwargs.get('qualitative_history_summary_string', '') or ''
        raw_md = f"HIST:\n{hist}\n\nQUALI:\n{quali}"
        return {
            'html_output': '<p>ok</p>',
            'raw_markdown': raw_md,
            'model_name': 'mocked',
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'input_cost': 0.0,
            'output_cost': 0.0,
            'total_cost': 0.0,
            'error': None,
        }

    routes_module.analyze_data_with_gemini = echo_analyze_data_with_gemini

    with app.app_context():
        # BD temporal
        tmp_db = os.path.join(app.instance_path, 'test_course_ambiguity.db')
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

        tmp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(tmp_dir, 'ambig.csv')

        # Dataset con dos paralelos del mismo nivel
        df_csv = pd.DataFrame([
            {nombre_col: 'Mario Ruiz',  curso_col: 'Primero Medio A', asig_col: 'Historia', nota_col: 5.2, asis_col: 0.91, obs_col: 'A: conducta A'},
            {nombre_col: 'Ana Lara',    curso_col: 'Primero Medio A', asig_col: 'Química',  nota_col: 6.1, asis_col: 0.95, obs_col: 'A: puntualidad destacada'},
            {nombre_col: 'Pedro Núñez', curso_col: 'Primero Medio B', asig_col: 'Historia', nota_col: 5.8, asis_col: 0.93, obs_col: 'B: conducta B'},
            {nombre_col: 'Sofía Mena',  curso_col: 'Primero Medio B', asig_col: 'Química',  nota_col: 5.9, asis_col: 0.92, obs_col: 'B: disciplina mejorada'},
        ])
        df_csv.to_csv(csv_path, index=False)

        # Guardar dos snapshots con cambios de observaciones
        ok1, msg1 = save_data_snapshot_to_db(df_csv, 'ambig_snap1.csv', app.config['DATABASE_FILE'])
        assert ok1, f"Guardar snap1 fallo: {msg1}"
        df2 = df_csv.copy()
        df2.loc[df2[nombre_col] == 'Mario Ruiz', obs_col] = 'A: nueva observación'
        df2.loc[df2[nombre_col] == 'Pedro Núñez', obs_col] = 'B: nueva observación'
        ok2, msg2 = save_data_snapshot_to_db(df2, 'ambig_snap2.csv', app.config['DATABASE_FILE'])
        assert ok2, f"Guardar snap2 fallo: {msg2}"

        client = app.test_client()
        with client.session_transaction() as sess:
            sess['current_file_path'] = csv_path
            sess['uploaded_filename'] = os.path.basename(csv_path)

        prompt = 'Necesito resumen cualitativo de Primero Medio'

        # Política ASK: debe incluir mensaje de ambigüedad
        app.config['COURSE_AMBIGUITY_POLICY'] = 'ask'
        r1 = client.post('/api/submit_advanced_chat', data=json.dumps({'prompt': prompt}), content_type='application/json')
        assert r1.status_code == 200, f"status: {r1.status_code} body: {r1.data}"
        md1 = r1.get_json()['raw_markdown']
        quali1 = md1.split('QUALI:\n')[-1].strip()
        assert 'Ambigüedad detectada' in quali1, 'No se encontró el mensaje de ambigüedad'
        assert 'A' in quali1 and 'B' in quali1, 'No lista letras disponibles'

        # Política DEFAULT con paralelo B -> debe resolver y no mostrar ambigüedad
        app.config['COURSE_AMBIGUITY_POLICY'] = 'default'
        app.config['DEFAULT_PARALLEL'] = 'B'
        r2 = client.post('/api/submit_advanced_chat', data=json.dumps({'prompt': prompt}), content_type='application/json')
        assert r2.status_code == 200, f"status: {r2.status_code} body: {r2.data}"
        md2 = r2.get_json()['raw_markdown']
        quali2 = md2.split('QUALI:\n')[-1].strip()
        assert 'Ambigüedad detectada' not in quali2, 'No debería mostrar ambigüedad con política default'
        assert 'B:' in quali2, 'El resumen cualitativo debería reflejar datos del paralelo B'

        print('✓ Manejo de ambigüedad de cursos verificado (ask y default)')
        return 0

if __name__ == '__main__':
    sys.exit(main())

