#!/usr/bin/env python3
"""
Pruebas de alias de curso para niveles de Media y casos sin paralelo.
- Verifica detección para 'Primero Medio A' con prompts: '1 Medio A', '1º Medio',
  'Primero Medio', '1M', '1 Med' y formas compactas ('1ma').
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

    # Analizador mock que devuelve el markdown con los bloques de resumen
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
        tmp_db = os.path.join(app.instance_path, 'test_course_aliases_media.db')
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
        csv_path = os.path.join(tmp_dir, 'media.csv')

        # Dataset con un curso: 'Primero Medio A'
        df_csv = pd.DataFrame([
            {nombre_col: 'Carlos Soto', curso_col: 'Primero Medio A', asig_col: 'Historia',  nota_col: 5.5, asis_col: 0.90, obs_col: 'Participa activamente'},
            {nombre_col: 'Lucía Rojas', curso_col: 'Primero Medio A', asig_col: 'Química',   nota_col: 6.0, asis_col: 0.96, obs_col: 'Puntual y responsable'},
        ])
        df_csv.to_csv(csv_path, index=False)

        # Dos snapshots para tener histórico/cualitativo
        ok1, msg1 = save_data_snapshot_to_db(df_csv, 'media_snap1.csv', app.config['DATABASE_FILE'])
        assert ok1, f"Guardar snap1 fallo: {msg1}"
        df2 = df_csv.copy()
        df2.loc[df2[nombre_col] == 'Carlos Soto', obs_col] = 'Dificultades de disciplina'
        ok2, msg2 = save_data_snapshot_to_db(df2, 'media_snap2.csv', app.config['DATABASE_FILE'])
        assert ok2, f"Guardar snap2 fallo: {msg2}"

        client = app.test_client()
        with client.session_transaction() as sess:
            sess['current_file_path'] = csv_path
            sess['uploaded_filename'] = os.path.basename(csv_path)

        prompts = [
            'Resumen cualitativo del curso 1 Medio A',
            'Resumen de conducta del 1º Medio',
            'Cualitativo Primero Medio',
            'Resumen cualitativo 1M',
            'Resumen cualitativo 1 Med',
            'Resumen cualitativo 1ma',
        ]

        for p in prompts:
            r = client.post('/api/submit_advanced_chat', data=json.dumps({'prompt': p}), content_type='application/json')
            assert r.status_code == 200, f"status: {r.status_code} body: {r.data}"
            md = r.get_json()['raw_markdown']
            quali_block = md.split('QUALI:\n')[-1].strip()
            assert len(quali_block) > 0, f"Bloque cualitativo vacío para prompt: {p}"

        print('✓ Alias de cursos Media y sin paralelo verificados')
        return 0

if __name__ == '__main__':
    sys.exit(main())

