#!/usr/bin/env python3
"""
Pruebas de detección de entidad CURSO con variaciones comunes en el prompt.
- Verifica que '/api/submit_advanced_chat' detecta el curso con: "3 Basico B",
  "3º Basico B" y "3o Basico B" cuando el CSV tiene "3° Básico B".
- Usa analizador mock que refleja los resúmenes en el markdown.
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    from mi_aplicacion import create_app
    from mi_aplicacion import routes as routes_module
    from mi_aplicacion.app_logic import init_sqlite_db, save_data_snapshot_to_db
    import pandas as pd

    app = create_app('dev')

    # Mock del analizador: refleja resúmenes para validar detección
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
        tmp_db = os.path.join(app.instance_path, 'test_entity_variations.db')
        if os.path.exists(tmp_db):
            os.remove(tmp_db)
        init_sqlite_db(tmp_db)
        app.config['DATABASE_FILE'] = tmp_db

        # Columnas
        nombre_col = app.config['NOMBRE_COL']
        curso_col = app.config['CURSO_COL']
        asig_col = app.config['ASIGNATURA_COL']
        nota_col = app.config['NOTA_COL']
        asis_col = app.config['ASISTENCIA_COL']
        obs_col = app.config['OBSERVACIONES_COL']
        fam_col = app.config['FAMILIA_COL']
        intv_col = app.config['ENTREVISTAS_COL']

        # CSV con curso '3° Básico B'
        tmp_dir = tempfile.mkdtemp()
        csv_path = os.path.join(tmp_dir, 'datos.csv')
        df_csv = pd.DataFrame([
            {nombre_col: 'Juan Pérez', curso_col: '3° Básico B', asig_col: 'Matemáticas', nota_col: 5.0, asis_col: 0.92, obs_col: 'Interrumpe en clases', fam_col: 'Se comunica', intv_col: 'Entrevista 1'},
            {nombre_col: 'Ana Díaz',  curso_col: '3° Básico B', asig_col: 'Lenguaje',    nota_col: 5.8, asis_col: 0.95, obs_col: 'Conducta destacada', fam_col: 'N/A',         intv_col: 'N/A'},
        ])
        df_csv.to_csv(csv_path, index=False)

        # Guardar dos snapshots para que haya datos cualitativos/históricos
        ok1, msg1 = save_data_snapshot_to_db(df_csv, 'snap1.csv', app.config['DATABASE_FILE'])
        assert ok1, f"Guardar snap1 fallo: {msg1}"
        df2 = df_csv.copy()
        df2.loc[df2[nombre_col] == 'Juan Pérez', asis_col] = 0.80
        df2.loc[df2[nombre_col] == 'Juan Pérez', obs_col] = 'Faltas de respeto'
        ok2, msg2 = save_data_snapshot_to_db(df2, 'snap2.csv', app.config['DATABASE_FILE'])
        assert ok2, f"Guardar snap2 fallo: {msg2}"

        client = app.test_client()
        with client.session_transaction() as sess:
            sess['current_file_path'] = csv_path
            sess['uploaded_filename'] = os.path.basename(csv_path)

        prompts = [
            'Necesito resumen de conducta del curso 3 Basico B',
            'Resumen cualitativo del curso 3º Basico B',
            'Resumen de conducta del 3o Basico B',
        ]

        for p in prompts:
            r = client.post('/api/submit_advanced_chat', data=json.dumps({'prompt': p}), content_type='application/json')
            assert r.status_code == 200, f"status: {r.status_code} body: {r.data}"
            md = r.get_json()['raw_markdown']
            quali_block = md.split('QUALI:\n')[-1].strip()
            assert len(quali_block) > 0, f"Bloque cualitativo vacío para prompt: {p}"

        print('✓ Detección de curso con variaciones verificada')
        return 0

if __name__ == '__main__':
    sys.exit(main())

