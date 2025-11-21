#!/usr/bin/env python3
import os
import sys
import csv

def main():
    from mi_aplicacion import create_app
    app = create_app('dev')
    with app.app_context():
        client = app.test_client()
        tmp_csv = os.path.join(app.config['UPLOAD_FOLDER'], 'tmp_suggest.csv')
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        rows = [
            ['Nombre','curso','Asignatura','Nota','Asistencia','Observacion de conducta'],
            ['Ana','3 basico A','Matematicas','3.5','0.80','interrumpe en clases'],
            ['Ana','3 basico A','Lenguaje','3.8','0.80',''],
            ['Luis','3 basico A','Matematicas','5.5','0.95',''],
        ]
        with open(tmp_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerows(rows)
        with client.session_transaction() as sess:
            sess['current_file_path'] = tmp_csv
        resp = client.get('/api/suggest_interventions/alumno/Ana')
        assert resp.status_code == 200, resp.data
        data = resp.get_json()
        assert data['student'] == 'Ana'
        assert isinstance(data['actions'], list) and len(data['actions']) > 0
        print('✓ Sugerencias de intervención OK:', len(data['actions']))

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('✗ Error en test_suggest_interventions:', e)
        import traceback
        traceback.print_exc()
        sys.exit(1)