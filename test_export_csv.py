#!/usr/bin/env python3
import os
import sys
import csv

def main():
    from mi_aplicacion import create_app
    app = create_app('dev')
    with app.app_context():
        client = app.test_client()
        tmp_csv = os.path.join(app.config['UPLOAD_FOLDER'], 'tmp_export.csv')
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        rows = [
            ['Nombre','curso','Asignatura','Nota','Asistencia','Observacion de conducta'],
            ['Ana','3 basico A','Matematicas','3.5','0.80',''],
            ['Ana','3 basico A','Lenguaje','3.8','0.80',''],
            ['Luis','3 basico A','Matematicas','5.5','0.95',''],
        ]
        with open(tmp_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerows(rows)
        with client.session_transaction() as sess:
            sess['current_file_path'] = tmp_csv
        resp = client.get('/api/risk_summary_export.csv')
        assert resp.status_code == 200, resp.data
        assert resp.headers.get('Content-Type').startswith('text/csv')
        body = resp.data.decode('utf-8')
        assert 'Curso,Alto,Medio,Bajo' in body
        print('✓ Export CSV OK')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('✗ Error en test_export_csv:', e)
        import traceback
        traceback.print_exc()
        sys.exit(1)