#!/usr/bin/env python3
import os
import csv
import sys

def main():
    from mi_aplicacion import create_app
    app = create_app('dev')
    with app.app_context():
        client = app.test_client()
        tmp_csv = os.path.join(app.config['UPLOAD_FOLDER'], 'tmp_risk.csv')
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        rows = [
            ['Nombre','curso','Asignatura','Nota','Asistencia','Observacion de conducta'],
            ['Ana','3 basico A','Matematicas','3.5','0.80','interrumpe en clases'],
            ['Ana','3 basico A','Lenguaje','3.8','0.80',''],
            ['Luis','3 basico A','Matematicas','4.5','0.88',''],
            ['Luis','3 basico A','Lenguaje','4.8','0.88',''],
        ]
        with open(tmp_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerows(rows)
        with client.session_transaction() as sess:
            sess['current_file_path'] = tmp_csv
        resp = client.get('/api/risk_summary')
        assert resp.status_code == 200, resp.data
        data = resp.get_json()
        assert 'totals' in data
        assert 'by_course' in data
        print('✓ /api/risk_summary OK:', data['totals'])

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('✗ Error en test_endpoints_risk:', e)
        import traceback
        traceback.print_exc()
        sys.exit(1)