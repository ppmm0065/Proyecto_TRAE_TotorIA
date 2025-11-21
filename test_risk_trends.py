#!/usr/bin/env python3
import os
import sys
import pandas as pd
import sqlite3

def main():
    from mi_aplicacion import create_app
    from mi_aplicacion.app_logic import build_feature_store_from_csv, compute_risk_scores_from_feature_store, init_sqlite_db

    app = create_app('dev')
    with app.app_context():
        test_db = os.path.join(app.config['UPLOAD_FOLDER'], 'risk_trends_test.db')
        app.config['DATABASE_FILE'] = test_db
        init_sqlite_db(test_db)
        with sqlite3.connect(test_db) as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO data_snapshots (filename, num_students, num_records) VALUES (?,?,?)", ('t1.csv', 2, 2))
            s1 = cur.lastrowid
            cur.execute("INSERT INTO data_snapshots (filename, num_students, num_records) VALUES (?,?,?)", ('t2.csv', 2, 2))
            s2 = cur.lastrowid
            cur.execute("INSERT INTO student_data_history (snapshot_id, student_name, student_course, subject_name, grade) VALUES (?,?,?,?,?)", (s1,'Ana','3 basico A','Matematicas',5.0))
            cur.execute("INSERT INTO student_data_history (snapshot_id, student_name, student_course, subject_name, grade) VALUES (?,?,?,?,?)", (s2,'Ana','3 basico A','Matematicas',3.5))
            conn.commit()

        data = [
            {'Nombre':'Ana','curso':'3 basico A','Asignatura':'Matematicas','Nota':3.5,'Asistencia':0.90,'Observacion de conducta':''},
            {'Nombre':'Luis','curso':'3 basico A','Asignatura':'Matematicas','Nota':5.0,'Asistencia':0.95,'Observacion de conducta':''},
        ]
        df = pd.DataFrame(data)
        fs = build_feature_store_from_csv(df)
        scores = compute_risk_scores_from_feature_store(fs)
        assert 'Ana' in scores
        assert scores['Ana']['score'] >= scores['Luis']['score']
        assert 'tendencia_baja_notas' in scores['Ana']['reasons']
        print('✓ Tendencias aplicadas en scoring:', scores['Ana'])

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('✗ Error en test_risk_trends:', e)
        import traceback
        traceback.print_exc()
        sys.exit(1)