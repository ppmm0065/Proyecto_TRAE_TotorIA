#!/usr/bin/env python3
import os
import sys
import pandas as pd

def main():
    from mi_aplicacion import create_app
    from mi_aplicacion.app_logic import build_feature_store_from_csv, compute_risk_scores_from_feature_store

    app = create_app('dev')
    with app.app_context():
        app.config['RISK_THRESHOLDS_BY_LEVEL'] = {
            'basico': {'attendance': 0.85, 'grade': 4.0},
            'medio': {'attendance': 0.90, 'grade': 4.5}
        }
        df = pd.DataFrame([
            {'Nombre':'Ana','curso':'4 basico A','Asignatura':'Matematicas','Nota':5.0,'Asistencia':0.88,'Observacion de conducta':''},
            {'Nombre':'Luis','curso':'4 medio A','Asignatura':'Matematicas','Nota':5.0,'Asistencia':0.88,'Observacion de conducta':''},
        ])
        fs = build_feature_store_from_csv(df)
        scores = compute_risk_scores_from_feature_store(fs)
        assert 'asistencia_baja' not in scores['Ana']['reasons']
        assert 'asistencia_baja' in scores['Luis']['reasons']
        print('✓ Umbrales por nivel aplicados:', scores)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('✗ Error en test_attendance_thresholds:', e)
        import traceback
        traceback.print_exc()
        sys.exit(1)