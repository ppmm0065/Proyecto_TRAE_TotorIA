#!/usr/bin/env python3
import sys
import pandas as pd

def main():
    from mi_aplicacion import create_app
    from mi_aplicacion.app_logic import build_feature_store_from_csv, compute_risk_scores_from_feature_store

    app = create_app('dev')
    with app.app_context():
        data = [
            {'Nombre':'Ana','curso':'3 basico A','Asignatura':'Matematicas','Nota':3.5,'Asistencia':0.95,'Observacion de conducta':''},
            {'Nombre':'Ana','curso':'3 basico A','Asignatura':'Lenguaje','Nota':3.6,'Asistencia':0.95,'Observacion de conducta':''},
            {'Nombre':'Luis','curso':'3 basico A','Asignatura':'Matematicas','Nota':3.9,'Asistencia':0.95,'Observacion de conducta':''},
            {'Nombre':'Luis','curso':'3 basico A','Asignatura':'Lenguaje','Nota':3.8,'Asistencia':0.95,'Observacion de conducta':''},
            {'Nombre':'Luis','curso':'3 basico A','Asignatura':'Historia','Nota':4.5,'Asistencia':0.95,'Observacion de conducta':''},
        ]
        df = pd.DataFrame(data)
        fs = build_feature_store_from_csv(df)
        scores = compute_risk_scores_from_feature_store(fs)
        assert scores['Ana']['level'] == 'alto', scores['Ana']
        assert scores['Luis']['level'] == 'medio', scores['Luis']
        print('✓ Sensibilidad de riesgo validada')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('✗ Error en test_risk_sensitivity:', e)
        import traceback
        traceback.print_exc()
        sys.exit(1)