#!/usr/bin/env python3
import sys
import pandas as pd

def main():
    from mi_aplicacion import create_app
    from mi_aplicacion.app_logic import build_feature_store_from_csv, compute_risk_scores_from_feature_store, build_risk_summary

    app = create_app('dev')
    with app.app_context():
        data = [
            {'Nombre':'Ana','curso':'3 basico A','Asignatura':'Matematicas','Nota':3.5,'Asistencia':0.80,'Observacion de conducta':'interrumpe en clases','Familia':'','Entrevistas':''},
            {'Nombre':'Ana','curso':'3 basico A','Asignatura':'Lenguaje','Nota':3.8,'Asistencia':0.80,'Observacion de conducta':'','Familia':'','Entrevistas':''},
            {'Nombre':'Luis','curso':'3 basico A','Asignatura':'Matematicas','Nota':4.5,'Asistencia':0.88,'Observacion de conducta':'','Familia':'','Entrevistas':''},
            {'Nombre':'Luis','curso':'3 basico A','Asignatura':'Lenguaje','Nota':4.8,'Asistencia':0.88,'Observacion de conducta':'','Familia':'','Entrevistas':''},
            {'Nombre':'Marta','curso':'3 basico B','Asignatura':'Matematicas','Nota':6.0,'Asistencia':0.95,'Observacion de conducta':'','Familia':'','Entrevistas':''},
            {'Nombre':'Marta','curso':'3 basico B','Asignatura':'Lenguaje','Nota':6.2,'Asistencia':0.95,'Observacion de conducta':'','Familia':'','Entrevistas':''},
        ]
        df = pd.DataFrame(data)
        fs = build_feature_store_from_csv(df)
        scores = compute_risk_scores_from_feature_store(fs)
        summ = build_risk_summary(fs)

        assert scores['Ana']['level'] in ('alto','medio')
        assert scores['Marta']['level'] == 'bajo'
        assert summ['totals']['alto'] >= 0
        print("✓ Scoring de riesgo validado:", scores)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("✗ Error en prueba de scoring:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)