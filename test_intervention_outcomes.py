#!/usr/bin/env python3
import os
import sys

def main():
    from mi_aplicacion import create_app
    from mi_aplicacion.app_logic import init_sqlite_db, save_intervention_plan_to_db
    app = create_app('dev')
    with app.app_context():
        client = app.test_client()
        test_db = os.path.join(app.config['UPLOAD_FOLDER'], 'outcomes_test.db')
        app.config['DATABASE_FILE'] = test_db
        init_sqlite_db(test_db)
        plan_id = save_intervention_plan_to_db(test_db, 't.csv', 'alumno', 'Ana', '# Plan', '# Base')
        assert plan_id is not None
        payload = {
            'follow_up_id': int(plan_id),
            'related_entity_type': 'alumno',
            'related_entity_name': 'Ana',
            'course_name': '3 basico A',
            'outcome_date': '2025-11-21',
            'compliance_pct': 0.8,
            'impact_grade_delta': 0.5,
            'impact_attendance_delta': 2.0,
            'notes': 'Mejora parcial'
        }
        resp = client.post('/api/intervention_outcome', json=payload)
        assert resp.status_code == 201, resp.data
        rid = resp.get_json().get('id')
        assert rid is not None
        resp2 = client.get('/api/intervention_effectiveness?days=365')
        assert resp2.status_code == 200
        data = resp2.get_json()
        assert data['totals']['count'] >= 1
        print('✓ Outcomes y efectividad OK')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('✗ Error en test_intervention_outcomes:', e)
        import traceback
        traceback.print_exc()
        sys.exit(1)