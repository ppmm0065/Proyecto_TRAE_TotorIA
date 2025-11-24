# mi_aplicacion/routes.py
import os
import unicodedata
import pandas as pd
from flask import (
    Blueprint, render_template, request, redirect, url_for, 
    session, flash, jsonify, current_app, make_response, Response
)
from werkzeug.utils import secure_filename
import datetime
import traceback
import markdown
from urllib.parse import unquote, quote 
import numpy as np 
import pytz
from pytz import timezone

from langchain_community.vectorstores import FAISS 

from .app_logic import (
    get_dataframe_from_session_file,
    load_data_as_string,
    analyze_data_with_gemini,
    format_chat_history_for_prompt,
    load_follow_ups_as_documents,
    embedding_model_instance, 
    vector_store,             
    vector_store_followups,
    reload_followup_vector_store,
    reload_institutional_context_vector_store,
    get_alumnos_menor_promedio_por_nivel, 
    get_alumnos_observaciones_negativas_por_nivel,
    _extract_level_from_course,
    get_student_vs_course_level_averages,
    get_course_vs_level_comparison_data,
    get_all_courses_in_level_breakdown_data,
    get_course_heatmap_data,
    get_level_kpis,
    get_course_attendance_kpis,
    get_advanced_establishment_alerts,
    generate_intervention_plan_with_gemini,
    save_intervention_plan_to_db,
    get_intervention_plans_for_entity,
    search_web_for_support_resources,
    save_reporte_360_to_db,
    get_historical_reportes_360_for_entity,
    save_observation_for_reporte_360,
    save_observation_for_entity,
    save_data_snapshot_to_db,
    get_student_evolution_summary,
    get_attendance_evolution_summary,
    get_student_qualitative_history,
    get_course_qualitative_summary,
    get_observations_for_reporte_360,
    build_feature_store_from_csv,
    build_feature_store_signals,
    build_risk_summary,
    recommend_interventions_for_student,
    compute_risk_scores_from_feature_store,
    save_intervention_outcome,
    get_intervention_effectiveness_summary,
    predict_risk_with_model_for_fs,
    train_predictive_risk_model,
    get_student_grade_evolution_value,
    get_student_attendance_evolution_value,
    reset_database_tables,
    get_lowest_grade_student_for_subject
)
from .utils import normalize_text, grade_to_qualitative
import sqlite3

main_bp = Blueprint('main', __name__)

# --- INICIO: DEFINICIÓN DE ZONA HORARIA ---
# Mantener constante para compatibilidad, pero usar helper que lee config en tiempo de ejecución.
SANTIAGO_TZ = timezone('America/Santiago')

def _get_tz():
    try:
        return timezone(current_app.config.get('TIMEZONE_NAME', 'America/Santiago'))
    except Exception:
        return SANTIAGO_TZ
# --- FIN: DEFINICIÓN DE ZONA HORARIA ---

# --- Manejador global de errores para API: devuelve JSON en vez de HTML ---
from werkzeug.exceptions import HTTPException

@main_bp.errorhandler(Exception)
def handle_main_bp_exception(e):
    try:
        # Para rutas de API aseguramos respuestas JSON siempre
        if request.path.startswith('/api/'):
            current_app.logger.exception(f"Excepción no controlada en API: {e}")
            if isinstance(e, HTTPException):
                return jsonify({"error": e.description}), e.code
            return jsonify({"error": "Error inesperado en el servidor."}), 500
    except Exception:
        # Si algo falla aquí, devolvemos un 500 genérico en JSON para API
        if request.path.startswith('/api/'):
            return jsonify({"error": "Error inesperado en el servidor."}), 500
    # Para rutas no-API, logeamos el error y redirigimos a la página de error
    current_app.logger.exception(f"Excepción no controlada en ruta no-API: {e}")
    flash(f'Error interno del servidor: {str(e)}', 'danger')
    return redirect(url_for('main.index'))

# --- Utilitario de normalización de texto centralizado en utils.normalize_text ---

def _build_key_docs_block(tipo_entidad: str, nombre_entidad: str, current_filename: str) -> str:
    """Construye un bloque seguro con documentos clave recientes (Reportes 360 y
    Planes de Intervención) para la entidad. Puede incluir extractos configurables.
    """
    try:
        db_path = current_app.config['DATABASE_FILE']
        max_items = int(current_app.config.get('KEY_DOCS_MAX_ITEMS', 2))
        include_excerpts = bool(current_app.config.get('KEY_DOCS_INCLUDE_EXCERPTS', True))
        excerpt_lines = int(current_app.config.get('KEY_DOCS_EXCERPT_LINES', 8))
        excerpt_chars = int(current_app.config.get('KEY_DOCS_EXCERPT_CHARS', 1000))
        log_sections = bool(current_app.config.get('LOG_PROMPT_SECTIONS', True))

        reportes = get_historical_reportes_360_for_entity(db_path, tipo_entidad, nombre_entidad, current_filename) or []
        planes = get_intervention_plans_for_entity(db_path, tipo_entidad, nombre_entidad, current_filename) or []

        # Seleccionar hasta N de cada tipo
        top_reportes = reportes[:max_items]
        top_planes = planes[:max_items]

        bloques = []

        def _mk_excerpt(md: str) -> str:
            if not include_excerpts or not md:
                return ""
            # Limitar por líneas y caracteres para controlar tokens
            lines = md.splitlines()
            first_lines = lines[:excerpt_lines]
            excerpt = "\n".join(first_lines)
            if len(excerpt) > excerpt_chars:
                excerpt = excerpt[:excerpt_chars]
            return f"\nResumen breve:\n---\n{excerpt}\n---"

        for r in top_reportes:
            ts = r.get('timestamp_formateado', r.get('timestamp', ''))
            md = r.get('reporte_markdown')
            entry = f"* Reporte 360 — {ts}{_mk_excerpt(md)}"
            bloques.append(entry)

        for p in top_planes:
            ts = p.get('timestamp', '')
            md = p.get('plan_markdown')
            entry = f"* Plan de Intervención — {ts}{_mk_excerpt(md)}"
            bloques.append(entry)

        if not bloques:
            return ""  # No añadir nada si no hay documentos

        header = "\n\nDocumentos clave recientes (contexto anclado):\n"
        block = header + "\n".join(bloques) + "\n"

        if log_sections:
            try:
                current_app.logger.info(
                    f"Prompt: Documentos clave incluidos — items={len(bloques)}, chars={len(block)}"
                )
            except Exception:
                pass

        return block
    except Exception as e:
        current_app.logger.exception(f"Error al construir bloque de documentos clave: {e}")
        return ""

# --- Helpers de Keywords desde configuración ---
def _get_keywords_from_config(config_key: str, default_list: list) -> list:
    try:
        override = current_app.config.get(config_key)
        if isinstance(override, (list, tuple)) and override:
            return [str(x).lower() for x in override]
    except Exception:
        pass
    return default_list

def get_attendance_keywords():
    return _get_keywords_from_config('ATTENDANCE_KEYWORDS', ATTENDANCE_KEYWORDS)

def get_qualitative_keywords():
    return _get_keywords_from_config('QUALITATIVE_KEYWORDS', QUALITATIVE_KEYWORDS)

def get_evolution_keywords():
    return _get_keywords_from_config('EVOLUTION_KEYWORDS', EVOLUTION_KEYWORDS)

def get_negative_evolution_keywords():
    return _get_keywords_from_config('NEGATIVE_EVOLUTION_KEYWORDS', NEGATIVE_EVOLUTION_KEYWORDS)

# Matcher robusto: normaliza prompt y keywords y detecta ocurrencias
def any_keyword_in_prompt(prompt_text: str, keywords: list) -> bool:
    if not prompt_text or not keywords:
        return False
    pt_norm = normalize_text(prompt_text)
    for kw in keywords:
        if not kw:
            continue
        if normalize_text(kw) in pt_norm:
            return True
    return False

# Aliases de cursos para detección robusta
def generate_course_aliases(course_name: str) -> set:
    import re
    aliases = set()
    if not course_name:
        return aliases
    base_norm = normalize_text(course_name)
    aliases.add(' '.join(base_norm.split()))
    no_ordinal = base_norm.replace('°', '').replace('º', '')
    aliases.add(' '.join(no_ordinal.split()))
    # quitar 'o' ordinal pegado al número (ej. "3o")
    aliases.add(re.sub(r"(\d)\s*o(\s)", r"\1\2", base_norm))
    aliases.add(re.sub(r"(\d)\s*o$", r"\1", base_norm))
    # abreviaciones de nivel
    aliases.add(base_norm.replace(' basico ', ' bas '))
    aliases.add(no_ordinal.replace(' basico ', ' bas '))
    aliases.add(base_norm.replace(' medio ', ' med '))
    aliases.add(no_ordinal.replace(' medio ', ' med '))
    aliases.add(base_norm.replace(' medio ', ' m '))
    aliases.add(no_ordinal.replace(' medio ', ' m '))
    # ordinales a número
    ordinal_map = {
        'primero': '1', 'primer': '1', '1ro': '1', '1ero': '1',
        'segundo': '2', '2do': '2', '2ndo': '2',
        'tercero': '3', '3ro': '3', '3ero': '3',
        'cuarto': '4', '4to': '4', '4to.': '4',
        'quinto': '5', '5to': '5',
        'sexto': '6', '6to': '6',
        'septimo': '7', 'séptimo': '7', '7mo': '7',
        'octavo': '8', '8vo': '8'
    }
    tokens = no_ordinal.split()
    tokens_num = [ordinal_map.get(t, t) for t in tokens]
    aliases.add(' '.join(tokens_num))
    # variantes según paralelo
    if tokens and len(tokens[-1]) == 1 and tokens[-1].isalpha():
        sin_letra = ' '.join(tokens[:-1])
        sin_letra_bas = sin_letra.replace(' basico ', ' bas ')
        sin_letra_med = sin_letra.replace(' medio ', ' med ')
        aliases.add(' '.join(sin_letra.split()))
        aliases.add(' '.join(sin_letra_bas.split()))
        aliases.add(' '.join(sin_letra_med.split()))
        compact = re.sub(r"(\d)\s+(med|m|bas)\s*", r"\1\2 ", ' '.join(tokens_num))
        aliases.add(compact)
        # variantes número + letra
        num_token = None
        for t in tokens_num:
            if re.match(r"^\d+$", t):
                num_token = t
                break
        letter = tokens[-1]
        if num_token:
            aliases.add(f"{num_token} {letter}")
            aliases.add(f"{num_token}{letter}")
            aliases.add(f"{num_token}° {letter}")
            aliases.add(f"{num_token}°{letter}")
            aliases.add((compact + letter).replace(' ', ''))
    # compactar espacios
    aliases = { ' '.join(a.split()) for a in aliases if a }
    return aliases

@main_bp.route('/api/risk_summary', methods=['GET'])
def api_risk_summary():
    try:
        df = get_dataframe_from_session_file()
        if df is None or df.empty:
            return jsonify({'error': 'no_data'}), 400
        fs = build_feature_store_from_csv(df)
        summary = build_risk_summary(fs)
        return jsonify(summary), 200
    except Exception as e:
        current_app.logger.exception(f"Error en /api/risk_summary: {e}")
        return jsonify({'error': 'server_error'}), 500

@main_bp.route('/api/risk_summary_export.csv', methods=['GET'])
def api_risk_summary_export_csv():
    try:
        df = get_dataframe_from_session_file()
        if df is None or df.empty:
            return jsonify({'error': 'no_data'}), 400
        fs = build_feature_store_from_csv(df)
        summary = build_risk_summary(fs)
        import io, csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Curso','Alto','Medio','Bajo'])
        for course, data in (summary.get('by_course') or {}).items():
            writer.writerow([course, data.get('alto',0), data.get('medio',0), data.get('bajo',0)])
        csv_data = output.getvalue()
        resp = Response(csv_data, mimetype='text/csv')
        resp.headers['Content-Disposition'] = 'attachment; filename=risk_summary.csv'
        return resp
    except Exception as e:
        current_app.logger.exception(f"Error en /api/risk_summary_export.csv: {e}")
        return jsonify({'error': 'server_error'}), 500

@main_bp.route('/api/risk_sensitivity', methods=['GET', 'POST'])
def api_risk_sensitivity():
    try:
        if request.method == 'GET':
            defaults = {
                'avg_below_high_threshold': float(current_app.config.get('LOW_PERFORMANCE_THRESHOLD_GRADE', 4.0)),
                'subjects_below_medium_count': 2,
                'neg_observations_medium_count': 2,
                'neg_observations_high_count': 4
            }
            overrides = session.get('risk_sensitivity') or {}
            result = {**defaults, **overrides}
            return jsonify(result), 200
        data = request.json or {}
        if data.get('reset'):
            session.pop('risk_sensitivity', None)
            return jsonify({'message': 'reset'}), 200
        cleaned = {}
        def as_float(v, d):
            try:
                return float(v)
            except Exception:
                return d
        def as_int(v, d):
            try:
                return int(v)
            except Exception:
                return d
        cleaned['avg_below_high_threshold'] = as_float(data.get('avg_below_high_threshold'), float(current_app.config.get('LOW_PERFORMANCE_THRESHOLD_GRADE', 4.0)))
        cleaned['subjects_below_medium_count'] = as_int(data.get('subjects_below_medium_count'), 2)
        cleaned['neg_observations_medium_count'] = as_int(data.get('neg_observations_medium_count'), 2)
        cleaned['neg_observations_high_count'] = as_int(data.get('neg_observations_high_count'), 4)
        session['risk_sensitivity'] = cleaned
        session.modified = True
        return jsonify({'message': 'saved', 'overrides': cleaned}), 200
    except Exception as e:
        current_app.logger.exception(f"Error en /api/risk_sensitivity: {e}")
        return jsonify({'error': 'server_error'}), 500

@main_bp.route('/riesgos', methods=['GET'])
def riesgos():
    try:
        df = get_dataframe_from_session_file()
        if df is None or df.empty:
            flash('No hay datos cargados para analizar riesgos.', 'warning')
            return redirect(url_for('main.index'))
        fs = build_feature_store_from_csv(df)
        summary = build_risk_summary(fs)
        return render_template('riesgos.html', summary=summary)
    except Exception as e:
        current_app.logger.exception(f"Error en /riesgos: {e}")
        flash('Error al construir el resumen de riesgos.', 'danger')
        return redirect(url_for('main.index'))

@main_bp.route('/api/suggest_interventions/alumno/<path:valor_codificado>', methods=['GET'])
def api_suggest_interventions_alumno(valor_codificado):
    try:
        nombre = unquote(valor_codificado)
    except Exception:
        return jsonify({'error': 'bad_request'}), 400
    try:
        df = get_dataframe_from_session_file()
        if df is None or df.empty:
            return jsonify({'error': 'no_data'}), 400
        fs = build_feature_store_from_csv(df)
        suggestions = recommend_interventions_for_student(fs, nombre)
        return jsonify(suggestions), 200
    except Exception as e:
        current_app.logger.exception(f"Error en /api/suggest_interventions/alumno: {e}")
        return jsonify({'error': 'server_error'}), 500

@main_bp.route('/sugerencias/alumno/<path:valor_codificado>', methods=['GET'])
def ver_sugerencias_alumno(valor_codificado):
    try:
        nombre = unquote(valor_codificado)
    except Exception:
        flash('Nombre de alumno inválido.', 'danger')
        return redirect(url_for('main.riesgos'))
    try:
        df = get_dataframe_from_session_file()
        if df is None or df.empty:
            flash('No hay datos cargados para generar sugerencias.', 'warning')
            return redirect(url_for('main.riesgos'))
        fs = build_feature_store_from_csv(df)
        sugg = recommend_interventions_for_student(fs, nombre)
        return render_template('sugerencias_alumno.html', suggestion=sugg)
    except Exception as e:
        current_app.logger.exception(f"Error en ver_sugerencias_alumno: {e}")
        flash('Error al generar sugerencias.', 'danger')
        return redirect(url_for('main.riesgos'))

@main_bp.route('/api/intervention_outcome', methods=['POST'])
def api_intervention_outcome():
    try:
        data = request.json or {}
        follow_up_id = data.get('follow_up_id')
        related_entity_type = data.get('related_entity_type')
        related_entity_name = data.get('related_entity_name')
        course_name = data.get('course_name')
        outcome_date = data.get('outcome_date')
        compliance_pct = data.get('compliance_pct')
        impact_grade_delta = data.get('impact_grade_delta')
        impact_attendance_delta = data.get('impact_attendance_delta')
        notes = data.get('notes')
        db_path = current_app.config['DATABASE_FILE']
        new_id = save_intervention_outcome(db_path, follow_up_id, related_entity_type, related_entity_name, course_name, outcome_date, compliance_pct, impact_grade_delta, impact_attendance_delta, notes)
        if not new_id:
            return jsonify({'error': 'save_failed'}), 500
        return jsonify({'id': int(new_id)}), 201
    except Exception as e:
        current_app.logger.exception(f"Error en /api/intervention_outcome: {e}")
        return jsonify({'error': 'server_error'}), 500

@main_bp.route('/api/intervention_effectiveness', methods=['GET'])
def api_intervention_effectiveness():
    try:
        days = request.args.get('days')
        dw = int(days) if days is not None and str(days).isdigit() else None
        db_path = current_app.config['DATABASE_FILE']
        summary = get_intervention_effectiveness_summary(db_path, dw)
        return jsonify(summary), 200
    except Exception as e:
        current_app.logger.exception(f"Error en /api/intervention_effectiveness: {e}")
        return jsonify({'error': 'server_error'}), 500

@main_bp.route('/riesgos/curso/<level>/<path:course>', methods=['GET'])
def riesgos_por_curso_nivel(level, course):
    try:
        df = get_dataframe_from_session_file()
        if df is None or df.empty:
            flash('No hay datos cargados para analizar riesgos.', 'warning')
            return redirect(url_for('main.index'))
        fs = build_feature_store_from_csv(df)
        scores = compute_risk_scores_from_feature_store(fs)
        level = str(level).lower()
        students = []
        course_norm = normalize_text(course)
        for name, info in (scores or {}).items():
            info_course = normalize_text(str(info.get('course') or ''))
            if info_course == course_norm and str(info.get('level') or '').lower() == level:
                students.append({
                    'name': name,
                    'score': info.get('score', 0),
                    'reasons': info.get('reasons', [])
                })
        students.sort(key=lambda x: x['score'], reverse=True)
        model_preds = {}
        try:
            if bool(current_app.config.get('ENABLE_PREDICTIVE_MODEL_VIS')):
                preds = predict_risk_with_model_for_fs(fs, current_app.config.get('MODEL_ARTIFACTS_DIR'))
                model_preds = preds or {}
        except Exception:
            model_preds = {}
        return render_template('riesgos_nivel_detalle.html', course=course, level=level, students=students, model_preds=model_preds)
    except Exception as e:
        current_app.logger.exception(f"Error en /riesgos/curso/{level}/{course}: {e}")
        flash('Error al construir la vista por nivel de riesgo.', 'danger')
        return redirect(url_for('main.riesgos'))

@main_bp.route('/sugerencias/alumno/<path:valor_codificado>/guardar_borrador', methods=['POST'])
def guardar_borrador_sugerencias_alumno(valor_codificado):
    try:
        nombre = unquote(valor_codificado)
    except Exception:
        flash('Nombre de alumno inválido.', 'danger')
        return redirect(url_for('main.riesgos'))
    try:
        df = get_dataframe_from_session_file()
        if df is None or df.empty:
            flash('No hay datos cargados para guardar borrador.', 'warning')
            return redirect(url_for('main.riesgos'))
        fs = build_feature_store_from_csv(df)
        sugg = recommend_interventions_for_student(fs, nombre)
        parts = []
        parts.append('<h2>Plan de Intervención (Borrador)</h2>')
        parts.append(f"<p><strong>Alumno:</strong> {sugg.get('student','')}</p>")
        parts.append(f"<p><strong>Curso:</strong> {sugg.get('course','')}</p>")
        risk = sugg.get('risk') or {}
        parts.append(f"<p><strong>Nivel de riesgo:</strong> {risk.get('level','')}</p>")
        reasons = risk.get('reasons') or []
        if reasons:
            parts.append('<p><strong>Razones:</strong></p><ul>')
            for r in reasons[:6]:
                parts.append(f"<li>{r}</li>")
            parts.append('</ul>')
        actions = sugg.get('actions') or []
        if actions:
            parts.append('<h3>Acciones Recomendadas</h3>')
            parts.append('<ol>')
            for a in actions:
                title = a.get('title','')
                action = a.get('action','')
                foundation = a.get('foundation','')
                parts.append(f"<li><p><strong>{title}</strong></p><p>{action}</p><p><em>Fundamento:</em> {foundation}</p></li>")
            parts.append('</ol>')
        plan_html = ''.join(parts)
        csv_path = session.get('current_file_path') or ''
        csv_filename = os.path.basename(csv_path) if csv_path else ''
        new_id = save_intervention_plan_to_db(current_app.config['DATABASE_FILE'], csv_filename, 'alumno', nombre, plan_html, 'Borrador generado desde sugerencias')
        try:
            reload_followup_vector_store()
        except Exception:
            pass
        if new_id:
            flash('Borrador de Plan de Intervención guardado.', 'success')
            return redirect(url_for('main.ver_sugerencias_alumno', valor_codificado=valor_codificado, saved='1'))
        else:
            flash('No se pudo guardar el borrador.', 'danger')
            return redirect(url_for('main.ver_sugerencias_alumno', valor_codificado=valor_codificado))
    except Exception as e:
        current_app.logger.exception(f"Error al guardar borrador de sugerencias: {e}")
        flash('Error al guardar borrador.', 'danger')
        return redirect(url_for('main.riesgos'))

def find_course_in_prompt(course_names: list, prompt_text: str) -> str:
    pt_norm = normalize_text(prompt_text)
    for name in course_names:
        for alias in generate_course_aliases(name):
            if alias and alias in pt_norm:
                return name
    return None

def detect_course_or_ambiguity(course_names: list, prompt_text: str):
    """
    Devuelve (resolved_course, ambiguous_level, candidates).
    - resolved_course: nombre del curso cuando hay coincidencia clara.
    - ambiguous_level: cadena del nivel detectado cuando hay múltiples paralelos.
    - candidates: lista de nombres de curso que comparten el nivel.
    """
    pt_norm = normalize_text(prompt_text)
    import re
    # Primero: coincidencia clara con alias completos
    for name in course_names:
        for alias in generate_course_aliases(name):
            if alias and alias in pt_norm:
                # Verificar si alias incluye paralelo
                tokens = alias.split()
                if tokens and len(tokens[-1]) == 1 and tokens[-1].isalpha():
                    return name, None, []
                # Alias sin letra podría ser ambigüo: seguir verificando por nivel
                # Construimos mapa de nivel -> cursos
                break
    # Construir mapa de nivel (sin paralelo) -> cursos
    def strip_parallel(nm: str):
        t = normalize_text(nm).split()
        if t and len(t[-1]) == 1 and t[-1].isalpha():
            t = t[:-1]
        return ' '.join(t)

    level_map = {}
    for nm in course_names:
        lvl = strip_parallel(nm)
        level_map.setdefault(lvl, []).append(nm)

    # Detectar niveles presentes en prompt
    for lvl, courses in level_map.items():
        # considerar alias del nivel también (aplica abreviaciones)
        lvl_aliases = generate_course_aliases(lvl)
        if any(a and a in pt_norm for a in lvl_aliases):
            if len(courses) == 1:
                return courses[0], None, []
            else:
                return None, lvl, courses
    return None, None, []

# --- Rutas Principales ---
@main_bp.route('/')
def index():
    df_global = get_dataframe_from_session_file()
    student_names = []
    course_names = []
    if df_global is not None and not df_global.empty:
        if current_app.config['NOMBRE_COL'] in df_global.columns:
            student_names = sorted(df_global[current_app.config['NOMBRE_COL']].astype(str).unique().tolist())
        if current_app.config['CURSO_COL'] in df_global.columns:
            course_names = sorted(df_global[current_app.config['CURSO_COL']].astype(str).unique().tolist())

    # LÍNEA AÑADIDA: Pasa el objeto 'datetime' a la plantilla
    return render_template('index.html',
                           page_title="Dashboard Principal - TutorIA360",
                           student_names_for_select=student_names,
                           course_names_for_select=course_names,
                           now=datetime.datetime.now())

@main_bp.route('/clear_session_file')
def clear_session_file():
    # No changes to this function
    keys_to_pop = [
        'current_file_path', 'uploaded_filename', 'file_summary', 
        'last_analysis_markdown', 'last_user_prompt', 'chat_history', 
        'advanced_chat_history', 'reporte_360_markdown', 
        'reporte_360_entidad_tipo', 'reporte_360_entidad_nombre',
        'current_reporte_360_id', 
        'current_intervention_plan_html', 'current_intervention_plan_markdown',
        'current_intervention_plan_date', 'current_intervention_plan_for_entity_type',
        'current_intervention_plan_for_entity_name',
        'consumo_sesion', # NUEVO: Limpiar contador de la sesión
        'last_analysis_result' # NUEVO: Limpiar el resultado del último análisis
    ]
    for key in keys_to_pop:
        session.pop(key, None)
    flash('Se ha limpiado la información del archivo anterior y los historiales.', 'info')
    return redirect(url_for('main.index'))

@main_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'datafile' not in request.files:
        flash('No se encontró el archivo en la solicitud.', 'warning')
        return redirect(url_for('main.index'))
    file = request.files['datafile']
    if file.filename == '':
        flash('No seleccionaste ningún archivo.', 'warning')
        return redirect(url_for('main.index'))
    if file:
        filename = secure_filename(file.filename)
        if not filename.lower().endswith('.csv'):
            flash('Error: Solo se permiten archivos CSV.', 'danger')
            return redirect(url_for('main.index'))
        
        clear_session_file() 

        save_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(save_path)
            session['current_file_path'] = save_path
            session['uploaded_filename'] = filename
            
            df = get_dataframe_from_session_file()
            if df is None or df.empty:
                 flash('Error crítico al leer o el archivo CSV está vacío o tiene un formato incorrecto (faltan columnas obligatorias).', 'danger')
                 session.pop('current_file_path', None); session.pop('uploaded_filename', None)
                 return redirect(url_for('main.index'))
            
            # --- INICIO: NUEVA LÓGICA DE CÁLCULO PARA FORMATO "LARGO" ---
            
            # Definir constantes de columnas desde config
            nombre_col = current_app.config['NOMBRE_COL']
            curso_col = current_app.config['CURSO_COL']
            promedio_col = current_app.config['PROMEDIO_COL'] # Esta columna es calculada, no leída
            asistencia_col = current_app.config.get('ASISTENCIA_COL')
            nota_col = current_app.config['NOTA_COL']
            asignatura_col = current_app.config['ASIGNATURA_COL']

            # Crear un DataFrame de estudiantes únicos para cálculos generales
            df_alumnos = df.drop_duplicates(subset=[nombre_col]).copy()

            total_alumnos = len(df_alumnos)
            total_cursos = df_alumnos[curso_col].nunique()
            promedio_general_calculado = df[nota_col].mean()
            
            # --- CÁLCULO PARA GRÁFICO DE DISTRIBUCIÓN CUALITATIVA DE PROMEDIOS ---
            average_distribution_data = {'labels': [], 'counts': []}
            if not df_alumnos[promedio_col].isnull().all():
                promedios_validos = df_alumnos[promedio_col]
                qual_labels = ['Muy Malo', 'Malo', 'Regular', 'Bueno', 'Muy bueno']
                try:
                    qual_series = promedios_validos.apply(grade_to_qualitative).dropna()
                    value_counts = qual_series.value_counts()
                    # Ordenar según las etiquetas deseadas
                    average_distribution_data['labels'] = qual_labels
                    average_distribution_data['counts'] = [int(value_counts.get(lbl, 0)) for lbl in qual_labels]
                except Exception as e:
                    print(f"Error en distribución cualitativa de promedios: {e}")

            # --- CÁLCULO PARA GRÁFICO DE DISTRIBUCIÓN DE ASISTENCIA ---
            attendance_distribution_data = {'labels': [], 'counts': []}
            asistencia_data_available = asistencia_col and asistencia_col in df.columns and not df[asistencia_col].isnull().all()
            if asistencia_data_available:
                asistencias_validas = df_alumnos[asistencia_col] * 100
                bins = [0, 80, 85, 90, 95, 101]; labels = ['<80%', '80-84%', '85-89%', '90-94%', '95-100%']
                try:
                    dist_asistencia_counts = pd.cut(asistencias_validas, bins=bins, labels=labels, right=False, include_lowest=True).value_counts().sort_index(ascending=False)
                    attendance_distribution_data['labels'] = dist_asistencia_counts.index.tolist()
                    attendance_distribution_data['counts'] = dist_asistencia_counts.values.tolist()
                except Exception as e: print(f"Error en rangos de distribución de asistencia: {e}")

            # --- CÁLCULO DE ALUMNO Y CURSO CON MENOR PROMEDIO ---
            alumno_menor_promedio = df_alumnos.loc[df_alumnos[promedio_col].idxmin()]
            
            promedio_por_curso = df.groupby(curso_col)[nota_col].mean().dropna()
            curso_menor_promedio_nombre = promedio_por_curso.idxmin() if not promedio_por_curso.empty else "N/A"
            curso_menor_promedio_valor = promedio_por_curso.min() if not promedio_por_curso.empty else np.nan

            # --- LLAMADAS A FUNCIONES AUXILIARES (QUE TAMBIÉN SERÁN REFACTORIZADAS) ---
            level_kpis_data = get_level_kpis(df) # Usa el df completo
            course_attendance_data = get_course_attendance_kpis(df) # Usa el df completo
            advanced_alerts_data = get_advanced_establishment_alerts(df, level_kpis_data) # Usa el df completo
            
            asistencia_promedio_global = df_alumnos[asistencia_col].mean() if asistencia_data_available else np.nan
            riesgo_threshold = current_app.config.get('LOW_PERFORMANCE_THRESHOLD_GRADE', 4.0)
            estudiantes_en_riesgo_count = len(df_alumnos[df_alumnos[promedio_col] < riesgo_threshold])
            niveles_activos_count = len(level_kpis_data)

            session['file_summary'] = { 
                'total_alumnos': total_alumnos, 
                'total_cursos': total_cursos, 
                'promedio_general': promedio_general_calculado if not pd.isna(promedio_general_calculado) else "N/A", 
                'asistencia_promedio_global': asistencia_promedio_global if not pd.isna(asistencia_promedio_global) else 0,
                'estudiantes_en_riesgo': estudiantes_en_riesgo_count,
                'niveles_activos': niveles_activos_count,
                'column_names': df.columns.tolist(), 
                'average_distribution_data': average_distribution_data,
                'attendance_distribution_data': attendance_distribution_data,
                'level_kpis': level_kpis_data,
                'course_attendance': course_attendance_data,
                'advanced_alerts': advanced_alerts_data,
                'asistencia_data_available': asistencia_data_available,
                'alumno_menor_promedio': {'nombre': alumno_menor_promedio[nombre_col], 'promedio': alumno_menor_promedio[promedio_col]}, 
                'curso_menor_promedio': {'nombre': curso_menor_promedio_nombre, 'promedio': curso_menor_promedio_valor if not pd.isna(curso_menor_promedio_valor) else "N/A"}, 
            }
            # --- FIN: NUEVA LÓGICA DE CÁLCULO ---

            session['consumo_sesion'] = {'total_tokens': 0, 'total_cost': 0.0}
            flash(f'Archivo "{filename}" cargado y procesado con el nuevo formato.', 'success')
            
            # --- INICIO: GUARDAR INSTANTÁNEA EN LA BASE DE DATOS ---
            try:
                db_path = current_app.config['DATABASE_FILE']
                success, message = save_data_snapshot_to_db(df, filename, db_path)
                if success:
                    flash(f"Historial de datos guardado. {message}", 'success')
                else:
                    flash(f"Error al guardar historial de datos: {message}", 'danger')
            except Exception as e_snap:
                flash(f"Error crítico al intentar guardar la instantánea de datos: {e_snap}", 'danger')
                traceback.print_exc()
            # --- FIN: GUARDAR INSTANTÁNEA EN LA BASE DE DATOS ---

            return redirect(url_for('main.index'))
        except Exception as e:
            flash(f'Error al procesar el archivo: {e}', 'danger')
            traceback.print_exc()
            keys_to_pop_on_error = ['current_file_path', 'uploaded_filename', 'file_summary', 'chat_history', 'advanced_chat_history']
            for key in keys_to_pop_on_error: session.pop(key, None)
            return redirect(url_for('main.index'))
    return redirect(url_for('main.index'))

@main_bp.route('/upload_context_pdf', methods=['POST'])
def upload_context_pdf():
    # No changes to this function
    if 'context_pdf_file' not in request.files: flash('No se encontró PDF.', 'warning'); return redirect(url_for('main.index'))
    file = request.files['context_pdf_file']
    if file.filename == '': flash('No seleccionaste PDF.', 'warning'); return redirect(url_for('main.index'))
    if file and file.filename.lower().endswith(('.pdf', '.txt')): 
        filename = secure_filename(file.filename)
        save_path = os.path.join(current_app.config['CONTEXT_DOCS_FOLDER'], filename)
        try:
            file.save(save_path); flash(f'Documento de contexto "{filename}" subido.', 'success')
            if reload_institutional_context_vector_store(current_app.config): 
                flash('Índice de contexto institucional actualizado.', 'info')
            else: 
                flash('Documento subido, pero ocurrió un error al actualizar el índice de contexto. Revise los logs.', 'warning')
        except Exception as e: 
            flash(f'Error al procesar el documento de contexto: {e}', 'danger')
            traceback.print_exc()
    else: flash('Error: Solo se permiten archivos PDF o TXT para el contexto.', 'danger')
    return redirect(url_for('main.index'))

@main_bp.route('/chat_avanzado')
def chat_avanzado():
    try:
        if not session.get('current_file_path'):
            flash('Carga un archivo CSV.', 'warning')
            return redirect(url_for('main.index'))
        hist = session.get('advanced_chat_history')
        if not isinstance(hist, list):
            hist = []
            session['advanced_chat_history'] = hist
        return render_template('advanced_chat.html', page_title="Chat Avanzado", filename=session.get('uploaded_filename'), advanced_chat_history=hist)
    except Exception as e:
        current_app.logger.exception(f"Error inesperado en /chat_avanzado: {e}")
        flash('Error inesperado al cargar Chat Avanzado. Intenta nuevamente.', 'danger')
        return redirect(url_for('main.index'))

@main_bp.route('/analyze', methods=['GET', 'POST']) 
def analyze_page():
    if not session.get('current_file_path'): flash('Carga un archivo CSV.', 'warning'); return redirect(url_for('main.index'))
    vs_inst_local, vs_followup_local, load_error = vector_store, vector_store_followups, False
    if not embedding_model_instance: 
        load_error = True
        flash("Error crítico: El modelo de embeddings no está disponible. El análisis contextual puede fallar.", 'danger')

    if request.method == 'POST':
        user_prompt = request.form.get('user_prompt', '')
        session['last_user_prompt'] = user_prompt
        
        # Inicializar el contador de sesión si no existe
        if 'consumo_sesion' not in session:
            session['consumo_sesion'] = {'total_tokens': 0, 'total_cost': 0.0}

        data_string = load_data_as_string(session.get('current_file_path'))
        chat_history_for_prompt = format_chat_history_for_prompt(session.get('chat_history', []))

        analysis_result = analyze_data_with_gemini(
            data_string, 
            user_prompt, 
            vs_inst_local, 
            vs_followup_local, 
            chat_history_string=chat_history_for_prompt,
            entity_type=None,
            entity_name=None
        )

        # Actualizar el historial de chat y el contador de la sesión
        if not analysis_result.get('error'):
            chat_history = session.get('chat_history', [])
            chat_history.append({
                'user': user_prompt, 
                'gemini_markdown': analysis_result['raw_markdown'], 
                'gemini_html': analysis_result['html_output']
            })
            session['chat_history'] = chat_history[-current_app.config.get('MAX_CHAT_HISTORY_SESSION_STORAGE', 10):]
            
            # Acumular el consumo en la sesión
            session['consumo_sesion']['total_tokens'] += analysis_result.get('total_tokens', 0)
            session['consumo_sesion']['total_cost'] += analysis_result.get('total_cost', 0)

        # Guardar el resultado completo para la página de resultados
        session['last_analysis_result'] = analysis_result
        session.modified = True
        return redirect(url_for('main.show_results'))

    chat_history_display = []
    if 'chat_history' in session:
        display_history = session['chat_history'][-current_app.config.get('MAX_CHAT_HISTORY_DISPLAY_ON_ANALYZE', 3):]
        for entry in display_history:
            gemini_html_content = entry.get('gemini_html', markdown.markdown(entry.get('gemini_markdown',''), extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists']))
            chat_history_display.append({'user': entry['user'], 'gemini': gemini_html_content})
    
    df_global = get_dataframe_from_session_file()
    student_names = []
    course_names = []
    if df_global is not None and not df_global.empty:
        if current_app.config['NOMBRE_COL'] in df_global.columns:
            student_names = sorted(df_global[current_app.config['NOMBRE_COL']].astype(str).unique().tolist())
        if current_app.config['CURSO_COL'] in df_global.columns:
            course_names = sorted(df_global[current_app.config['CURSO_COL']].astype(str).unique().tolist())
            
    return render_template('analyze.html', 
                           filename=session.get('uploaded_filename'), 
                           chat_history=chat_history_display,
                           student_names_for_select=student_names, # Pass for potential future use in analyze page
                           course_names_for_select=course_names)   # Pass for potential future use

@main_bp.route('/results')
def show_results():
    # MODIFIED: Pass student and course names for the entity selection in follow-up form
    df_global = get_dataframe_from_session_file()
    student_names = []
    course_names = []
    if df_global is not None and not df_global.empty:
        if current_app.config['NOMBRE_COL'] in df_global.columns:
            student_names = sorted(df_global[current_app.config['NOMBRE_COL']].astype(str).unique().tolist())
        if current_app.config['CURSO_COL'] in df_global.columns:
            course_names = sorted(df_global[current_app.config['CURSO_COL']].astype(str).unique().tolist())

    # NUEVO: Procesar el resultado completo del análisis desde la sesión
    analysis_result = session.get('last_analysis_result', {})
    analysis_html = analysis_result.get('html_output', "<p>No hay análisis disponible.</p>")
    
    chat_history_template = []
    if 'chat_history' in session:
        for entry in session.get('chat_history', []):
            gemini_html_content = entry.get('gemini_html')
            if not gemini_html_content and entry.get('gemini_markdown'): gemini_html_content = markdown.markdown(entry.get('gemini_markdown',''), extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists'])
            elif not gemini_html_content: gemini_html_content = "<p><em>Respuesta no disponible.</em></p>"
            chat_history_template.append({'user': entry['user'], 'gemini': gemini_html_content})
    
    follow_ups_list = []
    current_filename = session.get('uploaded_filename')
    if current_filename and current_filename != 'N/A': 
        try:
            with sqlite3.connect(current_app.config['DATABASE_FILE']) as conn:
                conn.row_factory = sqlite3.Row; cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, follow_up_comment, follow_up_type, related_entity_type, related_entity_name 
                    FROM follow_ups 
                    WHERE related_filename = ? AND (follow_up_type = 'general_comment' OR related_entity_type IS NOT NULL)
                    ORDER BY timestamp DESC
                """, (current_filename,))
                follow_ups_list = [dict(row) for row in cursor.fetchall()]
        except Exception as e: flash(f"Error al cargar seguimientos para {current_filename}: {e}", "warning"); traceback.print_exc()
    
    return render_template('results.html', 
                           analysis=analysis_html,
                           analysis_result=analysis_result, # NUEVO: Pasar el diccionario completo
                           filename=current_filename if current_filename else 'N/A', 
                           prompt=session.get('last_user_prompt', ''), 
                           chat_history=chat_history_template, 
                           follow_ups=follow_ups_list,
                           student_names_for_select=student_names,
                           course_names_for_select=course_names)

@main_bp.route('/add_follow_up', methods=['POST'])
def add_follow_up():
    filename = session.get('uploaded_filename')
    if not filename: flash('No hay archivo activo para añadir seguimiento.', 'warning'); return redirect(url_for('main.index'))
    
    comment = request.form.get('follow_up_comment')
    # MODIFIED: Get entity type and name from form
    related_entity_type = request.form.get('related_entity_type')
    related_entity_name = request.form.get('related_entity_name')

    # Normalize empty strings to None for DB
    if not related_entity_type or related_entity_type == "none": related_entity_type = None
    if not related_entity_name: related_entity_name = None
    
    # If one is provided, the other should ideally be too, but allow flexibility for now
    # Or enforce that if type is provided, name must be too.
    if related_entity_type and not related_entity_name:
        flash('Si selecciona un tipo de entidad, debe especificar el nombre.', 'warning')
        return redirect(url_for('main.show_results'))
    if not related_entity_type and related_entity_name: # Should not happen if UI is correct
        flash('Se especificó un nombre de entidad sin tipo. Seleccione el tipo.', 'warning')
        return redirect(url_for('main.show_results'))


    if not comment: flash('El comentario de seguimiento no puede estar vacío.', 'warning')
    else:
        try:
            last_prompt = session.get('last_user_prompt', 'Prompt no disponible')
            last_analysis_md = session.get('last_analysis_markdown', 'Análisis no disponible')
            
            with sqlite3.connect(current_app.config['DATABASE_FILE']) as conn:
                # MODIFIED: Add related_entity_type and related_entity_name to INSERT
                conn.cursor().execute('''INSERT INTO follow_ups 
                                         (related_filename, related_prompt, related_analysis, follow_up_comment, 
                                          follow_up_type, related_entity_type, related_entity_name) 
                                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                               (filename, last_prompt, last_analysis_md, comment, 
                                'general_comment', related_entity_type, related_entity_name)) # 'general_comment' type is kept, but now can have entity specifics
            flash('Seguimiento guardado exitosamente.', 'success')
            if embedding_model_instance and reload_followup_vector_store(current_app.config): flash('Índice de seguimientos actualizado.', 'info')
            elif not embedding_model_instance: flash('Seguimiento guardado, pero el modelo de embeddings no está disponible para actualizar el índice.', 'warning')
            else: flash('Seguimiento guardado, pero ocurrió un error al actualizar el índice de seguimientos.', 'warning')
        except Exception as e: flash(f'Error al guardar el seguimiento: {e}', 'danger'); traceback.print_exc()
    return redirect(url_for('main.show_results'))

# --- INICIO: DEFINIR PALABRAS CLAVE PARA DETECCIÓN DE INTENCIÓN ---
EVOLUTION_KEYWORDS = [
    'evolución', 'mejora', 'mejorado', 'empeorado', 'historial', 
    'comparar', 'progresado', 'avanzado', 'cambiado', 'rendimiento histórico',
    'desempeño histórico', 
    'variación', 'bajado', 'subido', 'aumentado', 'disminuido', 
    'datos históricos', 'diferentes datos cargados', 'listar notas'
]

NEGATIVE_EVOLUTION_KEYWORDS = [
    'peor', 'bajado', 'disminuido', 'caída', 'hacia abajo', 
    'empeorado', 'regresión', 'menor rendimiento', 'menor variación'
]

QUALITATIVE_KEYWORDS = [
    'comportamiento', 'conducta', 'actitud', 'disciplina', 
    'observaciones', 'observación', 'entrevista', 'entrevistas', 
    'familia', 'apoderado', 'anotaciones'
]

# Nuevo grupo: palabras clave de asistencia
ATTENDANCE_KEYWORDS = [
    'asistencia', 'inasistencia', 'ausencias', 'presentismo', 'falta', 'faltas'
]

# --- Rutas de API ---
@main_bp.route('/api/alertas/menor_promedio_niveles')
def api_alertas_menor_promedio_niveles():
    # No changes to this function
    df = get_dataframe_from_session_file()
    if df is None or df.empty: return jsonify({"error": "No hay datos cargados o el archivo está vacío."}), 400
    try:
        alertas_promedio = get_alumnos_menor_promedio_por_nivel(df)
        if not alertas_promedio: return jsonify({"message": "No se encontraron datos para generar alertas de promedio por nivel o faltan columnas."}), 200
        return jsonify(alertas_promedio)
    except Exception as e: traceback.print_exc(); return jsonify({"error": f"Error interno al generar alerta de promedios: {str(e)}"}), 500

@main_bp.route('/api/alertas/observaciones_negativas_niveles')
def api_alertas_observaciones_negativas_niveles():
    # No changes to this function
    df = get_dataframe_from_session_file()
    if df is None or df.empty: return jsonify({"error": "No hay datos cargados o el archivo está vacío."}), 400
    observaciones_col = current_app.config.get('OBSERVACIONES_COL')
    if not observaciones_col or observaciones_col not in df.columns: return jsonify({"error": f"La columna de observaciones '{observaciones_col}' no se encuentra en el archivo o no está configurada."}), 400
    try:
        alertas_observaciones = get_alumnos_observaciones_negativas_por_nivel(df)
        if not alertas_observaciones: return jsonify({"message": "No se encontraron alumnos con observaciones de conducta críticas según los criterios definidos."}), 200
        return jsonify(alertas_observaciones)
    except Exception as e: traceback.print_exc(); return jsonify({"error": f"Error interno al generar alerta de observaciones: {str(e)}"}), 500

# --- Nueva vista: Tarjetas de alumnos con menor promedio por nivel ---
@main_bp.route('/alertas/alumnos-menor-promedio')
def alertas_alumnos_menor_promedio():
    """Muestra una vista en el panel central con tarjetas por alumno (agrupadas por nivel).
    Cada tarjeta enlaza al dashboard del alumno. No modifica la API existente ni otras vistas.
    """
    if not session.get('current_file_path'):
        flash('Carga un archivo CSV.', 'warning')
        return redirect(url_for('main.index'))

    df = get_dataframe_from_session_file()
    if df is None or df.empty:
        flash('No hay datos disponibles para mostrar alertas.', 'warning')
        return redirect(url_for('main.index'))

    try:
        alertas_promedio = get_alumnos_menor_promedio_por_nivel(df) or {}
        # Si la función devolvió un error, vaciar y avisar
        if isinstance(alertas_promedio, dict) and 'error' in alertas_promedio:
            flash('No fue posible calcular alumnos con menor promedio por nivel. Revisa el archivo CSV y columnas de configuración.', 'warning')
            alertas_promedio = {}

        # Filtrar niveles que tengan listas válidas de alumnos
        alertas_promedio = {k: v for k, v in alertas_promedio.items() if isinstance(v, list) and len(v) > 0}
        # Orden explícito de cursos: 1° Básico → 8° Básico → 1° Medio → 4° Medio
        orden_basica = [f"{i}° Básico" for i in range(1, 9)]
        orden_media = [f"{i}° Medio" for i in range(1, 5)]
        course_order = orden_basica + orden_media

        # Normalización robusta para empatar claves del dict aunque varíe símbolo °/º o tildes/caso
        def _norm_level(s: str) -> str:
            if not s:
                return ""
            s = str(s).replace('º', '°')
            s_nfkd = unicodedata.normalize('NFKD', s)
            s_ascii = ''.join(c for c in s_nfkd if not unicodedata.combining(c))
            s_ascii = s_ascii.lower().replace('°', '')
            return ' '.join(s_ascii.split())

        ordered_levels = []
        if alertas_promedio:
            course_order_norm = [_norm_level(x) for x in course_order]
            present_levels_map = { _norm_level(k): k for k in alertas_promedio.keys() }
            ordered_levels = [present_levels_map[n] for n in course_order_norm if n in present_levels_map]
            # Añadir niveles extras que no estén en el orden predefinido, manteniendo estabilidad
            extras = [orig for n, orig in present_levels_map.items() if n not in course_order_norm]
            ordered_levels.extend(extras)
    except Exception as e:
        traceback.print_exc()
        flash(f'Error al calcular alumnos con menor promedio: {e}', 'danger')
        return redirect(url_for('main.index'))

    return render_template(
        'alumnos_menor_promedio_cards.html',
        page_title='Alumnos con menor promedio por nivel',
        alertas_promedio=alertas_promedio,
        ordered_levels=ordered_levels,
        course_order=course_order,
        filename=session.get('uploaded_filename', 'N/A')
    )

# --- Nueva vista: Tarjetas de alumnos con observaciones críticas por nivel ---
@main_bp.route('/alertas/alumnos-observaciones-criticas')
def alertas_alumnos_observaciones_criticas():
    """Muestra tarjetas por alumno con observaciones críticas, agrupadas por nivel.
    Cada tarjeta enlaza al dashboard del alumno.
    """
    if not session.get('current_file_path'):
        flash('Carga un archivo CSV.', 'warning')
        return redirect(url_for('main.index'))

    df = get_dataframe_from_session_file()
    if df is None or df.empty:
        flash('No hay datos disponibles para mostrar alertas.', 'warning')
        return redirect(url_for('main.index'))

    try:
        alertas_observaciones = get_alumnos_observaciones_negativas_por_nivel(df) or {}
        # Manejar posible estructura de error
        if isinstance(alertas_observaciones, dict) and 'error' in alertas_observaciones:
            flash('No fue posible calcular alumnos con observaciones críticas por nivel. Revisa el archivo CSV y columnas de configuración.', 'warning')
            alertas_observaciones = {}

        # Filtrar niveles con listas válidas
        alertas_observaciones = {k: v for k, v in alertas_observaciones.items() if isinstance(v, list) and len(v) > 0}

        # Orden explícito de cursos
        orden_basica = [f"{i}° Básico" for i in range(1, 9)]
        orden_media = [f"{i}° Medio" for i in range(1, 5)]
        course_order = orden_basica + orden_media

        # Normalización de niveles (mismo criterio que en menor promedio)
        def _norm_level(s: str) -> str:
            if not s:
                return ""
            s = str(s).replace('º', '°')
            s_nfkd = unicodedata.normalize('NFKD', s)
            s_ascii = ''.join(c for c in s_nfkd if not unicodedata.combining(c))
            s_ascii = s_ascii.lower().replace('°', '')
            return ' '.join(s_ascii.split())

        ordered_levels = []
        if alertas_observaciones:
            course_order_norm = [_norm_level(x) for x in course_order]
            present_levels_map = { _norm_level(k): k for k in alertas_observaciones.keys() }
            ordered_levels = [present_levels_map[n] for n in course_order_norm if n in present_levels_map]
            extras = [orig for n, orig in present_levels_map.items() if n not in course_order_norm]
            ordered_levels.extend(extras)
    except Exception as e:
        traceback.print_exc()
        flash(f'Error al calcular alumnos con observaciones críticas: {e}', 'danger')
        return redirect(url_for('main.index'))

    return render_template(
        'alumnos_observaciones_criticas_cards.html',
        page_title='Alumnos con observaciones críticas por nivel',
        alertas_observaciones=alertas_observaciones,
        ordered_levels=ordered_levels,
        course_order=course_order,
        filename=session.get('uploaded_filename', 'N/A')
    )

@main_bp.route('/api/get_courses')
def api_get_courses(): 
    # No changes to this function
    df = get_dataframe_from_session_file()
    if df is None or df.empty: return jsonify([]) 
    curso_col_const = current_app.config['CURSO_COL']
    if curso_col_const not in df.columns: return jsonify([])
    try: 
        courses = sorted(df[curso_col_const].astype(str).fillna('N/A').unique().tolist())
        return jsonify(courses)
    except Exception as e: traceback.print_exc(); return jsonify({"error": "Error obteniendo la lista de cursos."}), 500

@main_bp.route('/api/search_students')
def api_search_students(): 
    # No changes to this function
    term = request.args.get('term', '').strip().lower()
    if len(term) < 2: return jsonify([])
    df = get_dataframe_from_session_file()
    if df is None or df.empty: return jsonify([])
    nombre_col_const = current_app.config['NOMBRE_COL']
    if nombre_col_const not in df.columns: return jsonify([])
    try:
        student_column = df[nombre_col_const].astype(str).fillna('').str.lower()
        mask = student_column.str.contains(term, na=False)
        coincidentes = df[mask][nombre_col_const].unique().tolist() 
        return jsonify(coincidentes[:10]) 
    except Exception as e: traceback.print_exc(); return jsonify({"error": "Error procesando la búsqueda de estudiantes."}), 500

# --- Ruta de Detalle ---
@main_bp.route('/detalle/<tipo_entidad>/<path:valor_codificado>', methods=['GET'])
def detalle_entidad(tipo_entidad, valor_codificado): 
    try: 
        valor = unquote(valor_codificado) 
    except Exception as e: 
        flash('Valor de entidad no valido.', 'danger')
        return redirect(url_for('main.index'))
    
    if not session.get('current_file_path'): 
        flash('Carga un archivo CSV primero.', 'warning')
        return redirect(url_for('main.index'))
    
    df_original = get_dataframe_from_session_file() 
    if df_original is None or df_original.empty: 
        flash('No se pudo cargar el DataFrame o esta vacio.', 'danger')
        return redirect(url_for('main.index'))
    
    # --- (El resto de la lógica de sesión y BBDD no cambia) ---
    reporte_360_disponible_para_plan = (session.get('reporte_360_markdown') and
        session.get('reporte_360_entidad_tipo') == tipo_entidad and
        session.get('reporte_360_entidad_nombre') == valor)
    historial_planes = get_intervention_plans_for_entity(
        db_path=current_app.config['DATABASE_FILE'], tipo_entidad=tipo_entidad, nombre_entidad=valor,
        current_filename=session.get('uploaded_filename', 'N/A'))

    # Cargar historial de Reportes 360 y sus observaciones asociadas
    reportes_360_con_observaciones = []
    try:
        db_path = current_app.config['DATABASE_FILE']
        current_filename = session.get('uploaded_filename', 'N/A')
        reportes_360 = get_historical_reportes_360_for_entity(
            db_path=db_path,
            tipo_entidad=tipo_entidad,
            nombre_entidad=valor,
            current_filename=current_filename
        )
        for rep in reportes_360:
            observaciones = get_observations_for_reporte_360(db_path, rep.get('id'))
            rep_item = dict(rep)
            rep_item['observaciones'] = observaciones
            reportes_360_con_observaciones.append(rep_item)
    except Exception:
        traceback.print_exc()

    chat_history_key = f'chat_history_detalle_{tipo_entidad}_{valor}' 
    context = { 
        'tipo_entidad': tipo_entidad, 'nombre_entidad': valor, 'filename': session.get('uploaded_filename', 'N/A'), 
        'datos_dashboard': {}, 'error_message': None, 'chat_history_detalle': session.get(chat_history_key, []), 
        'reporte_360_disponible_para_plan': reporte_360_disponible_para_plan,
        'historial_planes_intervencion': historial_planes,
        'historial_reportes_360_con_observaciones': reportes_360_con_observaciones
    }
    
    # --- INICIO: NUEVA LÓGICA DE CÁLCULO PARA DETALLE ---
    nombre_col = current_app.config['NOMBRE_COL']; curso_col = current_app.config['CURSO_COL']
    promedio_col = current_app.config['PROMEDIO_COL']; asignatura_col = current_app.config['ASIGNATURA_COL']
    nota_col = current_app.config['NOTA_COL']
    
    try:
        valor_normalizado = valor.strip().lower()
        if tipo_entidad == 'alumno':
            # Filtrar todas las notas del alumno
            datos_alumno_df = df_original[df_original[nombre_col].astype(str).str.strip().str.lower() == valor_normalizado]
            if not datos_alumno_df.empty:
                # Obtener datos generales de la primera fila (se repiten)
                alumno_data_row = datos_alumno_df.drop_duplicates(subset=[nombre_col]).iloc[0]
                alumno_promedio_general = alumno_data_row.get(promedio_col)
                nombre_curso_alumno = str(alumno_data_row.get(curso_col, 'N/A'))
                
                # Encontrar la peor asignatura del alumno
                promedios_asignaturas_alumno = datos_alumno_df.groupby(asignatura_col)[nota_col].mean()
                peor_asignatura_key = promedios_asignaturas_alumno.idxmin()
                peor_asignatura_nota = promedios_asignaturas_alumno.min()
                
                # Formateo de presentación para peor asignatura del alumno (1 decimal, coma)
                try:
                    from .filters import nota_un_decimal
                    peor_asignatura_presentacion = f"{peor_asignatura_key} ({nota_un_decimal(peor_asignatura_nota)})"
                except Exception:
                    # Fallback si el filtro no está disponible
                    peor_asignatura_presentacion = f"{peor_asignatura_key} ({peor_asignatura_nota:.1f})"

                context['datos_dashboard']['info_general'] = {
                    'Promedio': f"{alumno_promedio_general:.2f}",
                    'Curso': nombre_curso_alumno,
                    'Edad': alumno_data_row.get(current_app.config.get('EDAD_COL'), 'N/A'),
                    'AsignaturaMenorPromedio': peor_asignatura_presentacion
                }
                
                # Gráfico de calificaciones individuales del alumno
                notas_chart = promedios_asignaturas_alumno.round(1)
                context['datos_dashboard']['notas_asignaturas_original'] = {'labels': notas_chart.index.tolist(), 'scores': notas_chart.values.tolist()}
                
                # Gráfico comparativo (ahora llama a la función refactorizada)
                context['datos_dashboard']['student_vs_course_level_chart_data'] = get_student_vs_course_level_averages(df_original, valor, nombre_curso_alumno)

        elif tipo_entidad == 'curso':
            datos_curso_df = df_original[df_original[curso_col].astype(str).str.strip().str.lower() == valor_normalizado]
            if not datos_curso_df.empty:
                df_alumnos_unicos_curso = datos_curso_df.drop_duplicates(subset=[nombre_col])

                # KPIs del curso
                promedio_general_curso = datos_curso_df[nota_col].mean()
                promedios_por_asignatura = datos_curso_df.groupby(asignatura_col)[nota_col].mean()
                peor_asignatura = promedios_por_asignatura.idxmin()
                peor_asignatura_prom = promedios_por_asignatura.min()
                
                alumno_peor_promedio = df_alumnos_unicos_curso.loc[df_alumnos_unicos_curso[promedio_col].idxmin()]

                # Formateo de presentación para peor asignatura del curso (1 decimal, coma)
                try:
                    from .filters import nota_un_decimal
                    peor_asignatura_curso_presentacion = f"{peor_asignatura} ({nota_un_decimal(peor_asignatura_prom)})"
                except Exception:
                    peor_asignatura_curso_presentacion = f"{peor_asignatura} ({peor_asignatura_prom:.1f})"

                context['datos_dashboard']['info_general'] = {
                    'NumeroAlumnos': len(df_alumnos_unicos_curso),
                    'PromedioGeneralCurso': f"{promedio_general_curso:.2f}",
                    'PeorAsignatura': peor_asignatura_curso_presentacion,
                    'AlumnoPeorPromedioNombre': alumno_peor_promedio[nombre_col],
                    'AlumnoPeorPromedioValor': f"{alumno_peor_promedio[promedio_col]:.2f}"
                }

                # Gráfico de distribución de promedios (usando alumnos únicos)
                promedios_curso = df_alumnos_unicos_curso[promedio_col]
                bins = [1.0, 3.99, 4.99, 5.99, 7.01]; labels_bins = ['< 4.0 (R)', '4.0-4.9 (S)', '5.0-5.9 (B)', '6.0-7.0 (MB)']
                dist_prom = pd.cut(promedios_curso, bins=bins, labels=labels_bins, right=True, include_lowest=True).value_counts().sort_index()
                context['datos_dashboard']['distribucion_promedios_curso'] = {'labels': dist_prom.index.tolist(), 'counts': dist_prom.values.tolist()}

                # Gráficos comparativos (llaman a las funciones refactorizadas)
                context['datos_dashboard']['course_vs_level_chart_data'] = get_course_vs_level_comparison_data(df_original, valor)
                current_course_level = _extract_level_from_course(valor)
                context['datos_dashboard']['all_courses_in_level_breakdown_data'] = get_all_courses_in_level_breakdown_data(df_original, current_course_level)
                
                # Heatmap
                context['datos_dashboard']['heatmap_data'] = get_course_heatmap_data(datos_curso_df, nombre_col, asignatura_col, nota_col)
        
    except KeyError as ke:
        traceback.print_exc()
        context['error_message'] = f"Error de datos: una columna esperada ('{ke}') no se encontro. Verifique el archivo CSV y la configuracion."
    except Exception as e:
        traceback.print_exc()
        context['error_message'] = "Ocurrio un error inesperado al obtener los datos para el dashboard."
        
    return render_template('detalle_dashboard.html', **context)

# --- Rutas de API para Chat ---
@main_bp.route('/api/detalle_chat', methods=['POST'])
def api_detalle_chat():
    data = request.json
    tipo_entidad = data.get('tipo_entidad')
    nombre_entidad = data.get('nombre_entidad')
    user_prompt = data.get('prompt')

    if not all([tipo_entidad, nombre_entidad, user_prompt]):
        return jsonify({"error": "Faltan parámetros."}), 400

    df_global = get_dataframe_from_session_file()
    if df_global is None or df_global.empty:
        return jsonify({"error": "No se pudo cargar el DataFrame."}), 500

    prompt_lower = user_prompt.lower()
    prompt_norm = normalize_text(user_prompt)
    ambiguity_message = ""
    prompt_norm = normalize_text(user_prompt)
    prompt_norm = normalize_text(user_prompt)
    
    # --- Lógica de Resumen Cuantitativo (Notas) ---
    historical_data_summary = ""
    if tipo_entidad == 'alumno' and any_keyword_in_prompt(user_prompt, get_evolution_keywords()):
        try:
            current_app.logger.info(f"Detectada intención de evolución CUANTITATIVA para: {nombre_entidad}")
            db_path = current_app.config['DATABASE_FILE']
            order_dir = 'ASC' if any_keyword_in_prompt(user_prompt, get_negative_evolution_keywords()) else 'DESC'
            
            historical_data_summary = get_student_evolution_summary(
                db_path, 
                entity_name=nombre_entidad,
                order_direction=order_dir
            )
        except Exception as e:
            current_app.logger.error(f"Error al obtener resumen de evolución de notas para {nombre_entidad}: {e}")
            historical_data_summary = f"Error al consultar historial de notas: {e}"

    # --- NUEVO: Evolución de Asistencia por Alumno ---
    if tipo_entidad == 'alumno' and any_keyword_in_prompt(user_prompt, get_attendance_keywords()) and current_app.config.get('ENABLE_ATTENDANCE_SUMMARY', True):
        try:
            current_app.logger.info(f"Detectada intención de evolución de ASISTENCIA para: {nombre_entidad}")
            db_path = current_app.config['DATABASE_FILE']
            order_dir_att = 'ASC' if any_keyword_in_prompt(user_prompt, get_negative_evolution_keywords()) else 'DESC'
            attendance_summary = get_attendance_evolution_summary(
                db_path,
                entity_name=nombre_entidad,
                order_direction=order_dir_att
            )
            historical_data_summary = (historical_data_summary + "\n\n" + attendance_summary).strip()
        except Exception as e:
            current_app.logger.error(f"Error al obtener resumen de asistencia para {nombre_entidad}: {e}")
            # No sobreescribir el resumen existente; solo añadir mensaje
            historical_data_summary = (historical_data_summary + f"\n\nError asistencia: {e}").strip()
            
    # --- INICIO: NUEVA LÓGICA DE RESUMEN CUALITATIVO (Comportamiento) ---
    qualitative_history_summary = ""
    if tipo_entidad == 'alumno' and any_keyword_in_prompt(user_prompt, get_qualitative_keywords()) and current_app.config.get('ENABLE_QUALITATIVE_SUMMARY', True):
        try:
            current_app.logger.info(f"Detectada intención de evolución CUALITATIVA para: {nombre_entidad}")
            db_path = current_app.config['DATABASE_FILE']
            qualitative_history_summary = get_student_qualitative_history(
                db_path, 
                student_name=nombre_entidad
            )
        except Exception as e:
            current_app.logger.error(f"Error al obtener resumen de evolución cualitativa para {nombre_entidad}: {e}")
            qualitative_history_summary = f"Error al consultar historial cualitativo: {e}"
    elif tipo_entidad == 'curso' and any_keyword_in_prompt(user_prompt, get_qualitative_keywords()) and current_app.config.get('ENABLE_QUALITATIVE_SUMMARY', True):
        try:
            current_app.logger.info(f"Detectada intención de resumen CUALITATIVO agregado para curso: {nombre_entidad}")
            db_path = current_app.config['DATABASE_FILE']
            qualitative_history_summary = get_course_qualitative_summary(
                db_path,
                course_name=nombre_entidad,
                max_entries=current_app.config.get('MAX_QUALITATIVE_ENTRIES', 30)
            )
        except Exception as e:
            current_app.logger.error(f"Error al obtener resumen cualitativo por curso {nombre_entidad}: {e}")
            qualitative_history_summary = f"Error al consultar resumen cualitativo del curso: {e}"

    # Adjuntar documentos clave recientes al bloque cualitativo
    include_docs = current_app.config.get('INCLUDE_KEY_DOCS_IN_PROMPT', True)
    key_docs_block = _build_key_docs_block(tipo_entidad, nombre_entidad, session.get('current_file_path')) if include_docs else ""
    if key_docs_block:
        qualitative_history_summary = (qualitative_history_summary + key_docs_block).strip()
    # --- FIN: NUEVA LÓGICA ---

    df_entidad = pd.DataFrame()
    # ... (lógica de Motor de Intenciones para respuestas directas no cambia) ...
    try:
        nombre_entidad_normalizado = nombre_entidad.strip().lower()
        if tipo_entidad == 'curso':
            df_entidad = df_global[df_global[current_app.config['CURSO_COL']].astype(str).str.strip().str.lower() == nombre_entidad_normalizado]
        elif tipo_entidad == 'alumno':
            df_entidad = df_global[df_global[current_app.config['NOMBRE_COL']].astype(str).str.strip().str.lower() == nombre_entidad_normalizado]
    except Exception as e:
        return jsonify({"error": f"Error al filtrar datos para la entidad: {str(e)}"}), 500

    if not df_entidad.empty:
        # ... (código del motor de intenciones para 'mejor alumno', 'peor asignatura', 'promedio', etc.) ...
        # (Este código no se ha modificado)
        pass # Dejamos que el código existente del motor de intenciones se ejecute
        
    # Si ninguna intención simple coincide, se procede con la consulta a Gemini.
    data_string_especifico = load_data_as_string(session.get('current_file_path'), specific_entity_df=df_entidad)
    chat_history_key = f'chat_history_detalle_{tipo_entidad}_{nombre_entidad}'
    chat_history_list_detalle = session.get(chat_history_key, [])
    chat_history_string_detalle = format_chat_history_for_prompt(chat_history_list_detalle)
    
    analysis_result = analyze_data_with_gemini(
        data_string_especifico, user_prompt, vector_store, vector_store_followups,
        chat_history_string=chat_history_string_detalle, is_direct_chat_query=True,
        entity_type=tipo_entidad, entity_name=nombre_entidad,
        historical_data_summary_string=historical_data_summary, # <-- Pasa el resumen de NOTAS
        qualitative_history_summary_string=qualitative_history_summary # <-- Pasa el resumen CUALITATIVO
    )
    
    # ... (lógica de guardado de historial de chat y consumo no cambia) ...
    if not analysis_result.get('error'):
        chat_history_list_detalle.append({'user': user_prompt, 'gemini_markdown': analysis_result['raw_markdown'], 'gemini_html': analysis_result['html_output']})
        session[chat_history_key] = chat_history_list_detalle[-current_app.config.get('MAX_CHAT_HISTORY_SESSION_STORAGE', 10):]
        session.setdefault('consumo_sesion', {'total_tokens': 0, 'total_cost': 0.0})
        session['consumo_sesion']['total_tokens'] += analysis_result.get('total_tokens', 0)
        session['consumo_sesion']['total_cost'] += analysis_result.get('total_cost', 0)
        analysis_result['consumo_sesion'] = session['consumo_sesion']
        session.modified = True
    return jsonify(analysis_result)

def crear_respuesta_directa(texto_markdown):
    """Función auxiliar para construir el objeto JSON de respuesta directa."""
    return {
        'html_output': markdown.markdown(texto_markdown),
        'raw_markdown': texto_markdown,
        'model_name': 'Cálculo Directo del Servidor',
        'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
        'input_cost': 0.0, 'output_cost': 0.0, 'total_cost': 0.0,
        'consumo_sesion': session.get('consumo_sesion', {'total_tokens': 0, 'total_cost': 0.0}),
        'error': None
    }

@main_bp.route('/api/submit_advanced_chat', methods=['POST'])
def api_submit_advanced_chat(): 
    try:
        if not session.get('current_file_path'):
            return jsonify({"error": "No hay archivo CSV."}), 400
        data = request.get_json(silent=True) or {}
        user_prompt = data.get('prompt')
        if not user_prompt:
            return jsonify({"error": "No se recibió prompt."}), 400

    except Exception as e:
        current_app.logger.exception(f"Error inesperado en api_submit_advanced_chat (validación): {e}")
        return jsonify({"error": f"Error inesperado en el servidor: {str(e)}"}), 500

    if 'consumo_sesion' not in session:
        session['consumo_sesion'] = {'total_tokens': 0, 'total_cost': 0.0}
    
    df_global = get_dataframe_from_session_file()
    if df_global is None or df_global.empty:
        return jsonify({"error": "No se pudo cargar el DataFrame."}), 500

    prompt_lower = user_prompt.lower()
    data_string = ""
    entity_type = None
    entity_name = None
    # Inicializaciones necesarias para evitar excepciones de variables no definidas
    prompt_norm = normalize_text(user_prompt)
    ambiguity_message = ""
    
    # --- Lógica de Resumen Cuantitativo (Notas) ---
    historical_data_summary = ""
    if any_keyword_in_prompt(user_prompt, get_evolution_keywords()):
        try:
            current_app.logger.info("Detectada intención de evolución CUANTITATIVA general (Chat Avanzado).")
            db_path = current_app.config['DATABASE_FILE']
            order_dir = 'ASC' if any_keyword_in_prompt(user_prompt, get_negative_evolution_keywords()) else 'DESC'

            historical_data_summary = get_student_evolution_summary(
                db_path, 
                top_n=current_app.config.get('TOP_N_EVOLUTION', 5),
                order_direction=order_dir
            )
        except Exception as e:
            current_app.logger.error(f"Error al obtener resumen de evolución general: {e}")
            historical_data_summary = f"Error al consultar historial: {e}"

    # Nuevo: Evolución de asistencia general (Top-N)
    if any_keyword_in_prompt(user_prompt, get_attendance_keywords()) and not entity_type and current_app.config.get('ENABLE_ATTENDANCE_SUMMARY', True):
        try:
            current_app.logger.info("Detectada intención de evolución de ASISTENCIA general (Chat Avanzado).")
            db_path = current_app.config['DATABASE_FILE']
            order_dir_att = 'ASC' if any_keyword_in_prompt(user_prompt, get_negative_evolution_keywords()) else 'DESC'
            attendance_summary = get_attendance_evolution_summary(
                db_path,
                top_n=current_app.config.get('TOP_N_EVOLUTION', 5),
                order_direction=order_dir_att
            )
            historical_data_summary = (historical_data_summary + "\n\n" + attendance_summary).strip()
        except Exception as e:
            current_app.logger.error(f"Error al obtener resumen de asistencia general: {e}")
            historical_data_summary = (historical_data_summary + f"\n\nError asistencia: {e}").strip()

    # --- Lógica de Detección de Entidad (para CUALITATIVO) ---
    file_path = session.get('current_file_path')
    nombre_col = current_app.config['NOMBRE_COL']
    curso_col = current_app.config['CURSO_COL']
    student_names = df_global[nombre_col].unique().tolist()
    course_names = df_global[curso_col].unique().tolist()

    found_student = next((name for name in student_names if normalize_text(name) in prompt_norm), None)
    if found_student:
        entity_type = 'alumno'
        entity_name = found_student
        current_app.logger.info(f"Chat Avanzado detectó entidad ALUMNO: {entity_name}")
        df_entidad = df_global[df_global[nombre_col] == entity_name]
        data_string = load_data_as_string(file_path, specific_entity_df=df_entidad)
    else:
        # Resolver curso o detectar ambigüedad por nivel sin paralelo
        resolved_course, amb_level, candidates = detect_course_or_ambiguity(course_names, user_prompt)
        if resolved_course:
            entity_type = 'curso'
            entity_name = resolved_course
            current_app.logger.info(f"Chat Avanzado detectó entidad CURSO: {entity_name}")
            df_entidad = df_global[df_global[curso_col] == entity_name]
            data_string = load_data_as_string(file_path, specific_entity_df=df_entidad)
        elif amb_level:
            policy = current_app.config.get('COURSE_AMBIGUITY_POLICY', 'ask')
            default_parallel = normalize_text(current_app.config.get('DEFAULT_PARALLEL', 'A'))
            chosen = None
            if policy == 'default':
                # Elegir curso cuyo último token (paralelo) coincide con DEFAULT_PARALLEL
                for c in candidates:
                    toks = normalize_text(c).split()
                    if toks and len(toks[-1]) == 1 and toks[-1].isalpha() and toks[-1] == default_parallel.lower():
                        chosen = c
                        break
            if chosen:
                entity_type = 'curso'
                entity_name = chosen
                current_app.logger.info(f"Ambigüedad resuelta por política DEFAULT: {entity_name}")
                df_entidad = df_global[df_global[curso_col] == entity_name]
                data_string = load_data_as_string(file_path, specific_entity_df=df_entidad)
            else:
                # Construir mensaje de ambigüedad para guiar al usuario
                letters = []
                for c in candidates:
                    toks = normalize_text(c).split()
                    if toks and len(toks[-1]) == 1 and toks[-1].isalpha():
                        letters.append(toks[-1].upper())
                unique_letters = sorted(set(letters))
                ambiguity_message = (
                    f"Ambigüedad detectada: el nivel '{amb_level}' tiene varios paralelos disponibles: "
                    f"{', '.join(unique_letters)}. Por favor, especifica la letra (p. ej., '{amb_level} {unique_letters[0]}')."
                )
                current_app.logger.info(ambiguity_message)

                # NUEVO: Aunque haya ambigüedad, proveer datos CSV filtrados al nivel (todos los paralelos)
                try:
                    df_nivel = df_global[df_global[curso_col].astype(str).str.contains(str(amb_level), case=False, na=False)]
                    if not df_nivel.empty:
                        data_string = load_data_as_string(file_path, specific_entity_df=df_nivel)
                        current_app.logger.info(f"Contexto CSV adjuntado para nivel ambiguo: {amb_level} (total filas: {len(df_nivel)})")
                except Exception as e:
                    current_app.logger.error(f"Error al adjuntar CSV por nivel ambiguo {amb_level}: {e}")
            
    # --- INICIO: NUEVA LÓGICA DE RESUMEN CUALITATIVO (Comportamiento) ---
    qualitative_history_summary = ""
    # Solo se activa si detectamos un ALUMNO y palabras clave cualitativas
    if entity_type == 'alumno' and entity_name and any_keyword_in_prompt(user_prompt, get_qualitative_keywords()) and current_app.config.get('ENABLE_QUALITATIVE_SUMMARY', True):
        try:
            current_app.logger.info(f"Detectada intención de evolución CUALITATIVA (Avanzado) para: {entity_name}")
            db_path = current_app.config['DATABASE_FILE']
            qualitative_history_summary = get_student_qualitative_history(
                db_path, 
                student_name=entity_name
            )
        except Exception as e:
            current_app.logger.error(f"Error al obtener resumen de evolución cualitativa (Avanzado) para {entity_name}: {e}")
            qualitative_history_summary = f"Error al consultar historial cualitativo: {e}"
    elif entity_type == 'curso' and entity_name and any_keyword_in_prompt(user_prompt, get_qualitative_keywords()) and current_app.config.get('ENABLE_QUALITATIVE_SUMMARY', True):
        try:
            current_app.logger.info(f"Detectada intención de resumen CUALITATIVO agregado (Avanzado) para curso: {entity_name}")
            db_path = current_app.config['DATABASE_FILE']
            qualitative_history_summary = get_course_qualitative_summary(
                db_path,
                course_name=entity_name,
                max_entries=current_app.config.get('MAX_QUALITATIVE_ENTRIES', 30)
            )
        except Exception as e:
            current_app.logger.error(f"Error al obtener resumen cualitativo por curso (Avanzado) {entity_name}: {e}")
            qualitative_history_summary = f"Error al consultar resumen cualitativo del curso: {e}"

    # Evolución de asistencia específica por alumno
    if entity_type == 'alumno' and entity_name and any_keyword_in_prompt(user_prompt, get_attendance_keywords()) and current_app.config.get('ENABLE_ATTENDANCE_SUMMARY', True):
        try:
            current_app.logger.info(f"Detectada intención de ASISTENCIA (Avanzado) para alumno: {entity_name}")
            db_path = current_app.config['DATABASE_FILE']
            order_dir_att = 'ASC' if any_keyword_in_prompt(user_prompt, get_negative_evolution_keywords()) else 'DESC'
            attendance_summary = get_attendance_evolution_summary(
                db_path,
                entity_name=entity_name,
                order_direction=order_dir_att
            )
            historical_data_summary = (historical_data_summary + "\n\n" + attendance_summary).strip()
        except Exception as e:
            current_app.logger.error(f"Error al obtener resumen de asistencia (Avanzado) para {entity_name}: {e}")
            historical_data_summary = (historical_data_summary + f"\n\nError asistencia: {e}").strip()

    # Adjuntar documentos clave recientes al bloque cualitativo
    include_docs = current_app.config.get('INCLUDE_KEY_DOCS_IN_PROMPT', True)
    key_docs_block = _build_key_docs_block(entity_type, entity_name, file_path) if include_docs and entity_type and entity_name else ""
    if key_docs_block:
        qualitative_history_summary = (qualitative_history_summary + key_docs_block).strip()
    # Mensaje de ambigüedad (si existe) se adjunta al bloque cualitativo
    if ambiguity_message:
        qualitative_history_summary = (qualitative_history_summary + ("\n\n" if qualitative_history_summary else "") + ambiguity_message).strip()
    # --- FIN: NUEVA LÓGICA ---

    # --- Lógica de Contexto General (ACTUALIZADA: siempre adjuntar CSV) ---
    # Si aún no se ha construido data_string por detección de entidad o nivel, adjuntar el CSV completo.
    if not data_string:
        try:
            data_string = load_data_as_string(file_path)
            current_app.logger.info("Chat Avanzado (general): CSV completo adjuntado como contexto principal.")
        except Exception as e:
            current_app.logger.error(f"Error al cargar CSV para contexto general: {e}")
            data_string = f"Error al cargar CSV para contexto general: {e}"
    
    # --- NUEVO: Construcción de Feature Store y generación de señales previas ---
    fs_signals = ""
    try:
        feature_store = build_feature_store_from_csv(df_global)
        fs_signals = build_feature_store_signals(
            feature_store,
            entity_type=entity_type,
            entity_name=entity_name,
            user_prompt=user_prompt
        )
        # Respuesta directa: alumno con nota más baja en una asignatura (si el prompt lo pide)
        try:
            pnorm = normalize_text(user_prompt)
            asks_lowest = any(kw in pnorm for kw in [
                'nota mas baja', 'menor nota', 'peor nota', 'nota minima', 'calificacion mas baja'
            ])
            if asks_lowest:
                res = get_lowest_grade_student_for_subject(df_global, user_prompt)
                if res:
                    if isinstance(res, dict) and res.get('multiple'):
                        alumnos = ", ".join([f"{r['nombre']} ({r['curso']})" for r in res['registros']])
                        direct = (
                            f"Respuesta directa: Nota más baja en {res['asignatura']}: {res['min_nota']:.2f}. "
                            f"Estudiantes: {alumnos}."
                        )
                    else:
                        direct = (
                            f"Respuesta directa: {res['nombre']} ({res['curso']}) tiene la nota más baja en "
                            f"{res['asignatura']}: {res['nota']:.2f}."
                        )
                    fs_signals = (direct + "\n" + fs_signals).strip()
        except Exception as e_dir:
            current_app.logger.error(f"Error en respuesta directa por asignatura: {e_dir}")
        if fs_signals:
            current_app.logger.info("Señales del Feature Store generadas y listas para el prompt del Chat Avanzado.")
    except Exception as e_fs:
        current_app.logger.error(f"Error al construir señales del Feature Store: {e_fs}")
        fs_signals = ""

    history_list = session.get('advanced_chat_history', [])
    history_fmt = format_chat_history_for_prompt(history_list)
    
    analysis_result = analyze_data_with_gemini(
        data_string, user_prompt, vector_store, vector_store_followups, history_fmt, 
        is_direct_chat_query=True, entity_type=entity_type, entity_name=entity_name,
        historical_data_summary_string=historical_data_summary, # <-- Pasa el resumen de NOTAS
        qualitative_history_summary_string=qualitative_history_summary, # <-- Pasa el resumen CUALITATIVO
        feature_store_signals_string=fs_signals # <-- NUEVO: Señales del Feature Store (CSV primero)
    )
    
    # ... (lógica de guardado de historial de chat y consumo no cambia) ...
    if not analysis_result.get('error'):
        history_list.append({
            'user': user_prompt, 
            'gemini_markdown': analysis_result['raw_markdown'], 
            'gemini_html': analysis_result['html_output']
        })
        session['advanced_chat_history'] = history_list[-current_app.config.get('MAX_CHAT_HISTORY_SESSION_STORAGE', 10):]
        session['consumo_sesion']['total_tokens'] += analysis_result.get('total_tokens', 0)
        session['consumo_sesion']['total_cost'] += analysis_result.get('total_cost', 0)
        analysis_result['consumo_sesion'] = session['consumo_sesion']
        session.modified = True

        return jsonify(analysis_result)

@main_bp.route('/api/add_advanced_chat_follow_up', methods=['POST'])
def api_add_advanced_chat_follow_up(): 
    data = request.json
    comment = data.get('follow_up_comment')
    user_prompt_fu = data.get('user_prompt')
    analysis_md_fu = data.get('gemini_analysis_markdown')
    # MODIFIED: Accept entity type and name from payload
    related_entity_type = data.get('related_entity_type') 
    related_entity_name = data.get('related_entity_name')

    filename = session.get('uploaded_filename')
    if not filename: return jsonify({"error": "No hay archivo activo."}), 400
    if not all([comment, user_prompt_fu, analysis_md_fu]): return jsonify({"error": "Faltan datos para el seguimiento (comentario, prompt o análisis)."}), 400
    
    # Normalize empty strings to None for DB
    if not related_entity_type: related_entity_type = None
    if not related_entity_name: related_entity_name = None

    prompt_db_identifier = f"Chat Avanzado (Archivo: {filename}) - Pregunta: '{user_prompt_fu[:100]}{'...' if len(user_prompt_fu) > 100 else ''}'"
    try:
        with sqlite3.connect(current_app.config['DATABASE_FILE']) as conn:
            # MODIFIED: Include entity type and name in insert
            conn.cursor().execute('''INSERT INTO follow_ups 
                                     (related_filename, related_prompt, related_analysis, follow_up_comment, 
                                      follow_up_type, related_entity_type, related_entity_name) 
                                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                           (filename, prompt_db_identifier, analysis_md_fu, comment, 
                            'advanced_chat_note', related_entity_type, related_entity_name))
        message = "Nota de Chat Avanzado guardada."
        if related_entity_type and related_entity_name:
            message += f" Asociada a {related_entity_type}: {related_entity_name}."
        if embedding_model_instance and reload_followup_vector_store(current_app.config): message += " Índice de seguimientos actualizado."
        elif not embedding_model_instance: message += " Modelo de embeddings no disponible, índice de seguimientos no actualizado."
        else: message += " Error al actualizar índice de seguimientos."
        return jsonify({"message": message}), 201
    except Exception as e: traceback.print_exc(); return jsonify({"error": f"Error interno al guardar seguimiento de chat avanzado: {e}"}), 500

@main_bp.route('/api/add_contextual_follow_up', methods=['POST'])
def api_add_contextual_follow_up(): 
    # This function already correctly handles entity_type and entity_name
    data = request.json; comment, user_prompt_ctx, analysis_md_ctx = data.get('follow_up_comment'), data.get('user_prompt'), data.get('gemini_analysis_markdown')
    tipo_ctx, nombre_ctx, filename_ctx = data.get('tipo_entidad'), data.get('nombre_entidad'), session.get('uploaded_filename')
    if not filename_ctx: return jsonify({"error": "No hay archivo activo."}), 400
    if not all([comment, user_prompt_ctx, analysis_md_ctx, tipo_ctx, nombre_ctx]): return jsonify({"error": "Faltan datos para el seguimiento contextual."}), 400
    prompt_db_ctx_identifier = f"Seguimiento para {tipo_ctx}: {nombre_ctx} (Archivo: {filename_ctx}). Pregunta: '{user_prompt_ctx[:100]}{'...' if len(user_prompt_ctx) > 100 else ''}'"
    try:
        with sqlite3.connect(current_app.config['DATABASE_FILE']) as conn:
            conn.cursor().execute('''INSERT INTO follow_ups 
                                     (related_filename, related_prompt, related_analysis, follow_up_comment, 
                                      follow_up_type, related_entity_type, related_entity_name) 
                                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                           (filename_ctx, prompt_db_ctx_identifier, analysis_md_ctx, comment, 'contextual_note', tipo_ctx, nombre_ctx))
        message = "Seguimiento contextual guardado."
        if embedding_model_instance and reload_followup_vector_store(current_app.config): message += " Índice de seguimientos actualizado."
        elif not embedding_model_instance: message += " Modelo de embeddings no disponible, índice de seguimientos no actualizado."
        else: message += " Error al actualizar índice de seguimientos."
        return jsonify({"message": message}), 201
    except Exception as e: traceback.print_exc(); return jsonify({"error": f"Error interno al guardar seguimiento contextual: {e}"}), 500

# --- RUTAS PARA REPORTE 360 ---
@main_bp.route('/reporte_360/<tipo_entidad>/<path:valor_codificado>')
def generar_reporte_360(tipo_entidad, valor_codificado):
    if not session.get('current_file_path'): 
        flash('Por favor, carga un archivo CSV primero.', 'warning')
        return redirect(url_for('main.index'))
    try: 
        nombre_entidad = unquote(valor_codificado)
    except Exception: 
        flash('Nombre de entidad no válido.', 'danger')
        return redirect(url_for('main.index'))

    df_global = get_dataframe_from_session_file()
    if df_global is None or df_global.empty: 
        flash('No se pudieron cargar los datos del archivo CSV.', 'danger')
        return redirect(url_for('main.detalle_entidad', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad)))
    
    df_entidad = pd.DataFrame()
    try:
        nombre_entidad_normalizado = nombre_entidad.strip().lower()
        if tipo_entidad == 'alumno': 
            df_entidad = df_global[df_global[current_app.config['NOMBRE_COL']].astype(str).str.strip().str.lower() == nombre_entidad_normalizado]
        elif tipo_entidad == 'curso': 
            df_entidad = df_global[df_global[current_app.config['CURSO_COL']].astype(str).str.strip().str.lower() == nombre_entidad_normalizado]
        else: 
            flash('Tipo de entidad no reconocido para el reporte.', 'danger')
            return redirect(url_for('main.index'))
    except KeyError as e: 
        flash(f"Error de configuración: La columna '{e}' no se encuentra.", 'danger')
        return redirect(url_for('main.detalle_entidad', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad)))
    
    if df_entidad.empty: 
        flash(f'No se encontraron datos para {tipo_entidad} "{nombre_entidad}".', 'warning')
        return redirect(url_for('main.detalle_entidad', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad)))
    
    datos_entidad_string = load_data_as_string(session.get('current_file_path'), specific_entity_df=df_entidad)
    if datos_entidad_string.startswith("Error:"): 
        flash(f'Error al cargar datos para el reporte: {datos_entidad_string}', 'danger')
        return redirect(url_for('main.detalle_entidad', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad)))
    
    # Usamos la nueva constante del config para el prompt, formateándola con los datos de la entidad.
    prompt_template = current_app.config.get('PROMPT_REPORTE_360', "Generar reporte para {tipo_entidad} {nombre_entidad}")
    prompt_reporte_360_base = prompt_template.format(tipo_entidad=tipo_entidad, nombre_entidad=nombre_entidad)
    
    analysis_result = analyze_data_with_gemini(
        data_string=datos_entidad_string, 
        user_prompt=prompt_reporte_360_base, 
        vs_inst=vector_store,
        vs_followup=vector_store_followups,
        chat_history_string="", 
        is_reporte_360=True,
        entity_type=tipo_entidad,
        entity_name=nombre_entidad
    )

    if analysis_result.get('error'):
        flash(f"Error al generar el Reporte 360: {analysis_result['error']}", 'danger')
        return redirect(url_for('main.detalle_entidad', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad)))

    reporte_html = analysis_result['html_output']
    reporte_markdown = analysis_result['raw_markdown']

    current_csv_filename = session.get('uploaded_filename', 'N/A')
    db_path = current_app.config['DATABASE_FILE']
    
    reporte_360_id = save_reporte_360_to_db(db_path, current_csv_filename, tipo_entidad, nombre_entidad, reporte_markdown, prompt_reporte_360_base)
    
    if reporte_360_id:
        flash(f'Reporte 360 para {nombre_entidad} guardado exitosamente (ID: {reporte_360_id}).', 'success')
        session['current_reporte_360_id'] = reporte_360_id 
        if embedding_model_instance and reload_followup_vector_store(current_app.config):
            flash('Índice de seguimientos (incluyendo Reportes 360) actualizado.', 'info')
    else:
        flash(f'Reporte 360 generado, pero hubo un error al guardarlo en la base de datos.', 'warning')
        session.pop('current_reporte_360_id', None)

    session['reporte_360_markdown'] = reporte_markdown
    session['reporte_360_entidad_tipo'] = tipo_entidad
    session['reporte_360_entidad_nombre'] = nombre_entidad 
    session.modified = True
    
    observaciones_del_reporte = []
    if reporte_360_id:
        observaciones_del_reporte = get_observations_for_reporte_360(db_path, reporte_360_id)

# --- INICIO: MODIFICACIÓN ---
    # Generar el timestamp actual en UTC y convertirlo a Santiago
    utc_now = datetime.datetime.now(pytz.utc)
    santiago_now = utc_now.astimezone(_get_tz())
    timestamp_generacion_actual = santiago_now.strftime('%d/%m/%Y %H:%M')
    # --- FIN: MODIFICACIÓN ---

    return render_template('reporte_360.html', 
                           page_title=f"Reporte 360 - {nombre_entidad}", 
                           tipo_entidad=tipo_entidad, 
                           nombre_entidad=nombre_entidad, 
                           reporte_html=reporte_html, 
                           reporte_360_id=reporte_360_id, 
                           observaciones_reporte=observaciones_del_reporte,
                           filename=current_csv_filename,
                           timestamp_generacion=timestamp_generacion_actual) # <-- Variable añadida

@main_bp.route('/descargar_reporte_360_html/<tipo_entidad>/<path:valor_codificado>')
def descargar_reporte_360_html(tipo_entidad, valor_codificado):
    # No changes to this function
    try: nombre_entidad_url = unquote(valor_codificado)
    except Exception: flash('Nombre de entidad no válido para descarga.', 'danger'); return redirect(url_for('main.index'))
    session_tipo = session.get('reporte_360_entidad_tipo')
    session_nombre = session.get('reporte_360_entidad_nombre')
    condicion_tipo_falla = session_tipo != tipo_entidad
    condicion_nombre_falla = not (session_nombre and nombre_entidad_url and session_nombre.strip() == nombre_entidad_url.strip())
    condicion_markdown_falla = not session.get('reporte_360_markdown')
    if condicion_tipo_falla or condicion_nombre_falla or condicion_markdown_falla:
        flash_message = 'No hay un reporte 360 activo en sesión para esta entidad o los datos no coinciden. Por favor, genere el reporte primero.'
        flash(flash_message, 'warning')
        return redirect(url_for('main.detalle_entidad', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad_url)))
    reporte_markdown_contenido = session.get('reporte_360_markdown')
    reporte_html_contenido = markdown.markdown(reporte_markdown_contenido, extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists'])
    html_para_descarga = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"><title>Reporte 360 - {nombre_entidad_url}</title><style>body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol"; margin: 20px; line-height: 1.6; color: #333; }} h1 {{ font-size: 1.8em; color: #2c3e50; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-top:0; }} .report-header {{ margin-bottom: 20px; font-size: 0.9em; color: #555; padding-bottom:10px; border-bottom: 1px dashed #ccc;}} .report-header strong {{ color: #000; }} .report-content {{ margin-top: 20px; }} table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; font-size: 0.9em; }} th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }} th {{ background-color: #f2f2f2; font-weight: bold; }} ul, ol {{ padding-left: 20px; }} li {{ margin-bottom: 5px; }} .report-content h1 {{ font-size: 1.6em; }} .report-content h2 {{ font-size: 1.4em; }} .report-content h3 {{ font-size: 1.2em; }}</style></head><body><h1>Reporte 360</h1><div class="report-header"><strong>Entidad:</strong> {nombre_entidad_url} ({tipo_entidad.capitalize()})<br><strong>Archivo de Datos Origen:</strong> {session.get('uploaded_filename', 'N/A')}<br><strong>Generado el:</strong> {datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</div><div class="report-content">{reporte_html_contenido}</div></body></html>"""
    safe_filename_base = "".join(c if c.isalnum() else "_" for c in nombre_entidad_url)
    html_filename = f"Reporte_360_{tipo_entidad}_{safe_filename_base}.html"
    response = Response(html_para_descarga, mimetype='text/html')
    response.headers['Content-Disposition'] = f'attachment; filename="{html_filename}"'
    return response

@main_bp.route('/api/add_observacion_reporte_360', methods=['POST'])
def api_add_observacion_reporte_360():
    # No changes to this function
    data = request.json
    reporte_360_id = data.get('reporte_360_id')
    observacion_texto = data.get('observacion_texto')
    observador_nombre = data.get('observador_nombre')
    tipo_entidad = data.get('tipo_entidad')
    nombre_entidad = data.get('nombre_entidad')
    
    current_csv_filename = session.get('uploaded_filename')

    if not all([reporte_360_id, observacion_texto, observador_nombre, tipo_entidad, nombre_entidad, current_csv_filename]):
        return jsonify({"error": "Faltan datos para guardar la observación."}), 400

    db_path = current_app.config['DATABASE_FILE']
    if save_observation_for_reporte_360(db_path, reporte_360_id, observador_nombre, observacion_texto, tipo_entidad, nombre_entidad, current_csv_filename):
        if embedding_model_instance and reload_followup_vector_store(current_app.config):
            flash_message = 'Observación guardada y índice RAG actualizado.'
            status_code = 201
        elif not embedding_model_instance:
            flash_message = 'Observación guardada, pero el índice RAG no se actualizó (modelo embeddings no disponible).'
            status_code = 201 
        else:
            flash_message = 'Observación guardada, pero hubo un error al actualizar el índice RAG.'
            status_code = 201 
        
        observaciones_actualizadas = get_observations_for_reporte_360(db_path, reporte_360_id)
        return jsonify({"message": flash_message, "observaciones": observaciones_actualizadas}), status_code
    else:
        return jsonify({"error": "Error interno al guardar la observación."}), 500

# --- RUTAS PARA PLAN DE INTERVENCIÓN ---
@main_bp.route('/generar_plan_intervencion/<tipo_entidad>/<path:valor_codificado>')
def generar_plan_intervencion(tipo_entidad, valor_codificado):
    if not session.get('current_file_path'):
        flash('Por favor, carga un archivo CSV primero.', 'warning')
        return redirect(url_for('main.index'))

    try:
        nombre_entidad = unquote(valor_codificado)
    except Exception:
        flash('Nombre de entidad no válido para el plan.', 'danger')
        return redirect(url_for('main.index'))

    if not (session.get('reporte_360_markdown') and
            session.get('reporte_360_entidad_tipo') == tipo_entidad and
            session.get('reporte_360_entidad_nombre') == nombre_entidad):
        flash('Primero debes generar el "Reporte 360" para esta entidad.', 'warning')
        return redirect(url_for('main.detalle_entidad', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad)))

    reporte_360_base_md = session['reporte_360_markdown']

    plan_html, plan_markdown = generate_intervention_plan_with_gemini(
        reporte_360_markdown=reporte_360_base_md,
        tipo_entidad=tipo_entidad,
        nombre_entidad=nombre_entidad
    )

    if isinstance(plan_html, str) and plan_html.startswith("Error:"):
        flash(f"Error al generar el Plan de Intervención: {plan_html.replace('Error: ', '')}", 'danger')
        return redirect(url_for('main.detalle_entidad', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad)))

    db_path = current_app.config['DATABASE_FILE']
    current_csv_filename = session.get('uploaded_filename', 'N/A')
    
    last_plan_id = save_intervention_plan_to_db(db_path, current_csv_filename, tipo_entidad, nombre_entidad, plan_markdown, reporte_360_base_md)

    if last_plan_id:
        flash('Plan de Intervención generado y guardado exitosamente.', 'success')
        # --- LÍNEA AÑADIDA ---
        # Guardamos el ID del plan recién creado en la sesión.
        session['current_intervention_plan_id'] = last_plan_id
    else:
        flash('Plan de Intervención generado, pero hubo un error al guardarlo en la base de datos.', 'warning')

    session['current_intervention_plan_html'] = plan_html
    session['current_intervention_plan_markdown'] = plan_markdown 
    # --- INICIO: MODIFICACIÓN ---
    # Guardar el timestamp de Santiago en la sesión
    utc_now = datetime.datetime.now(pytz.utc)
    santiago_now = utc_now.astimezone(_get_tz())
    session['current_intervention_plan_date'] = santiago_now.strftime('%d/%m/%Y %H:%M')
    # --- FIN: MODIFICACIÓN ---
    session['current_intervention_plan_for_entity_type'] = tipo_entidad
    session['current_intervention_plan_for_entity_name'] = nombre_entidad
    session.modified = True
    
    plan_ref_for_url = last_plan_id if last_plan_id is not None else 'current'

    return redirect(url_for('main.visualizar_plan_intervencion', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad), plan_ref=plan_ref_for_url))

@main_bp.route('/visualizar_plan_intervencion/<tipo_entidad>/<path:valor_codificado>/<plan_ref>')
def visualizar_plan_intervencion(tipo_entidad, valor_codificado, plan_ref):
    try:
        nombre_entidad = unquote(valor_codificado)
    except Exception:
        flash('Nombre de entidad no válido.', 'danger')
        return redirect(url_for('main.index'))

    plan_html_content = None
    plan_date = "Fecha no disponible"
    plan_title = f"Plan de Intervención para {tipo_entidad.capitalize()}: {nombre_entidad}"
    # Obtenemos el filename de la sesión solo como un fallback
    current_filename = session.get('uploaded_filename', 'N/A')

    if plan_ref == 'current': 
        if (session.get('current_intervention_plan_html') and
            session.get('current_intervention_plan_for_entity_type') == tipo_entidad and
            session.get('current_intervention_plan_for_entity_name') == nombre_entidad):
            plan_html_content = session['current_intervention_plan_html']
            plan_date = session.get('current_intervention_plan_date', plan_date)
        else:
            flash('No hay un plan de intervención actual en sesión para esta entidad.', 'warning')
            return redirect(url_for('main.detalle_entidad', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad)))
    else: 
        try:
            plan_id_to_load = int(plan_ref)
            db_path = current_app.config['DATABASE_FILE']
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # --- BLOQUE CORREGIDO ---
                # Se eliminó el filtro AND related_filename = ?
                # La consulta ahora valida que el ID, tipo y nombre coincidan,
                # lo cual es suficiente para cargar el reporte correcto.
                cursor.execute("""
                    SELECT timestamp, follow_up_comment, related_filename FROM follow_ups 
                    WHERE id = ? AND follow_up_type = 'intervention_plan' 
                    AND related_entity_type = ? AND related_entity_name = ?
                    """, 
                               (plan_id_to_load, tipo_entidad, nombre_entidad))
                plan_data = cursor.fetchone()
                
                if plan_data:
                    # Renderizar el Markdown del plan histórico con las mismas extensiones que usamos al generarlo
                    # (sin 'nl2br' para evitar saltos forzados y mejorar la justificación en "Fundamentación").
                    plan_html_content = markdown.markdown(
                        plan_data['follow_up_comment'],
                        extensions=['fenced_code', 'tables', 'sane_lists']
                    )
                    # --- INICIO: MODIFICACIÓN ---
                    # Convertir el timestamp de la BD (UTC) a Santiago
                    try:
                        naive_dt = datetime.datetime.strptime(plan_data["timestamp"], '%Y-%m-%d %H:%M:%S')
                        utc_dt = pytz.utc.localize(naive_dt)
                        santiago_dt = utc_dt.astimezone(_get_tz())
                        plan_date = santiago_dt.strftime('%d/%m/%Y %H:%M')
                    except (ValueError, TypeError):
                        plan_date = plan_data.get("timestamp", "Fecha no válida")
                    # --- FIN: MODIFICACIÓN ---
                    current_filename = plan_data['related_filename'] # Usamos el nombre del archivo original del reporte
                else:
                    flash(f'No se encontró el plan de intervención con ID {plan_id_to_load} para esta entidad.', 'warning')
                    return redirect(url_for('main.biblioteca_reportes'))
        except (ValueError, TypeError):
            flash('Referencia de plan no válida.', 'danger')
            return redirect(url_for('main.biblioteca_reportes'))
        except Exception as e:
            flash(f'Error al cargar el plan de intervención histórico: {str(e)}', 'danger')
            traceback.print_exc()
            return redirect(url_for('main.biblioteca_reportes'))

    return render_template('visualizar_plan_intervencion.html',
                           page_title=plan_title,
                           tipo_entidad=tipo_entidad,
                           nombre_entidad=nombre_entidad,
                           plan_html=plan_html_content,
                           fecha_emision_plan=plan_date,
                           plan_ref=plan_ref, 
                           filename=current_filename)

# --- RUTA PARA RECURSOS DE APOYO ---
@main_bp.route('/generar_recursos_apoyo/<tipo_entidad>/<path:valor_codificado>/<plan_ref>')
def generar_recursos_apoyo(tipo_entidad, valor_codificado, plan_ref):
    # No changes to this function
    if not session.get('current_file_path'):
        flash('Por favor, carga un archivo CSV primero.', 'warning')
        return redirect(url_for('main.index'))

    try:
        nombre_entidad = unquote(valor_codificado)
        plan_id_to_load = int(plan_ref) 
    except (ValueError, TypeError):
        flash('Referencia de plan o entidad no válida.', 'danger')
        return redirect(url_for('main.index'))

    plan_markdown_content = None
    plan_timestamp_str = "Fecha no disponible"
    db_path = current_app.config['DATABASE_FILE']
    current_csv_filename = session.get('uploaded_filename', 'N/A')

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, follow_up_comment FROM follow_ups 
                WHERE id = ? AND follow_up_type = 'intervention_plan'
                AND related_entity_type = ? AND related_entity_name = ?
                AND related_filename = ?
            """, (plan_id_to_load, tipo_entidad, nombre_entidad, current_csv_filename))
            plan_data_row = cursor.fetchone()
            if plan_data_row:
                plan_markdown_content = plan_data_row['follow_up_comment']
                plan_timestamp_str = datetime.datetime.strptime(plan_data_row["timestamp"], '%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y %H:%M')
            else:
                flash(f'No se encontró el Plan de Intervención con ID {plan_id_to_load} para {nombre_entidad} del archivo actual.', 'warning')
                return redirect(url_for('main.visualizar_plan_intervencion', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad), plan_ref=plan_ref))
    except Exception as e:
        flash(f'Error al cargar el Plan de Intervención desde la base de datos: {str(e)}', 'danger')
        traceback.print_exc()
        return redirect(url_for('main.visualizar_plan_intervencion', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad), plan_ref=plan_ref))

    if not plan_markdown_content: 
        flash('No se pudo obtener el contenido del plan de intervención.', 'danger')
        return redirect(url_for('main.visualizar_plan_intervencion', tipo_entidad=tipo_entidad, valor_codificado=quote(nombre_entidad), plan_ref=plan_ref))

    recursos_html_output = search_web_for_support_resources(plan_markdown_content, tipo_entidad, nombre_entidad)

    return render_template('recursos_de_apoyo.html',
                           page_title=f"Recursos de Apoyo para {nombre_entidad}",
                           tipo_entidad=tipo_entidad,
                           nombre_entidad=nombre_entidad,
                           recursos_html=recursos_html_output,
                           fecha_emision_plan=plan_timestamp_str, 
                           plan_ref=plan_ref, 
                           filename=current_csv_filename)

def crear_respuesta_directa(texto_markdown):
    """Función auxiliar para construir el objeto JSON de respuesta directa."""
    return {
        'html_output': markdown.markdown(texto_markdown),
        'raw_markdown': texto_markdown,
        'model_name': 'Cálculo Directo del Servidor',
        'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
        'input_cost': 0.0, 'output_cost': 0.0, 'total_cost': 0.0,
        'consumo_sesion': session.get('consumo_sesion', {'total_tokens': 0, 'total_cost': 0.0}),
        'error': None
    }

@main_bp.route('/biblioteca')
def biblioteca_reportes():
    if not session.get('current_file_path'):
        flash('Primero debes cargar un archivo CSV para ver su historial de reportes.', 'warning')
        return redirect(url_for('main.index'))

    db_path = current_app.config['DATABASE_FILE']
    current_filename = session.get('uploaded_filename')
    
    # --- NUEVA LÓGICA DE FILTRADO ---
    search_tipo = request.args.get('tipo_entidad')
    search_nombre = request.args.get('nombre_entidad')
    
    reportes = []
    query_params = [current_filename]
    
    base_query = """
        SELECT id, timestamp, report_date, follow_up_type, related_entity_type, related_entity_name
        FROM follow_ups
        WHERE (follow_up_type = 'reporte_360' OR follow_up_type = 'intervention_plan' OR follow_up_type = 'observacion_entidad')
        AND related_filename = ?
    """
    
    # Define un título por defecto
    page_title = "Biblioteca de Reportes (Todos)"
    
    if search_tipo and search_nombre:
        base_query += " AND related_entity_type = ? AND related_entity_name = ?"
        query_params.extend([search_tipo, search_nombre])
        # Actualiza el título si hay un filtro
        page_title = f"Biblioteca: {search_nombre}"

    base_query += " ORDER BY timestamp DESC"
    # --- FIN LÓGICA DE FILTRADO ---

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(base_query, tuple(query_params))
            # --- INICIO: MODIFICACIÓN ---
            # Formatear el timestamp aquí, convirtiendo de UTC a Santiago
            rows = cursor.fetchall()
            reportes = []
            for row in rows:
                reporte_dict = dict(row)
                try:
                    # 1. Parsear el string de la BD (asumiendo UTC)
                    naive_dt = datetime.datetime.strptime(reporte_dict['timestamp'], '%Y-%m-%d %H:%M:%S')
                    # 2. Localizarlo como UTC
                    utc_dt = pytz.utc.localize(naive_dt)
                    # 3. Convertir a zona horaria de Santiago
                    santiago_dt = utc_dt.astimezone(_get_tz())
                    # 4. Formatear
                    reporte_dict['timestamp_formateado'] = santiago_dt.strftime('%d/%m/%Y %H:%M')
                except (ValueError, TypeError):
                    reporte_dict['timestamp_formateado'] = reporte_dict.get('timestamp', 'Fecha no válida')
                reportes.append(reporte_dict)
            # --- FIN: MODIFICACIÓN ---

    except Exception as e:
        flash(f'Error al cargar la biblioteca de reportes: {e}', 'danger')
        traceback.print_exc()

    return render_template('biblioteca.html',
                           page_title=page_title, # Pasa el título dinámico
                           reportes=reportes,
                           filename=current_filename,
                           tipo_entidad=search_tipo,
                           nombre_entidad=search_nombre)

@main_bp.route('/registrar_observacion')
def registrar_observacion():
    if not session.get('current_file_path'):
        flash('Primero debes cargar un archivo CSV para registrar observaciones.', 'warning')
        return redirect(url_for('main.index'))
    page_title = 'Registrar Observación'
    current_filename = session.get('uploaded_filename')
    # Prefill desde el selector de entidad (opcional)
    pre_tipo = request.args.get('tipo_entidad')
    pre_nombre = request.args.get('nombre_entidad')
    return render_template('registrar_observacion.html', page_title=page_title, filename=current_filename,
                           pre_tipo_entidad=pre_tipo, pre_nombre_entidad=pre_nombre)

@main_bp.route('/api/add_observacion_entidad', methods=['POST'])
def api_add_observacion_entidad():
    try:
        data = request.json or {}
        entity_type = data.get('tipo_entidad')
        entity_name = data.get('nombre_entidad')
        observer_name = data.get('observador_nombre')
        observation_text = data.get('observacion_texto')
        current_csv_filename = session.get('uploaded_filename')

        if not all([entity_type, entity_name, observer_name, observation_text, current_csv_filename]):
            return jsonify({"error": "Faltan datos para guardar la observación."}), 400

        db_path = current_app.config['DATABASE_FILE']
        ok = save_observation_for_entity(db_path, entity_type, entity_name, observer_name, observation_text, current_csv_filename)
        if not ok:
            return jsonify({"error": "Error interno al guardar la observación."}), 500

        updated_index = False
        if embedding_model_instance:
            updated_index = reload_followup_vector_store(current_app.config)

        msg = 'Observación guardada.'
        if embedding_model_instance and updated_index:
            msg += ' Índice RAG actualizado.'
        elif not embedding_model_instance:
            msg += ' Índice RAG no actualizado (modelo embeddings no disponible).'
        elif embedding_model_instance and not updated_index:
            msg += ' Hubo un error al actualizar el índice RAG.'

        return jsonify({"message": msg}), 201
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Excepción al guardar la observación: {str(e)}"}), 500

@main_bp.route('/ver_observacion/<int:obs_id>')
def ver_observacion(obs_id):
    db_path = current_app.config['DATABASE_FILE']
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM follow_ups WHERE id = ? AND follow_up_type = 'observacion_entidad'", (obs_id,))
            obs_row = cursor.fetchone()
        if not obs_row:
            flash('No se encontró la observación solicitada.', 'warning')
            return redirect(url_for('main.biblioteca_reportes'))

        observacion_markdown = obs_row['follow_up_comment']
        observacion_html = markdown.markdown(observacion_markdown, extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists'])
        observador_nombre = str(obs_row['related_prompt']).replace('Observación de:', '').strip() if obs_row['related_prompt'] else 'N/D'
        # Convertir timestamp a horario local
        try:
            naive_dt = datetime.datetime.strptime(obs_row['timestamp'], '%Y-%m-%d %H:%M:%S')
            utc_dt = pytz.utc.localize(naive_dt)
            santiago_dt = utc_dt.astimezone(_get_tz())
            ts_form = santiago_dt.strftime('%d/%m/%Y %H:%M')
        except Exception:
            ts_form = obs_row.get('timestamp', '')

        return render_template('ver_observacion.html',
                               page_title=f"Observación - {obs_row['related_entity_name']}",
                               observacion_html=observacion_html,
                               observacion_markdown=observacion_markdown,
                               observador_nombre=observador_nombre,
                               tipo_entidad=obs_row['related_entity_type'],
                               nombre_entidad=obs_row['related_entity_name'],
                               timestamp_formateado=ts_form)
    except Exception as e:
        traceback.print_exc()
        flash(f'Error al cargar la observación: {e}', 'danger')
        return redirect(url_for('main.biblioteca_reportes'))

@main_bp.route('/reporte_360/ver/<int:reporte_id>')
def ver_reporte_360(reporte_id):
    db_path = current_app.config['DATABASE_FILE']
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM follow_ups WHERE id = ? AND follow_up_type = 'reporte_360'", (reporte_id,))
            report_data = cursor.fetchone()

        if report_data:
            reporte_html = markdown.markdown(report_data['follow_up_comment'])
            observaciones = get_observations_for_reporte_360(db_path, reporte_id)
            
            # --- INICIO: MODIFICACIÓN ---
            # Formatear el timestamp histórico, convirtiendo de UTC a Santiago
            try:
                naive_dt = datetime.datetime.strptime(report_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                utc_dt = pytz.utc.localize(naive_dt)
                santiago_dt = utc_dt.astimezone(_get_tz())
                timestamp_formateado = santiago_dt.strftime('%d/%m/%Y %H:%M')
            except (ValueError, TypeError):
                timestamp_formateado = report_data.get('timestamp', 'Fecha no válida')
            # --- FIN: MODIFICACIÓN ---

            session['reporte_360_markdown'] = report_data['follow_up_comment']
            session['reporte_360_entidad_tipo'] = report_data['related_entity_type']
            session['reporte_360_entidad_nombre'] = report_data['related_entity_name']
            session['current_reporte_360_id'] = reporte_id

            return render_template('reporte_360.html',
                                   page_title=f"Reporte 360 Histórico - {report_data['related_entity_name']}",
                                   tipo_entidad=report_data['related_entity_type'],
                                   nombre_entidad=report_data['related_entity_name'],
                                   reporte_html=reporte_html,
                                   reporte_360_id=reporte_id,
                                   observaciones_reporte=observaciones,
                                   filename=report_data['related_filename'],
                                   timestamp_generacion=timestamp_formateado) # <-- Variable añadida
        else:
            flash('No se encontró el Reporte 360 solicitado.', 'warning')
            return redirect(url_for('main.biblioteca_reportes'))
    except Exception as e:
        flash(f'Error al cargar el reporte histórico: {e}', 'danger')
        traceback.print_exc()
        return redirect(url_for('main.biblioteca_reportes'))
    
@main_bp.route('/api/get_context_docs')
def api_get_context_docs():
    context_folder = current_app.config['CONTEXT_DOCS_FOLDER']
    try:
        if os.path.exists(context_folder) and os.path.isdir(context_folder):
            docs = [f for f in os.listdir(context_folder) if f.lower().endswith(('.pdf', '.txt'))]
            return jsonify(sorted(docs))
        return jsonify([])
    except Exception as e:
        return jsonify({"error": f"Error al leer la carpeta de documentos: {e}"}), 500

@main_bp.route('/api/delete_context_doc', methods=['POST'])
def api_delete_context_doc():
    data = request.json
    filename = data.get('filename')
    if not filename:
        return jsonify({"error": "No se proporcionó el nombre del archivo."}), 400

    safe_filename = secure_filename(filename)
    if safe_filename != filename:
        return jsonify({"error": "Nombre de archivo no válido."}), 400

    context_folder = current_app.config['CONTEXT_DOCS_FOLDER']
    file_path = os.path.join(context_folder, safe_filename)

    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            # Crucial: Re-indexar el RAG después de eliminar un archivo
            if reload_institutional_context_vector_store(current_app.config):
                return jsonify({"message": f"Archivo '{safe_filename}' eliminado y el índice de contexto ha sido actualizado."})
            else:
                return jsonify({"message": f"Archivo '{safe_filename}' eliminado, pero hubo un error al actualizar el índice de contexto."}), 500
        else:
            return jsonify({"error": "El archivo no fue encontrado."}), 404
    except Exception as e:
        return jsonify({"error": f"Error al eliminar el archivo: {e}"}), 500

# --- NUEVO: Endpoint para totales de consumo de tokens y costos (hoy y mes) ---
@main_bp.route('/api/consumo/totales')
def api_consumo_totales():
    """Devuelve los totales acumulados de tokens y costos para el día actual y el mes actual.
    Respuesta JSON con dos bloques: 'dia' y 'mes'. Cada bloque incluye tokens_subida,
    tokens_bajada, total_tokens y costo_total.
    """
    try:
        db_path = current_app.config['DATABASE_FILE']
        hoy = datetime.date.today()
        fecha_hoy = hoy.isoformat()
        inicio_mes = hoy.replace(day=1).isoformat()

        def _sum_query(conn, fecha_ini: str, fecha_fin: str):
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 
                    COALESCE(SUM(tokens_subida), 0) as subida,
                    COALESCE(SUM(tokens_bajada), 0) as bajada,
                    COALESCE(SUM(costo_total), 0.0) as costo
                FROM consumo_tokens_diario
                WHERE fecha >= ? AND fecha <= ?
                """,
                (fecha_ini, fecha_fin)
            )
            row = cur.fetchone()
            subida = int(row[0] or 0)
            bajada = int(row[1] or 0)
            costo = float(row[2] or 0.0)
            return {
                'tokens_subida': subida,
                'tokens_bajada': bajada,
                'total_tokens': subida + bajada,
                'costo_total': costo
            }

        with sqlite3.connect(db_path) as conn:
            totals_dia = _sum_query(conn, fecha_hoy, fecha_hoy)
            totals_mes = _sum_query(conn, inicio_mes, fecha_hoy)

        return jsonify({'dia': totals_dia, 'mes': totals_mes})
    except Exception as e:
        current_app.logger.exception(f"Error al obtener totales de consumo: {e}")
        return jsonify({"error": "No se pudieron obtener los totales de consumo."}), 500
@main_bp.route('/efectividad', methods=['GET'])
def efectividad_intervenciones():
    try:
        db_path = current_app.config['DATABASE_FILE']
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT related_entity_name FROM follow_ups WHERE follow_up_type = 'intervention_plan' AND related_entity_type = 'alumno'")
            rows = cur.fetchall()
            alumnos = [r['related_entity_name'] for r in rows]
        data = []
        for nombre in alumnos:
            grade_delta = get_student_grade_evolution_value(db_path, nombre)
            att_delta = get_student_attendance_evolution_value(db_path, nombre)
            data.append({'student': nombre, 'grade_delta': grade_delta, 'attendance_delta': att_delta})
        return render_template('efectividad_intervenciones.html', data=data)
    except Exception as e:
        current_app.logger.exception(f"Error en /efectividad: {e}")
        flash('Error al construir la vista de efectividad.', 'danger')
        return redirect(url_for('main.index'))

@main_bp.route('/api/train_risk_model', methods=['POST'])
def api_train_risk_model():
    try:
        metrics = train_predictive_risk_model(current_app.config['DATABASE_FILE'], current_app.config['MODEL_ARTIFACTS_DIR'])
        if metrics is None:
            return jsonify({'error': 'no_data'}), 400
        return jsonify(metrics), 200
    except Exception as e:
        current_app.logger.exception(f"Error en /api/train_risk_model: {e}")
        return jsonify({'error': 'server_error'}), 500

@main_bp.route('/api/reset_database', methods=['POST'])
def api_reset_database():
    try:
        flags = request.json or {}
        db_path = current_app.config['DATABASE_FILE']
        ok = reset_database_tables(db_path)
        if not ok:
            return jsonify({'error': 'reset_failed'}), 500
        if bool(flags.get('remove_artifacts')):
            try:
                import shutil
                artifacts_dir = current_app.config.get('MODEL_ARTIFACTS_DIR')
                if artifacts_dir and os.path.isdir(artifacts_dir):
                    for name in os.listdir(artifacts_dir):
                        p = os.path.join(artifacts_dir, name)
                        if os.path.isdir(p):
                            shutil.rmtree(p, ignore_errors=True)
                        else:
                            try:
                                os.remove(p)
                            except Exception:
                                pass
            except Exception:
                pass
        if bool(flags.get('remove_vectorstores')):
            try:
                import shutil
                vp = current_app.config.get('FAISS_INDEX_PATH')
                vpf = current_app.config.get('FAISS_FOLLOWUP_INDEX_PATH')
                for d in [vp, vpf]:
                    if d and os.path.isdir(d):
                        shutil.rmtree(d, ignore_errors=True)
                try:
                    reload_institutional_context_vector_store()
                except Exception:
                    pass
                try:
                    reload_followup_vector_store()
                except Exception:
                    pass
            except Exception:
                pass
        return jsonify({'status': 'ok'}), 200
    except Exception as e:
        current_app.logger.exception(f"Error en /api/reset_database: {e}")
        return jsonify({'error': 'server_error'}), 500
