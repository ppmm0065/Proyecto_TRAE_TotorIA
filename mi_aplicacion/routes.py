# mi_aplicacion/routes.py
import os
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
    get_observations_for_reporte_360
)
import sqlite3

main_bp = Blueprint('main', __name__)

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
            
            # --- CÁLCULO PARA GRÁFICO DE DISTRIBUCIÓN DE PROMEDIOS ---
            average_distribution_data = {'labels': [], 'counts': []}
            if not df_alumnos[promedio_col].isnull().all():
                promedios_validos = df_alumnos[promedio_col]
                bins = [2.0, 3.0, 4.0, 5.0, 6.0, 7.01]; labels = ['2.0-2.9', '3.0-3.9', '4.0-4.9', '5.0-5.9', '6.0-7.0']
                try:
                    dist_counts = pd.cut(promedios_validos, bins=bins, labels=labels, right=False, include_lowest=True).value_counts().sort_index()
                    average_distribution_data['labels'] = dist_counts.index.tolist(); average_distribution_data['counts'] = dist_counts.values.tolist()
                except Exception as e: print(f"Error en rangos de distribución de promedios: {e}")

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
    # No changes to this function
    if not session.get('current_file_path'): flash('Carga un archivo CSV.', 'warning'); return redirect(url_for('main.index'))
    if 'advanced_chat_history' not in session: session['advanced_chat_history'] = []
    return render_template('advanced_chat.html', page_title="Chat Avanzado", filename=session.get('uploaded_filename'), advanced_chat_history=session.get('advanced_chat_history', []))

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
    reportes_360_con_observaciones = [] # (La lógica para obtener reportes y observaciones no cambia)

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
                
                context['datos_dashboard']['info_general'] = {
                    'Promedio': f"{alumno_promedio_general:.2f}",
                    'Curso': nombre_curso_alumno,
                    'Edad': alumno_data_row.get(current_app.config.get('EDAD_COL'), 'N/A'),
                    'AsignaturaMenorPromedio': f"{peor_asignatura_key} ({peor_asignatura_nota:.1f})"
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

                context['datos_dashboard']['info_general'] = {
                    'NumeroAlumnos': len(df_alumnos_unicos_curso),
                    'PromedioGeneralCurso': f"{promedio_general_curso:.2f}",
                    'PeorAsignatura': f"{peor_asignatura} ({peor_asignatura_prom:.2f})",
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
    df_entidad = pd.DataFrame()
    
    try:
        nombre_entidad_normalizado = nombre_entidad.strip().lower()
        if tipo_entidad == 'curso':
            df_entidad = df_global[df_global[current_app.config['CURSO_COL']].astype(str).str.strip().str.lower() == nombre_entidad_normalizado]
        elif tipo_entidad == 'alumno':
            df_entidad = df_global[df_global[current_app.config['NOMBRE_COL']].astype(str).str.strip().str.lower() == nombre_entidad_normalizado]
    except Exception as e:
        return jsonify({"error": f"Error al filtrar datos para la entidad: {str(e)}"}), 500

    if not df_entidad.empty:
        nombre_col = current_app.config['NOMBRE_COL']
        nota_col = current_app.config['NOTA_COL']
        asignatura_col = current_app.config['ASIGNATURA_COL']
        promedio_col = current_app.config['PROMEDIO_COL']

        # --- INICIO: Motor de Intenciones CORREGIDO Y REORDENADO ---

        # Intención 1 (Más específica): Mejor/Peor ALUMNO de un CURSO
        if tipo_entidad == 'curso' and ('alumno' in prompt_lower or 'estudiante' in prompt_lower) and any(kw in prompt_lower for kw in ['mejor', 'peor', 'mayor', 'menor', 'promedio']):
            df_alumnos_unicos = df_entidad.drop_duplicates(subset=[nombre_col])
            if not df_alumnos_unicos.empty:
                if any(kw in prompt_lower for kw in ['mejor', 'mayor']):
                    mejor_alumno = df_alumnos_unicos.loc[df_alumnos_unicos[promedio_col].idxmax()]
                    response_md = f"El alumno con el mejor promedio en {nombre_entidad} es **{mejor_alumno[nombre_col]}** con un **{mejor_alumno[promedio_col]:.2f}**."
                    return jsonify(crear_respuesta_directa(response_md))
                elif any(kw in prompt_lower for kw in ['peor', 'menor']):
                    peor_alumno = df_alumnos_unicos.loc[df_alumnos_unicos[promedio_col].idxmin()]
                    response_md = f"El alumno con el peor promedio en {nombre_entidad} es **{peor_alumno[nombre_col]}** con un **{peor_alumno[promedio_col]:.2f}**."
                    return jsonify(crear_respuesta_directa(response_md))

        # Intención 2: Mejor/Peor ASIGNATURA (para CURSO o ALUMNO)
        if any(kw in prompt_lower for kw in ['asignatura', 'materia', 'ramo']) and any(kw in prompt_lower for kw in ['peor', 'mejor', 'más alta', 'más baja']):
            promedios_asignaturas = df_entidad.groupby(asignatura_col)[nota_col].mean().dropna()
            if not promedios_asignaturas.empty:
                if any(kw in prompt_lower for kw in ['mejor', 'más alta']):
                    mejor_key = promedios_asignaturas.idxmax()
                    response_md = f"La asignatura con el mejor rendimiento para **{nombre_entidad}** es **{mejor_key}** con un promedio de **{promedios_asignaturas.max():.2f}**."
                    return jsonify(crear_respuesta_directa(response_md))
                elif any(kw in prompt_lower for kw in ['peor', 'más baja']):
                    peor_key = promedios_asignaturas.idxmin()
                    response_md = f"La asignatura con el rendimiento más bajo para **{nombre_entidad}** es **{peor_key}** con un promedio de **{promedios_asignaturas.min():.2f}**."
                    return jsonify(crear_respuesta_directa(response_md))

        # Intención 2.1: Edad de un Alumno
        elif tipo_entidad == 'alumno' and 'edad' in prompt_lower:
            edad_col = current_app.config.get('EDAD_COL')
            if edad_col and edad_col in df_entidad.columns:
                edad = df_entidad.iloc[0][edad_col]
                if pd.notna(edad):
                    response_md = f"**{nombre_entidad}** tiene **{int(edad)} años**."
                    return jsonify(crear_respuesta_directa(response_md))
                
        # Intención 3 (Más general): Promedio de un CURSO
        if tipo_entidad == 'curso' and ('promedio' in prompt_lower or 'media' in prompt_lower):
            promedio_curso = df_entidad[nota_col].mean()
            response_md = f"El promedio general del curso **{nombre_entidad}** es **{promedio_curso:.2f}**."
            return jsonify(crear_respuesta_directa(response_md))

        # Intención 4: Promedio de un ALUMNO
        if tipo_entidad == 'alumno' and ('promedio' in prompt_lower or 'media' in prompt_lower):
            promedio_alumno = df_entidad.iloc[0][promedio_col]
            response_md = f"El promedio general de **{nombre_entidad}** es **{promedio_alumno:.2f}**."
            return jsonify(crear_respuesta_directa(response_md))

        # Intención 5: Contar alumnos en un CURSO
        if tipo_entidad == 'curso' and ('alumnos' in prompt_lower or 'estudiantes' in prompt_lower) and any(kw in prompt_lower for kw in ['cuántos', 'cantidad', 'número', 'total']):
            count = df_entidad[nombre_col].nunique()
            response_md = f"El curso **{nombre_entidad}** tiene **{count}** alumnos."
            return jsonify(crear_respuesta_directa(response_md))

        # --- FIN: Motor de Intenciones ---

    # Si ninguna intención simple coincide, se procede con la consulta a Gemini.
    data_string_especifico = load_data_as_string(session.get('current_file_path'), specific_entity_df=df_entidad)
    chat_history_key = f'chat_history_detalle_{tipo_entidad}_{nombre_entidad}'
    chat_history_list_detalle = session.get(chat_history_key, [])
    chat_history_string_detalle = format_chat_history_for_prompt(chat_history_list_detalle)
    analysis_result = analyze_data_with_gemini(
        data_string_especifico, user_prompt, vector_store, vector_store_followups,
        chat_history_string=chat_history_string_detalle, is_direct_chat_query=True,
        entity_type=tipo_entidad, entity_name=nombre_entidad
    )
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
    if not session.get('current_file_path'): return jsonify({"error": "No hay archivo CSV."}), 400
    data = request.json; user_prompt = data.get('prompt')
    if not user_prompt: return jsonify({"error": "No se recibió prompt."}), 400

    if 'consumo_sesion' not in session:
        session['consumo_sesion'] = {'total_tokens': 0, 'total_cost': 0.0}
    
    df_global = get_dataframe_from_session_file()
    if df_global is None or df_global.empty:
        return jsonify({"error": "No se pudo cargar el DataFrame."}), 500

    prompt_lower = user_prompt.lower()
    data_string = ""
    entity_type = None
    entity_name = None
    
    # Obtenemos la ruta del archivo una sola vez
    file_path = session.get('current_file_path')

    nombre_col = current_app.config['NOMBRE_COL']
    curso_col = current_app.config['CURSO_COL']
    student_names = df_global[nombre_col].unique().tolist()
    course_names = df_global[curso_col].unique().tolist()

    found_student = next((name for name in student_names if name.lower() in prompt_lower), None)
    if found_student:
        entity_type = 'alumno'
        entity_name = found_student
        print(f"DEBUG: Chat Avanzado detectó entidad ALUMNO: {entity_name}")
        df_entidad = df_global[df_global[nombre_col] == entity_name]
        # --- LÍNEA CORREGIDA ---
        # Pasamos el file_path requerido como primer argumento.
        data_string = load_data_as_string(file_path, specific_entity_df=df_entidad)
    else:
        found_course = next((name for name in course_names if name.lower() in prompt_lower), None)
        if found_course:
            entity_type = 'curso'
            entity_name = found_course
            print(f"DEBUG: Chat Avanzado detectó entidad CURSO: {entity_name}")
            df_entidad = df_global[df_global[curso_col] == entity_name]
            # --- LÍNEA CORREGIDA ---
            # Pasamos el file_path requerido como primer argumento.
            data_string = load_data_as_string(file_path, specific_entity_df=df_entidad)

    if not data_string:
        print("DEBUG: Chat Avanzado detectó consulta GENERAL. Usando resumen del dashboard.")
        file_summary = session.get('file_summary', {})
        import json
        data_string = "Contexto: A continuación se presenta un resumen estadístico general del colegio, no el listado completo de alumnos.\n\n"
        data_string += json.dumps(file_summary, indent=2, ensure_ascii=False)
    
    history_list = session.get('advanced_chat_history', [])
    history_fmt = format_chat_history_for_prompt(history_list)
    
    analysis_result = analyze_data_with_gemini(
        data_string, user_prompt, vector_store, vector_store_followups, history_fmt, 
        is_direct_chat_query=True, entity_type=entity_type, entity_name=entity_name
    )
    
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

    return render_template('reporte_360.html', 
                           page_title=f"Reporte 360 - {nombre_entidad}", 
                           tipo_entidad=tipo_entidad, 
                           nombre_entidad=nombre_entidad, 
                           reporte_html=reporte_html, 
                           reporte_360_id=reporte_360_id, 
                           observaciones_reporte=observaciones_del_reporte,
                           filename=current_csv_filename)

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
    session['current_intervention_plan_date'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
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
                # --- LÍNEA CORREGIDA: Se eliminó el filtro AND related_filename = ? ---
                cursor.execute("""
                    SELECT timestamp, follow_up_comment, related_filename FROM follow_ups 
                    WHERE id = ? AND follow_up_type = 'intervention_plan' 
                    AND related_entity_type = ? AND related_entity_name = ?
                    """, 
                               (plan_id_to_load, tipo_entidad, nombre_entidad))
                plan_data = cursor.fetchone()
                if plan_data:
                    plan_html_content = markdown.markdown(plan_data['follow_up_comment'])
                    plan_date = datetime.datetime.strptime(plan_data["timestamp"], '%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y %H:%M:%S')
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
    reportes = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Seleccionamos solo los reportes asociados al archivo CSV activo
            cursor.execute("""
                SELECT id, timestamp, report_date, follow_up_type, related_entity_type, related_entity_name
                FROM follow_ups
                WHERE (follow_up_type = 'reporte_360' OR follow_up_type = 'intervention_plan')
                AND related_filename = ?
                ORDER BY timestamp DESC
            """, (current_filename,))
            reportes = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        flash(f'Error al cargar la biblioteca de reportes: {e}', 'danger')
        traceback.print_exc()

    return render_template('biblioteca.html',
                           page_title="Biblioteca de Reportes",
                           reportes=reportes,
                           filename=current_filename)

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
            
            # Guardamos en sesión para mantener el flujo contextual
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
                                   filename=report_data['related_filename'])
        else:
            flash('No se encontró el Reporte 360 solicitado.', 'warning')
            return redirect(url_for('main.biblioteca_reportes'))
    except Exception as e:
        flash(f'Error al cargar el reporte histórico: {e}', 'danger')
        traceback.print_exc()
        return redirect(url_for('main.biblioteca_reportes'))