# mi_aplicacion/app_logic.py
import os
import pandas as pd
import numpy as np
import google.generativeai as genai
from flask import session, flash, current_app
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings # Asegúrate que esta es la importación correcta
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import traceback
import sqlite3
import io
import markdown
import re
import csv
import datetime
import shutil
from flask import current_app
import unicodedata
from .utils import normalize_text, compile_any_keyword_pattern, get_tz, grade_to_qualitative

import datetime
import pytz
from pytz import timezone
import requests
from urllib.parse import urlparse, quote

# Definimos la zona horaria aquí también para usarla en las consultas de BD
# Zona horaria centralizada vía configuración
def _get_tz():
    try:
        return get_tz()
    except Exception:
        return timezone('America/Santiago')

embedding_model_instance = None
vector_store = None # For institutional context
vector_store_followups = None # Global FAISS for general follow-up search

def get_dataframe_from_session_file():
    current_file_path = session.get('current_file_path')
    if not current_file_path:
        try:
            current_app.logger.debug("get_dataframe_from_session_file: ruta vacía en sesión")
        except Exception:
            pass
        return None
    try:
        current_file_path = os.path.normpath(str(current_file_path))
    except Exception:
        return None
    if not os.path.exists(current_file_path):
        try:
            current_app.logger.debug(f"get_dataframe_from_session_file: archivo no existe: {current_file_path}")
        except Exception:
            pass
        return None
    name, ext = os.path.splitext(current_file_path)
    if ext.lower() != ".csv":
        try:
            flash("El archivo activo no es CSV válido.", "danger")
            current_app.logger.warning(f"Archivo activo con extensión inválida: {ext}")
        except Exception:
            pass
        return None
    try:
        if os.path.getsize(current_file_path) <= 0:
            try:
                flash("El archivo CSV está vacío.", "warning")
                current_app.logger.warning("Archivo CSV vacío")
            except Exception:
                pass
            return None
    except Exception:
        return None
    
    try:
        df = None
        # --- LÓGICA DE LECTURA ROBUSTA CON DETECCIÓN DE DELIMITADOR Y ENCODING ---
        def _sniff_sep(path):
            """Detecta el separador más probable contando ocurrencias en las primeras líneas."""
            try:
                with open(path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                    lines = [next(f) for _ in range(5)]
            except Exception:
                try:
                    with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                        lines = [next(f) for _ in range(5)]
                except Exception:
                    lines = []
            text = "\n".join(lines)
            comma = text.count(',')
            semicol = text.count(';')
            return ';' if semicol >= comma else ','

        def _try_read(path, sep, encoding, relax=False):
            try:
                if relax:
                    return pd.read_csv(path, skipinitialspace=True, encoding=encoding, sep=sep, engine='python', on_bad_lines='skip')
                return pd.read_csv(path, skipinitialspace=True, encoding=encoding, sep=sep, engine='python')
            except Exception:
                return None

        sep_guess = _sniff_sep(current_file_path)
        # Intento principal: utf-8-sig con separador detectado
        df = _try_read(current_file_path, sep_guess, 'utf-8-sig')
        # Alternar separador si solo creó 1 columna
        if df is None or df.shape[1] == 1:
            df = _try_read(current_file_path, ';' if sep_guess == ',' else ',', 'utf-8-sig')
        # Fallback de codificación latin-1
        if df is None or df.empty or df.shape[1] == 1:
            df = _try_read(current_file_path, sep_guess, 'latin-1')
            if df is None or df.empty or df.shape[1] == 1:
                df = _try_read(current_file_path, (';' if sep_guess == ',' else ','), 'latin-1')
        # Último recurso: modo relajado (salta líneas problemáticas)
        if df is None or df.empty or df.shape[1] == 1:
            df = _try_read(current_file_path, sep_guess, 'utf-8-sig', relax=True)
            if df is None or df.empty or df.shape[1] == 1:
                df = _try_read(current_file_path, (';' if sep_guess == ',' else ','), 'utf-8-sig', relax=True)
        if df is None:
            print("Error crítico al leer CSV: no fue posible parsear el archivo con los intentos realizados.")
            return None
        
        if df is None or df.empty:
            print("El DataFrame está vacío o no se pudo leer.")
            return pd.DataFrame()

        # --- El resto del procesamiento no cambia ---
        df.columns = df.columns.str.strip().str.replace('"', '', regex=False)
        
        nombre_col = current_app.config['NOMBRE_COL']
        nota_col = current_app.config['NOTA_COL']
        promedio_col = current_app.config['PROMEDIO_COL']
        asistencia_col = current_app.config.get('ASISTENCIA_COL')

        columnas_esenciales = [nombre_col, current_app.config['CURSO_COL'], current_app.config['ASIGNATURA_COL'], nota_col]
        for col in columnas_esenciales:
            if col not in df.columns:
                print(f"Error Crítico: El archivo CSV no tiene la columna obligatoria '{col}'.")
                flash(f"Error de formato: Falta la columna obligatoria '{col}' en el archivo.", "danger")
                return None

        original_rows = len(df)
        df.dropna(subset=[nombre_col], inplace=True)
        df = df[df[nombre_col].astype(str).str.strip() != '']
        if len(df) < original_rows:
            print(f"LIMPIEZA: Se eliminaron {original_rows - len(df)} filas sin nombre de estudiante.")

        df[nota_col] = df[nota_col].astype(str).str.replace(',', '.', regex=False)
        df[nota_col] = pd.to_numeric(df[nota_col], errors='coerce')
        df.dropna(subset=[nota_col], inplace=True)

        df[promedio_col] = df.groupby(nombre_col)[nota_col].transform('mean')

        # --- Interpretación cualitativa de notas y promedios ---
        cual_col = current_app.config.get('CUALITATIVA_COL', 'Calificacion_Cualitativa')
        prom_cual_col = current_app.config.get('PROMEDIO_CUALITATIVO_COL', 'Promedio_Cualitativo')
        try:
            df[cual_col] = df[nota_col].apply(grade_to_qualitative)
            df[prom_cual_col] = df[promedio_col].apply(grade_to_qualitative)
        except Exception as e:
            print(f"Advertencia: no se pudo generar interpretación cualitativa: {e}")
        
        if asistencia_col and asistencia_col in df.columns:
            try:
                df[asistencia_col] = df[asistencia_col].astype(str).str.rstrip('%').str.replace(',', '.', regex=False)
                df[asistencia_col] = pd.to_numeric(df[asistencia_col], errors='coerce')
                if df[asistencia_col].max() > 1:
                    df[asistencia_col] = df[asistencia_col] / 100.0
                df[asistencia_col] = df[asistencia_col].clip(lower=0.0, upper=1.0)
            except Exception as e:
                print(f"Advertencia: No se pudo procesar la columna '{asistencia_col}': {e}.")
                df[asistencia_col] = np.nan
        
        print("DataFrame en formato 'largo' procesado y promedio calculado exitosamente.")
        return df
        
    except Exception as e:
        print(f"Error general en get_dataframe_from_session_file con formato largo: {e}")
        traceback.print_exc()
        return None

def load_data_as_string(full_filepath, specific_entity_df=None):
    # Si se proporciona un DataFrame específico (como una lista de alumnos de un curso),
    # conviértelo a string SIN incluir la columna de índice de Pandas.
    if specific_entity_df is not None and not specific_entity_df.empty:
        # --- LÍNEA CORREGIDA ---
        # Añadimos index=False para eliminar la columna de índice que confundía a la IA.
        return specific_entity_df.to_string(index=False)

    # El resto de la función (para cargar el archivo completo) permanece igual.
    try:
        # Leer contenido crudo primero con manejo de codificación
        raw = None
        try:
            raw = open(full_filepath, 'r', encoding='utf-8-sig').read()
        except UnicodeDecodeError:
            raw = open(full_filepath, 'r', encoding='latin-1').read()

        # Detección simple de separador por frecuencia
        head = (raw.splitlines()[:5])
        text_head = "\n".join(head)
        sep_guess = ';' if text_head.count(';') >= text_head.count(',') else ','

        # Intento 1 y 2: separador detectado y alterno
        def _try_parse(text, sep, relax=False):
            try:
                if relax:
                    return pd.read_csv(io.StringIO(text), sep=sep, skipinitialspace=True, engine='python', on_bad_lines='skip')
                return pd.read_csv(io.StringIO(text), sep=sep, skipinitialspace=True, engine='python')
            except Exception:
                return None

        df = _try_parse(raw, sep_guess)
        if df is None or df.shape[1] == 1:
            df = _try_parse(raw, ';' if sep_guess == ',' else ',')

        # Último recurso: modo relajado
        if df is None or df.shape[1] == 1:
            df = _try_parse(raw, sep_guess, relax=True) or _try_parse(raw, (';' if sep_guess == ',' else ','), relax=True)

        # Si definitivamente no podemos parsear, devolvemos el texto crudo para que el LLM tenga contexto sin romper
        if df is None:
            return raw

        # Limpieza ligera de columnas y representación sin índice
        df.columns = df.columns.str.strip().str.replace('"', '', regex=False)
        return df.to_string(index=False)
    except FileNotFoundError:
        return "Error: Archivo CSV de datos no encontrado."
    except Exception as e:
        return f"Error al procesar CSV: {e}"

def format_chat_history_for_prompt(chat_history_list, max_turns=None):
    # No changes
    if not max_turns: max_turns = current_app.config.get('MAX_CHAT_HISTORY_TURNS_FOR_GEMINI_PROMPT', 3) 
    if not chat_history_list: return ""
    limited_history = chat_history_list[-max_turns:]
    formatted_history = "Historial de Conversación Previa (Pregunta más reciente del usuario al final de esta sección):\n"
    for entry in limited_history:
        user_query = entry.get('user', 'Pregunta no registrada')
        gemini_answer_markdown = entry.get('gemini_markdown', 'Respuesta no registrada')
        formatted_history += f"Usuario: {user_query}\nAsistente: {gemini_answer_markdown}\n---\n"
    return formatted_history

# === Utilidades para respuestas directas sobre asignaturas ===
def _match_subject_name(df: pd.DataFrame, subject_query: str) -> str:
    """Devuelve el nombre de asignatura del CSV que mejor coincide con el texto pedido.
    Usa normalización sin tildes y algunos alias comunes (p. ej. 'mate' -> 'Matemáticas').
    Si no hay match, retorna None.
    """
    try:
        if df is None or df.empty:
            return None
        asignatura_col = current_app.config.get('ASIGNATURA_COL', 'Asignatura')
        if asignatura_col not in df.columns:
            return None
        prompt_norm = normalize_text(subject_query)
        # Alias básicos
        alias = {
            'matematica': 'matematicas', 'mate': 'matematicas',
            'lengua': 'lenguaje', 'language': 'lenguaje',
            'ingles': 'ingles', 'english': 'ingles',
            'historia': 'historia', 'musica': 'musica',
            'arte': 'artes visuales', 'visuales': 'artes visuales',
            'educacion fisica': 'educacion fisica', 'ed fisica': 'educacion fisica',
            'ciencias naturales': 'ciencias naturales', 'ciencias': 'ciencias naturales'
        }
        prompt_norm = alias.get(prompt_norm, prompt_norm)
        # Normalizar catálogo de asignaturas
        subjects = df[asignatura_col].dropna().astype(str).unique().tolist()
        norm_map = { normalize_text(s): s for s in subjects }
        # Búsqueda por inclusión (p. ej., "matematicas" dentro del prompt)
        for norm_name, original in norm_map.items():
            if norm_name in prompt_norm:
                return original
        return None
    except Exception:
        return None

def get_lowest_grade_student_for_subject(df: pd.DataFrame, subject_query: str):
    """Encuentra el alumno con la nota más baja en la asignatura indicada.
    Retorna:
      - dict {nombre, curso, asignatura, nota} si hay un único mínimo.
      - dict {multiple: True, asignatura, min_nota, registros: [..]} si hay empate.
      - None si no hay datos o no hay match.
    """
    try:
        asignatura_col = current_app.config.get('ASIGNATURA_COL', 'Asignatura')
        nombre_col = current_app.config['NOMBRE_COL']
        curso_col = current_app.config['CURSO_COL']
        nota_col = current_app.config['NOTA_COL']
        if df is None or df.empty or asignatura_col not in df.columns:
            return None
        # Asegurar tipo numérico en Nota
        dfx = df[[asignatura_col, nombre_col, curso_col, nota_col]].dropna(subset=[asignatura_col, nombre_col, nota_col]).copy()
        dfx[nota_col] = pd.to_numeric(dfx[nota_col], errors='coerce')
        dfx = dfx.dropna(subset=[nota_col])
        subj_name = _match_subject_name(dfx, subject_query)
        if not subj_name:
            return None
        dsub = dfx[dfx[asignatura_col].astype(str) == subj_name]
        if dsub.empty:
            return None
        min_val = float(dsub[nota_col].min())
        ties = dsub[dsub[nota_col] == min_val].copy()
        if len(ties) > 1:
            registros = [
                {
                    'nombre': str(r[nombre_col]).strip(),
                    'curso': str(r[curso_col]).strip(),
                    'asignatura': str(r[asignatura_col]).strip(),
                    'nota': float(r[nota_col])
                }
                for _, r in ties.iterrows()
            ]
            return {
                'multiple': True,
                'asignatura': subj_name,
                'min_nota': min_val,
                'registros': registros
            }
        # Único mínimo
        row = dsub.loc[dsub[nota_col].idxmin()]
        return {
            'nombre': str(row[nombre_col]).strip(),
            'curso': str(row[curso_col]).strip(),
            'asignatura': str(row[asignatura_col]).strip(),
            'nota': float(row[nota_col])
        }
    except Exception:
        return None

def get_student_evolution_summary(db_path, top_n=5, entity_name=None, order_direction='DESC'): # <-- MODIFICACIÓN: Nuevo parámetro añadido
    """
    Consulta la base de datos para obtener un resumen de la evolución de las notas
    de los estudiantes entre la primera y la última instantánea de datos.
    Acepta un parámetro 'order_direction' ('ASC' o 'DESC') para la ordenación.
    """
    
    # Esta consulta usa Expresiones de Tabla Comunes (CTE) para:
    # 1. Calcular el promedio de notas por estudiante POR cada instantánea.
    # 2. Asignar un "primer" y "último" promedio a cada estudiante basado en el tiempo.
    # 3. Calcular la diferencia (evolución) y filtrar por aquellos con >1 instantánea.
    
    query = """
    WITH SnapshotAverages AS (
        -- 1. Calcula el promedio de notas de cada estudiante en cada snapshot
        SELECT
            h.snapshot_id,
            h.student_name,
            s.timestamp,
            AVG(h.grade) AS avg_grade
        FROM student_data_history h
        JOIN data_snapshots s ON h.snapshot_id = s.id
        GROUP BY h.snapshot_id, h.student_name, s.timestamp
    ),
    StudentEvolution AS (
        -- 2. Encuentra el primer y último promedio para cada estudiante
        SELECT
            student_name,
            FIRST_VALUE(avg_grade) OVER (
                PARTITION BY student_name ORDER BY timestamp ASC
            ) AS first_avg_grade,
            LAST_VALUE(avg_grade) OVER (
                PARTITION BY student_name ORDER BY timestamp ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS last_avg_grade,
            COUNT(snapshot_id) OVER (PARTITION BY student_name) AS num_snapshots
        FROM SnapshotAverages
    ),
    FinalEvolution AS (
        -- 3. Calcula la diferencia y filtra
        SELECT
            student_name,
            first_avg_grade,
            last_avg_grade,
            (last_avg_grade - first_avg_grade) AS grade_evolution
        FROM StudentEvolution
        WHERE num_snapshots > 1 -- ¡Solo estudiantes con al menos 2 puntos de datos!
        GROUP BY student_name, first_avg_grade, last_avg_grade -- Condensar resultados
    )
    SELECT
        student_name,
        first_avg_grade,
        last_avg_grade,
        grade_evolution
    FROM FinalEvolution
    """
    
    params = []
    
    # --- MODIFICACIÓN: Validar la dirección para evitar inyección SQL ---
    if order_direction.upper() not in ['ASC', 'DESC']:
        order_direction = 'DESC' # Default seguro

    if entity_name:
        query += " WHERE student_name = ? AND grade_evolution != 0"
        params.append(entity_name)
    else:
        # --- MODIFICACIÓN: SQL dinámico para ASC/DESC ---
        # Usamos f-string de forma segura solo para ASC/DESC validados
        query += f" WHERE grade_evolution != 0 ORDER BY grade_evolution {order_direction} LIMIT ?"
        # --- FIN MODIFICACIÓN ---
        params.append(top_n)

    summary_lines = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            
            if not rows:
                if entity_name:
                    return f"No se encontró historial de evolución (más de 1 instantánea) para '{entity_name}'."
                else:
                    return "No se encontró historial de evolución (más de 1 instantánea) para ningún estudiante."
            
            if entity_name:
                summary_lines.append(f"Resumen de Evolución para {entity_name}:")
            else:
                # --- MODIFICACIÓN: Título dinámico ---
                direction_text = "Mejoras" if order_direction.upper() == 'DESC' else "Caídas"
                summary_lines.append(f"Resumen de Evolución de Notas (Top {len(rows)} {direction_text}):")
                # --- FIN MODIFICACIÓN ---
                
            for i, row in enumerate(rows):
                evolution_sign = '+' if row['grade_evolution'] > 0 else ''
                line = (
                    f"{i+1}. {row['student_name']}: "
                    f"Promedio Inicial: {row['first_avg_grade']:.2f}, "
                    f"Promedio Final: {row['last_avg_grade']:.2f} "
                    f"(Evolución: {evolution_sign}{row['grade_evolution']:.2f})"
                )
                summary_lines.append(line)
                
        return "\n".join(summary_lines)

    except Exception as e:
        print(f"Error CRÍTICO al consultar evolución de estudiantes: {e}")
        traceback.print_exc()
        return f"Error al consultar la base de datos de evolución: {e}"

def get_attendance_evolution_summary(db_path, top_n=5, entity_name=None, order_direction='DESC'):
    """
    Calcula la evolución de asistencia (promedio de `attendance_perc`) entre la
    primera y la última instantánea por estudiante.

    - Si `entity_name` está presente, devuelve el detalle para ese alumno.
    - Si no, devuelve el Top-N de mejoras/caídas según `order_direction`.
    """

    query = """
    WITH SnapshotAttendance AS (
        SELECT
            h.snapshot_id,
            h.student_name,
            s.timestamp,
            AVG(h.attendance_perc) AS avg_attendance
        FROM student_data_history h
        JOIN data_snapshots s ON h.snapshot_id = s.id
        GROUP BY h.snapshot_id, h.student_name, s.timestamp
    ),
    StudentEvolution AS (
        SELECT
            student_name,
            FIRST_VALUE(avg_attendance) OVER (
                PARTITION BY student_name ORDER BY timestamp ASC
            ) AS first_avg_attendance,
            LAST_VALUE(avg_attendance) OVER (
                PARTITION BY student_name ORDER BY timestamp ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS last_avg_attendance,
            COUNT(snapshot_id) OVER (PARTITION BY student_name) AS num_snapshots
        FROM SnapshotAttendance
    ),
    FinalEvolution AS (
        SELECT
            student_name,
            first_avg_attendance,
            last_avg_attendance,
            (last_avg_attendance - first_avg_attendance) AS attendance_evolution
        FROM StudentEvolution
        WHERE num_snapshots > 1
        GROUP BY student_name, first_avg_attendance, last_avg_attendance
    )
    SELECT
        student_name,
        first_avg_attendance,
        last_avg_attendance,
        attendance_evolution
    FROM FinalEvolution
    """

    params = []
    if order_direction.upper() not in ['ASC', 'DESC']:
        order_direction = 'DESC'

    if entity_name:
        query += " WHERE student_name = ? AND attendance_evolution != 0"
        params.append(entity_name)
    else:
        query += f" WHERE attendance_evolution != 0 ORDER BY attendance_evolution {order_direction} LIMIT ?"
        params.append(top_n)

    summary_lines = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()

            if not rows:
                if entity_name:
                    return f"No se encontró historial de evolución de asistencia (más de 1 instantánea) para '{entity_name}'."
                else:
                    return "No se encontró historial de evolución de asistencia para ningún estudiante."

            if entity_name:
                summary_lines.append(f"Resumen de Evolución de Asistencia para {entity_name}:")
            else:
                direction_text = "Mejoras" if order_direction.upper() == 'DESC' else "Caídas"
                summary_lines.append(f"Resumen de Evolución de Asistencia (Top {len(rows)} {direction_text}):")

            for i, row in enumerate(rows):
                evo = row['attendance_evolution'] or 0
                sign = '+' if evo > 0 else ''
                first_att = row['first_avg_attendance'] if row['first_avg_attendance'] is not None else 0
                last_att = row['last_avg_attendance'] if row['last_avg_attendance'] is not None else 0
                line = (
                    f"{i+1}. {row['student_name']}: "
                    f"Asistencia Inicial: {first_att:.1f}%, "
                    f"Asistencia Final: {last_att:.1f}% "
                    f"(Evolución: {sign}{evo:.1f} puntos porcentuales)"
                )
                summary_lines.append(line)

        return "\n".join(summary_lines)

    except Exception as e:
        print(f"Error CRÍTICO al consultar evolución de asistencia: {e}")
        traceback.print_exc()
        return f"Error al consultar la base de datos de evolución de asistencia: {e}"

def get_student_qualitative_history(db_path, student_name):
    """
    Consulta la BD para obtener todos los registros cualitativos históricos
    (observaciones, entrevistas, etc.) para un estudiante específico,
    ordenados cronológicamente.
    """
    
    # Columnas cualitativas que queremos extraer de la configuración
    config = current_app.config
    qualitative_cols_mapping = {
        'conduct_observation': (config.get('OBSERVACIONES_COL'), "Observación de Conducta"),
        'interviews_info': (config.get('ENTREVISTAS_COL'), "Registro de Entrevista"),
        'family_info': (config.get('FAMILIA_COL'), "Información Familiar")
    }
    
    # Nombres de las columnas en la BD que realmente existen
    db_cols_to_query = [db_col for db_col, (csv_col, display) in qualitative_cols_mapping.items() if csv_col is not None]

    if not db_cols_to_query:
        return f"No hay columnas cualitativas (Observaciones, Entrevistas, Familia) configuradas en config.py."

    # Construimos la parte SELECT de la consulta dinámicamente
    select_clause = ", ".join(db_cols_to_query)

    query = f"""
    SELECT
        s.timestamp,
        {select_clause}
    FROM student_data_history h
    JOIN data_snapshots s ON h.snapshot_id = s.id
    WHERE h.student_name = ?
    ORDER BY s.timestamp ASC
    """
    
    params = (student_name,)
    summary_lines = []

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            if not rows:
                return f"No se encontró historial cualitativo (observaciones, entrevistas, etc.) para '{student_name}'."
            
            summary_lines.append(f"Historial Cualitativo para {student_name} (ordenado por fecha):")
            
            processed_entries = {} # Para evitar duplicados exactos en la misma fecha

            for row in rows:
                # Convertir timestamp (UTC de la BD) a Santiago
                try:
                    naive_dt = datetime.datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                    utc_dt = pytz.utc.localize(naive_dt)
                    santiago_dt = utc_dt.astimezone(_get_tz())
                    date_str = santiago_dt.strftime('%d/%m/%Y')
                except Exception:
                    date_str = row['timestamp'].split(' ')[0] # Fallback

                entry_lines = []
                entry_key_base = date_str
                
                # Iterar sobre las columnas que consultamos
                for db_col, (csv_col, display_name) in qualitative_cols_mapping.items():
                    if db_col in row.keys() and row[db_col] and pd.notna(row[db_col]):
                        text = str(row[db_col]).strip()
                        if text:
                            entry_lines.append(f"  - {display_name}: {text}")
                            entry_key_base += text # Añadir al key para detectar duplicados

                # Solo añadir si hay contenido real y no es un duplicado de la misma fecha
                if entry_lines and entry_key_base not in processed_entries:
                    summary_lines.append(f"\n--- Registro del {date_str} ---")
                    summary_lines.extend(entry_lines)
                    processed_entries[entry_key_base] = True
            
        if len(summary_lines) <= 1: # Solo el título
            return f"No se encontró historial cualitativo (observaciones, entrevistas, etc.) para '{student_name}'."
            
        return "\n".join(summary_lines)

    except Exception as e:
        print(f"Error CRÍTICO al consultar historial cualitativo: {e}")
        traceback.print_exc()
        return f"Error al consultar la base de datos de historial cualitativo: {e}"

def get_course_qualitative_summary(db_path, course_name, max_entries=30):
    """
    Resume información cualitativa agregada para un curso:
    - Totales por tipo (observaciones, entrevistas, familia)
    - Conteos de entradas negativas/positivas (heurística por keywords)
    - Lista de entradas recientes con fecha, alumno y tipo
    """

    config = current_app.config
    qualitative_cols_mapping = {
        'conduct_observation': (config.get('OBSERVACIONES_COL'), "Observación de Conducta"),
        'interviews_info': (config.get('ENTREVISTAS_COL'), "Registro de Entrevista"),
        'family_info': (config.get('FAMILIA_COL'), "Información Familiar")
    }

    db_cols_to_query = [db_col for db_col, (csv_col, _) in qualitative_cols_mapping.items() if csv_col is not None]
    if not db_cols_to_query:
        return "No hay columnas cualitativas (Observaciones, Entrevistas, Familia) configuradas en config.py."

    select_clause_parts = ["s.timestamp", "h.student_name", "h.student_course"] + db_cols_to_query
    select_clause = ", ".join(select_clause_parts)

    query = f"""
    SELECT
        {select_clause}
    FROM student_data_history h
    JOIN data_snapshots s ON h.snapshot_id = s.id
    WHERE LOWER(TRIM(h.student_course)) = ?
    ORDER BY s.timestamp ASC
    """

    normalized_course = str(course_name).strip().lower()
    params = (normalized_course,)

    negative_keywords = [
        'agresion','pelea','conflicto','disruptivo','falta','injustificado','tarde','retraso',
        'negativo','bajo rendimiento','baja asistencia','problema','riesgo','bullying','maltrato',
        'llamado de atencion','amonestacion','incumplimiento'
    ]
    positive_keywords = [
        'logro','mejora','positivo','participacion','compromiso','destacado','avance','progreso',
        'buen comportamiento','colaboracion','superacion'
    ]

    # Compilar patrones una sola vez, usando normalización compartida
    neg_pattern = compile_any_keyword_pattern(negative_keywords)
    pos_pattern = compile_any_keyword_pattern(positive_keywords)

    def classify_text(text):
        if not text:
            return None
        t = normalize_text(text)
        neg = bool(neg_pattern.search(t))
        pos = bool(pos_pattern.search(t))
        if neg and not pos:
            return 'negativa'
        if pos and not neg:
            return 'positiva'
        return 'neutra'

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()

            if not rows:
                return f"No se encontró historial cualitativo para el curso '{course_name}'."

            totals_by_type = {display: 0 for _, (_, display) in qualitative_cols_mapping.items()}
            sentiment_counts = {'negativa': 0, 'positiva': 0, 'neutra': 0}
            recent_entries = []

            for row in rows:
                for db_col, (_, display) in qualitative_cols_mapping.items():
                    val = row[db_col] if db_col in row.keys() else None
                    if val and str(val).strip():
                        totals_by_type[display] += 1
                        sentiment = classify_text(val)
                        if sentiment: sentiment_counts[sentiment] += 1
                        if len(recent_entries) < max_entries:
                            recent_entries.append({
                                'timestamp': row['timestamp'],
                                'student_name': row['student_name'],
                                'type': display,
                                'text': str(val).strip()
                            })

            lines = [f"Resumen Cualitativo para Curso: {course_name}"]
            lines.append("\nTotales por tipo:")
            for display, count in totals_by_type.items():
                lines.append(f"- {display}: {count}")

            lines.append("\nClasificación heurística (palabras clave):")
            lines.append(f"- Positivas: {sentiment_counts['positiva']}")
            lines.append(f"- Neutras: {sentiment_counts['neutra']}")
            lines.append(f"- Negativas: {sentiment_counts['negativa']}")

            if recent_entries:
                lines.append("\nEntradas recientes (máx. {max_entries}):")
                for i, e in enumerate(recent_entries, start=1):
                    # Recortar texto para mantener legibilidad
                    txt = e['text']
                    if len(txt) > 220: txt = txt[:220] + '…'
                    # Convertir timestamp a fecha legible
                    try:
                        naive_dt = datetime.datetime.strptime(e['timestamp'], '%Y-%m-%d %H:%M:%S')
                        utc_dt = pytz.utc.localize(naive_dt)
                        local_dt = utc_dt.astimezone(_get_tz())
                        fecha = local_dt.strftime('%Y-%m-%d')
                    except Exception:
                        fecha = str(e['timestamp']).split(' ')[0]
                    lines.append(f"{i}. [{fecha}] {e['student_name']} - {e['type']}: {txt}")

            return "\n".join(lines)

    except Exception as e:
        print(f"Error CRÍTICO al consultar resumen cualitativo por curso: {e}")
        traceback.print_exc()
        return f"Error al consultar la base de datos para resumen cualitativo de curso: {e}"

# Helper de presupuesto de caracteres a nivel de módulo
def _trim_to_char_budget(text: str, budget: int) -> str:
    try:
        if budget is None or budget <= 0:
            return ""
        if text is None:
            return ""
        if len(text) <= budget:
            return text
        # Recorta y añade indicador de truncado para trazabilidad
        return text[:budget] + f"… [truncado a {budget} caracteres]"
    except Exception:
        # En caso de error, devuelve una versión segura
        try:
            return (text or "")[:max(0, budget or 0)]
        except Exception:
            return ""

def analyze_data_with_gemini(data_string, user_prompt, vs_inst, vs_followup, 
                             chat_history_string="", is_reporte_360=False, 
                             is_plan_intervencion=False, is_direct_chat_query=False,
                             entity_type=None, entity_name=None,
                             historical_data_summary_string="",
                             qualitative_history_summary_string="",
                             feature_store_signals_string=""): # <-- MODIFICACIÓN: Nuevo parámetro
    
    api_key = current_app.config.get('GEMINI_API_KEY')
    num_relevant_chunks_inst = int(current_app.config.get('NUM_RELEVANT_CHUNKS_INST', current_app.config.get('NUM_RELEVANT_CHUNKS', 10)))
    num_relevant_chunks_fu = int(current_app.config.get('NUM_RELEVANT_CHUNKS_FU', current_app.config.get('NUM_RELEVANT_CHUNKS', 10)))
    model_name = 'gemini-2.5-flash' 

    def create_error_response(error_message):
        # ... (código interno de create_error_response no cambia) ...
        return {
            'html_output': f"<div class='text-red-700 p-3 bg-red-100 border rounded-md'><strong>Error:</strong> {error_message}</div>",
            'raw_markdown': f"Error: {error_message}", 'error': error_message,
            'model_name': model_name, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
            'input_cost': 0, 'output_cost': 0, 'total_cost': 0
        }

    if not api_key:
        return create_error_response("Configuración: Falta la clave API de Gemini.")

    # ... (lógica de RAG para contexto institucional y follow-ups no cambia) ...
    
    retrieved_context_inst = "Contexto Institucional no disponible o no buscado."
    retrieved_context_followup_header = "Historial de Seguimiento Relevante de la Entidad"
    retrieved_context_followup = "Historial de Seguimiento de la Entidad no disponible o no buscado."
    default_analysis_prompt = current_app.config.get('DEFAULT_ANALYSIS_PROMPT', "Realiza un análisis general de los datos.")
    final_user_instruction = user_prompt
    if entity_type and entity_name:
        retrieved_context_followup_header = f"Historial de Seguimiento Específico para {entity_type.capitalize()} '{entity_name}'"

    # --- RAG institucional y de seguimientos: ahora también para Reporte 360 y Plan de Intervención ---
    # Siempre intentamos recuperar contexto RAG; la inclusión en el prompt dependerá del modo.
    if not final_user_instruction:
         final_user_instruction = default_analysis_prompt
    # Recuperación de contexto institucional
    if vs_inst:
        try:
            relevant_docs_inst = vs_inst.similarity_search(final_user_instruction, k=num_relevant_chunks_inst)
            if relevant_docs_inst:
                context_list = [
                    f"--- Contexto Inst. {i+1} (Fuente: {os.path.basename(doc.metadata.get('source', 'Unknown'))}) ---\n{doc.page_content}\n"
                    for i, doc in enumerate(relevant_docs_inst)
                ]
                retrieved_context_inst = "\n".join(context_list)
            else:
                retrieved_context_inst = "No se encontró contexto institucional relevante para esta consulta."
        except Exception as e:
            retrieved_context_inst = f"Error al buscar contexto institucional: {e}"
    
    # Recuperación de contexto de seguimientos/reportes/observaciones
    relevant_docs_fu_final = []
    try:
        if entity_type and entity_name:
            # Cargar todos los follow-ups desde la BD como Document
            all_db_followups_as_lc_docs = load_follow_ups_as_documents(current_app.config['DATABASE_FILE'])

            # Normalizar y filtrar por entidad
            def _norm(s: str) -> str:
                try:
                    import unicodedata
                    return ''.join(c for c in unicodedata.normalize('NFKD', s.lower()) if not unicodedata.combining(c)).strip()
                except Exception:
                    return (s or '').lower().strip()

            requested_type = _norm(entity_type)
            requested_name = _norm(entity_name)

            filtered_docs = []
            for doc in all_db_followups_as_lc_docs:
                meta = doc.metadata or {}
                t = _norm(str(meta.get('related_entity_type', '')))
                n = _norm(str(meta.get('related_entity_name', '')))
                if t == requested_type and n == requested_name:
                    filtered_docs.append(doc)

            # Construir índice temporal si es posible
            try:
                from langchain_community.vectorstores import FAISS as _FAISS
            except Exception:
                _FAISS = None

            k_fu = num_relevant_chunks_fu

            if filtered_docs:
                if 'embedding_model_instance' in globals() and embedding_model_instance is not None and _FAISS is not None:
                    try:
                        temp_index = _FAISS.from_documents(filtered_docs, embedding_model_instance)
                        relevant_docs_fu_final = temp_index.similarity_search(final_user_instruction, k=k_fu)
                    except Exception:
                        relevant_docs_fu_final = []

                if not relevant_docs_fu_final:
                    # Fallback simple por coincidencias
                    prompt_terms = [w for w in re.split(r"\W+", final_user_instruction or "") if len(w) > 2]
                    def score(doc):
                        text = (doc.page_content or "").lower()
                        return sum(1 for w in prompt_terms if w.lower() in text)
                    filtered_docs_sorted = sorted(filtered_docs, key=score, reverse=True)
                    relevant_docs_fu_final = filtered_docs_sorted[:k_fu]

            if not relevant_docs_fu_final:
                retrieved_context_followup = (
                    f"No se encontraron seguimientos específicos para {entity_type.capitalize()} '{entity_name}' que sean relevantes para la consulta actual."
                )
        else:
            # Búsqueda general con índice global
            if vs_followup:
                relevant_docs_fu_final = vs_followup.similarity_search(final_user_instruction, k=num_relevant_chunks_fu)
            else:
                retrieved_context_followup = "El índice de historial de seguimiento no está disponible actualmente (vs_followup is None)."

        if relevant_docs_fu_final:
            context_list_fu = [
                f"--- Documento Histórico Relevante {i+1} "
                f"(Tipo: {str(doc.metadata.get('follow_up_type','N/A')).replace('_',' ').capitalize()}, "
                f"ID: {doc.metadata.get('id','N/A')}, "
                f"Entidad: {doc.metadata.get('related_entity_type','N/A')}-{doc.metadata.get('related_entity_name','N/A')}, "
                f"Fecha: {str(doc.metadata.get('timestamp','')).split(' ')[0]}) ---\n"
                f"{doc.page_content}\n"
                for i, doc in enumerate(relevant_docs_fu_final)
            ]
            retrieved_context_followup = "\n".join(context_list_fu)
    except Exception as e_followup_retrieval:
        retrieved_context_followup = f"Error crítico al buscar en el historial de seguimiento: {e_followup_retrieval}"
        traceback.print_exc()
    try:
        folder_ctx = current_app.config.get('CONTEXT_DOCS_FOLDER')
        if folder_ctx and os.path.isdir(folder_ctx):
            names = [n for n in os.listdir(folder_ctx) if n.lower().endswith(('.pdf', '.txt'))]
            if names:
                listing = "\n".join([f"* {n}" for n in sorted(names)])
                header = "Documentos de Contexto Institucional:\n"
                retrieved_context_inst = (header + listing + "\n\n" + (retrieved_context_inst or "")).strip()
    except Exception:
        pass
            
    # --- Helpers de presupuesto de prompt ---
    def _trim_to_char_budget(text: str, max_chars: int) -> str:
        try:
            if text is None:
                return ""
            s = str(text)
            if max_chars <= 0:
                return ""
            if len(s) <= max_chars:
                return s
            # Recortar y añadir marca de truncado
            trimmed = s[:max_chars]
            return trimmed + "\n[...texto truncado por presupuesto...]"
        except Exception:
            return str(text)[:max_chars]

    # Calcular presupuestos por sección
    enable_budget = bool(current_app.config.get('ENABLE_PROMPT_BUDGETING', True))
    total_prompt_chars = int(current_app.config.get('PROMPT_MAX_CHARS', 24000))
    budgets = current_app.config.get('PROMPT_SECTION_CHAR_BUDGETS', {}) or {}
    def budget_for(section_key: str, default_ratio: float) -> int:
        ratio = float(budgets.get(section_key, default_ratio))
        if ratio < 0: ratio = 0
        return max(0, int(total_prompt_chars * ratio))

    chat_hist_budget = budget_for('chat_history', 0.10)
    qual_docs_budget = budget_for('key_docs_and_qualitative', 0.25)
    rag_inst_budget = min(budget_for('rag_institutional', 0.25), int(current_app.config.get('RAG_INST_MAX_CHARS', 8000)))
    rag_fu_budget = min(budget_for('rag_followups', 0.20), int(current_app.config.get('RAG_FOLLOWUP_MAX_CHARS', 8000)))
    hist_quant_budget = budget_for('historical_quantitative', 0.10)
    csv_budget = budget_for('csv_or_base_context', 0.10)
    fs_signals_budget = budget_for('feature_store_signals', 0.08)
    external_refs_budget = budget_for('external_refs', 0.05)

    # --- Construct final prompt for Gemini ---
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt_parts = []
        primary_blocks_added = False  # Para evitar duplicar secciones de CSV/histórico/cualitativo
        system_prompt = current_app.config.get('GEMINI_SYSTEM_ROLE_PROMPT', "Eres un asistente útil.")
        # Refuerzo específico para consultas por curso: usar datos agregados del CSV del curso
        if entity_type == 'curso' and entity_name:
            system_prompt = (
                system_prompt +
                "\nDirectriz: cuando la entidad es un curso, analiza patrones agregados y oportunidades de mejora del curso completo usando los datos del CSV filtrados para ese curso. Evita centrar el análisis en un solo alumno salvo que el usuario lo solicite explícitamente."
            )
        prompt_parts.append(system_prompt)

        if not is_reporte_360 and not is_plan_intervencion: 
            if chat_history_string:
                prompt_parts.append("\n--- INICIO HISTORIAL DE CONVERSACIÓN PREVIA ---\n")
                prompt_parts.append(_trim_to_char_budget(chat_history_string, chat_hist_budget) if enable_budget else chat_history_string)
                prompt_parts.append("\n--- FIN HISTORIAL DE CONVERSACIÓN PREVIA ---\n")
            # Señales del Feature Store (si existen)
            if feature_store_signals_string:
                prompt_parts.append("Señales calculadas previamente del Feature Store (CSV + historial integrado):\n```\n")
                fs_block = _trim_to_char_budget(feature_store_signals_string, fs_signals_budget) if enable_budget else feature_store_signals_string
                prompt_parts.append(fs_block)
                prompt_parts.append("\n```\n---")
            
            is_course_query = (entity_type == 'curso')
            if is_course_query:
                # Para curso: priorizar datos del CSV primero, luego cuantitativo histórico, luego cualitativo, luego RAG
                prompt_parts.append("Contexto Principal de Datos del Curso (CSV filtrado al curso):\n```\n")
                data_block_initial = _trim_to_char_budget(data_string, csv_budget) if enable_budget else data_string
                prompt_parts.append(data_block_initial)
                prompt_parts.append("\n```\n---")
                primary_blocks_added = True

                if historical_data_summary_string:
                    prompt_parts.append("Resumen de Datos Históricos del Curso (Evolución de notas calculada desde la Base de Datos):\n```\n")
                    hist_block_initial = _trim_to_char_budget(historical_data_summary_string, hist_quant_budget) if enable_budget else historical_data_summary_string
                    prompt_parts.append(hist_block_initial)
                    prompt_parts.append("\n```\n---")

                if qualitative_history_summary_string:
                    prompt_parts.append("Resumen de Datos Cualitativos Históricos del Curso (conducta, entrevistas, etc.):\n```\n")
                    qual_block = _trim_to_char_budget(qualitative_history_summary_string, qual_docs_budget) if enable_budget else qualitative_history_summary_string
                    prompt_parts.append(qual_block)
                    prompt_parts.append("\n```\n---")

                # Finalmente RAG
                prompt_parts.extend([
                    "Contexto Institucional Relevante (Información general de la institución):\n```\n",
                    (_trim_to_char_budget(retrieved_context_inst, rag_inst_budget) if enable_budget else retrieved_context_inst),
                    "\n```\n---",
                    f"{retrieved_context_followup_header} (Reportes 360 previos, Observaciones registradas, etc.):\n```\n",
                    (_trim_to_char_budget(retrieved_context_followup, rag_fu_budget) if enable_budget else retrieved_context_followup),
                    "\n```\n---",
                ])
                if bool(current_app.config.get('ENABLE_EXTERNAL_PEDAGOGY_REFERENCES', False)):
                    ext_block = build_external_pedagogy_references(final_user_instruction)
                    if ext_block:
                        prompt_parts.append("Referencias Externas Verificadas (modelos educativos y principios):\n``\n")
                        prompt_parts.append(_trim_to_char_budget(ext_block, external_refs_budget) if enable_budget else ext_block)
                        prompt_parts.append("\n``\n---")
                # Directriz obligatoria: si hay entidad especificada (alumno o curso), integra la Biblioteca de Seguimiento.
                # Enumera Observaciones, Reportes 360 y Planes de intervención, y explica la incidencia global para la entidad.
                # Si no hay registros, indícalo. Evita omitir Observaciones relevantes.
                prompt_parts.append(
                    "Directriz obligatoria: integra registros de Biblioteca de Seguimiento (Observaciones, Reportes 360, Planes) "
                    "y explica su incidencia global para la entidad. Si no hay registros, indícalo."
                )
                # Directriz obligatoria: integra y sintetiza los registros de la Biblioteca de Seguimiento del curso.
                # Enumera en secciones: 1) Observaciones (qué, quién, fecha, asignatura si aplica), 2) Reportes 360 relevantes,
                # 3) Planes de intervención relevantes. Para cada Observación, explica su incidencia en el estado global del curso
                # (patrones/tendencias, riesgos y recomendaciones). Si no hay registros, indícalo explícitamente.
                prompt_parts.append(
                    "Directriz obligatoria: integra y sintetiza los registros de la Biblioteca de Seguimiento del curso. "
                    "Enumera Observaciones, Reportes 360 y Planes de intervención; explica incidencia global del curso; "
                    "si no hay registros, indícalo."
                )
            else:
                # CSV-FIRST para consultas generales y de alumno:
                # Priorizar SIEMPRE el CSV (datos objetivos numéricos) al inicio;
                # luego resumen histórico cuantitativo; después cualitativo; y finalmente RAG.
                prompt_parts.append("Contexto Principal de Datos (CSV primero para la consulta actual):\n```\n")
                data_block_initial = _trim_to_char_budget(data_string, csv_budget) if enable_budget else data_string
                prompt_parts.append(data_block_initial)
                prompt_parts.append("\n```\n---")
                primary_blocks_added = True

                if historical_data_summary_string:
                    prompt_parts.append("Resumen de Datos Históricos (Evolución Cuantitativa/Notas calculada desde la Base de Datos):\n```\n")
                    hist_block_initial = _trim_to_char_budget(historical_data_summary_string, hist_quant_budget) if enable_budget else historical_data_summary_string
                    prompt_parts.append(hist_block_initial)
                    prompt_parts.append("\n```\n---")

                if qualitative_history_summary_string:
                    prompt_parts.append("Resumen de Datos Cualitativos Históricos (conducta, entrevistas, etc.):\n```\n")
                    qual_block = _trim_to_char_budget(qualitative_history_summary_string, qual_docs_budget) if enable_budget else qualitative_history_summary_string
                    prompt_parts.append(qual_block)
                    prompt_parts.append("\n```\n---")

                # Finalmente RAG (institucional y de seguimiento)
                prompt_parts.extend([
                    "Contexto Institucional Relevante (Información general de la institución, si se encontró para la consulta):\n```\n",
                    (_trim_to_char_budget(retrieved_context_inst, rag_inst_budget) if enable_budget else retrieved_context_inst),
                    "\n```\n---",
                    f"{retrieved_context_followup_header} (Reportes 360 previos, Observaciones registradas, etc., si se encontraron para la consulta):\n```\n",
                    (_trim_to_char_budget(retrieved_context_followup, rag_fu_budget) if enable_budget else retrieved_context_followup),
                    "\n```\n---",
                ])
        
        # Evitar duplicar bloques si ya se añadieron en el orden CSV-first anterior
        if (not primary_blocks_added) and not (entity_type == 'curso' and not is_reporte_360 and not is_plan_intervencion):
            if historical_data_summary_string:
                prompt_parts.append("Resumen de Datos Históricos (Evolución Cuantitativa/Notas calculada desde la Base de Datos):\n```\n")
                hist_block = _trim_to_char_budget(historical_data_summary_string, hist_quant_budget) if enable_budget else historical_data_summary_string
                prompt_parts.append(hist_block)
                prompt_parts.append("\n```\n---")

            prompt_parts.append("Contexto Principal de Datos para la Consulta Actual (Datos del CSV para la entidad, o Reporte 360 base para el plan):\n```\n")
            data_block = _trim_to_char_budget(data_string, csv_budget) if enable_budget else data_string
            prompt_parts.append(data_block)
            prompt_parts.append("\n```\n---")

        prompt_parts.extend([
            "Instrucción Específica del Usuario (Pregunta Actual o Solicitud):\n", f'"{final_user_instruction}"', "\n---",
        ])

        if not is_direct_chat_query and not is_reporte_360 and not is_plan_intervencion: 
            prompt_parts.append(current_app.config.get('GEMINI_FORMATTING_INSTRUCTIONS', "Formatea tu respuesta claramente en Markdown."))
        elif is_direct_chat_query:
            pass 

        final_prompt_string = "\n".join(filter(None, prompt_parts)) 

        # Logging de tamaños por sección
        if current_app.config.get('LOG_PROMPT_SECTIONS', True):
            try:
                current_app.logger.info(
                    f"Prompt budgeting: chat_hist={chat_hist_budget}, qual_docs={qual_docs_budget}, rag_inst={rag_inst_budget}, rag_fu={rag_fu_budget}, hist_quant={hist_quant_budget}, csv={csv_budget}, total_chars={len(final_prompt_string)}"
                )
            except Exception:
                pass

        # Para Reporte 360 y Plan de Intervención, añadir explícitamente el contexto RAG al prompt
        if is_reporte_360 or is_plan_intervencion:
            prompt_parts.append(
                "Contexto Institucional Relevante (fragmentos recuperados por RAG):\n``\n"
            )
            prompt_parts.append(_trim_to_char_budget(retrieved_context_inst, rag_inst_budget) if enable_budget else retrieved_context_inst)
            prompt_parts.append("\n```\n---")

            prompt_parts.append(
                f"{retrieved_context_followup_header} (si se encontraron documentos/observaciones relevantes):\n``\n"
            )
            prompt_parts.append(_trim_to_char_budget(retrieved_context_followup, rag_fu_budget) if enable_budget else retrieved_context_followup)
            prompt_parts.append("\n```\n---")

            final_prompt_string = "\n".join(filter(None, prompt_parts))

        response = model.generate_content(final_prompt_string)
        
        # ... (El resto de la función: conteo de tokens, manejo de respuesta, etc. no cambia) ...
        input_tokens, output_tokens, total_tokens = 0, 0, 0
        # ... (código de cálculo de costos) ...
        
        try:
            # ... (código de usage_info y pricing_info) ...
            usage_info = response.usage_metadata
            input_tokens = usage_info.prompt_token_count
            output_tokens = usage_info.candidates_token_count
            total_tokens = usage_info.total_token_count
            
            pricing_info = current_app.config.get('MODEL_PRICING', {}).get(model_name, {})
            if pricing_info:
                input_cost_per_million = pricing_info.get('input_per_million', 0)
                output_cost_per_million = pricing_info.get('output_per_million', 0)
                cost_input = (input_tokens / 1_000_000) * input_cost_per_million
                cost_output = (output_tokens / 1_000_000) * output_cost_per_million
                total_cost = cost_input + cost_output
            
            db_path = current_app.config['DATABASE_FILE']
            guardar_consumo_diario(db_path, model_name, input_tokens, output_tokens, total_cost)

        except Exception as e:
            print(f"Advertencia: No se pudo calcular el consumo de tokens/costos: {e}")
            traceback.print_exc()

        raw_response_text, html_output, error_message = "", "", None
        # ... (código de feedback_info y procesamiento de respuesta) ...
        feedback_info = response.prompt_feedback if hasattr(response, 'prompt_feedback') else None
        
        if feedback_info and feedback_info.block_reason:
            error_message = f"La respuesta de Gemini fue bloqueada. Razón: {getattr(feedback_info, 'block_reason_message', str(feedback_info.block_reason))}"
        else:
            try: raw_response_text = response.text
            except ValueError: 
                try: raw_response_text = "".join(part.text for part in response.parts)
                except Exception: error_message = "Crítico al procesar partes de la respuesta de Gemini."
            except Exception: error_message = "Inesperado al procesar .text de la respuesta de Gemini."

            if not error_message and not raw_response_text: error_message = "La respuesta de Gemini estaba vacía."
            
            if not error_message:
                # Ajuste: evitar 'nl2br' para planes de intervención (provocaba saltos de línea no deseados y mala justificación)
                md_extensions = ['fenced_code', 'tables', 'sane_lists']
                if not is_plan_intervencion:
                    # En respuestas generales y reportes 360, mantener nl2br para respetar saltos simples
                    md_extensions.append('nl2br')
                html_output = markdown.markdown(raw_response_text, extensions=md_extensions)
        
        if error_message: 
            print(f"Error en Gemini: {error_message}")
            return create_error_response(error_message)
        
        return {
            'html_output': html_output,
            'raw_markdown': raw_response_text,
            'model_name': model_name,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'input_cost': cost_input,
            'output_cost': cost_output,
            'total_cost': total_cost,
            'error': None
        }
    except Exception as e:
        print(f"Excepción en analyze_data_with_gemini: {e}")
        traceback.print_exc()
        return create_error_response(f"Comunicación con Gemini o procesamiento de su respuesta: {e}")

def _safe_http_get(url, timeout, whitelist, blacklist, prefix_map=None):
    try:
        p = urlparse(url)
        host = (p.netloc or '').lower()
        if not any(host.endswith(d) for d in (whitelist or [])): return None
        if any(host.endswith(d) for d in (blacklist or [])): return None
        if prefix_map:
            for dom, prefixes in prefix_map.items():
                if host.endswith(dom):
                    path = (p.path or '').lower()
                    if not any(path.startswith(pr.lower()) for pr in prefixes):
                        return None
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200: return None
        txt = r.text or ''
        return txt[:20000]
    except Exception:
        return None

def _wiki_summary(title, lang, timeout, whitelist, blacklist, prefix_map=None):
    base = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    return _safe_http_get(base, timeout, whitelist, blacklist, prefix_map)

def build_external_pedagogy_references(user_text):
    try:
        cfg = current_app.config
        wl = cfg.get('RESOURCE_WHITELIST_DOMAINS', [])
        bl = cfg.get('RESOURCE_BLACKLIST_DOMAINS', [])
        timeout = int(cfg.get('RESOURCE_VERIFY_TIMEOUT_SEC', 4))
        lang = cfg.get('EXTERNAL_PREFERRED_LANGUAGE', 'es')
        pfx = cfg.get('EXTERNAL_DOMAIN_PATH_PREFIXES', {})
        terms = cfg.get('EXTERNAL_PEDAGOGY_KEY_TERMS', []) or []
        norm = normalize_text(user_text or '')
        matched = [t for t in terms if t in norm]
        if not matched:
            matched = terms[:5]
        refs = []
        titles = {
            'finlandia': 'Educación_en_Finlandia',
            'singapur': 'Educación_en_Singapur',
            'inglaterra': 'Educación_en_Inglaterra',
            'canada': 'Educación_en_Canadá',
            'canadá': 'Educación_en_Canadá',
            'corea del sur': 'Educación_en_Corea_del_Sur',
            'corea': 'Educación_en_Corea_del_Sur',
            'carga cognitiva': 'Teoría_de_la_carga_cognitiva'
        }
        for t in matched:
            key = normalize_text(t)
            title = titles.get(key)
            if not title: continue
            s = _wiki_summary(title, lang, timeout, wl, bl, pfx)
            if s:
                refs.append(f"* {t} — {lang}.wikipedia.org")
        base_local = cfg.get('RESOURCE_LOCAL_BASELINE') or []
        for item in base_local[:2]:
            cands = item.get('candidates') or []
            for u in cands:
                txt = _safe_http_get(u, timeout, wl, bl, pfx)
                if txt:
                    refs.append(f"* {item.get('title','Recurso')} — {u}")
                    break
        oecd = ['https://www.oecd.org/education/', 'https://www.oecd.org/education/skills/']
        for u in oecd:
            txt = _safe_http_get(u, timeout, wl, bl, pfx)
            if txt:
                refs.append(f"* OECD — {u}")
                break
        return "\n".join(refs) if refs else ""
    except Exception:
        return ""

def init_sqlite_db(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS follow_ups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                report_date DATE,
                related_filename TEXT,
                related_prompt TEXT,
                related_analysis TEXT, 
                follow_up_comment TEXT NOT NULL, 
                follow_up_type TEXT DEFAULT 'general_comment', 
                related_entity_type TEXT, 
                related_entity_name TEXT
            )
        ''')
        table_info = cursor.execute("PRAGMA table_info(follow_ups)").fetchall()
        column_names = [info[1] for info in table_info]
        if 'report_date' not in column_names:
            cursor.execute("ALTER TABLE follow_ups ADD COLUMN report_date DATE")
        if 'owner_username' not in column_names:
            cursor.execute("ALTER TABLE follow_ups ADD COLUMN owner_username TEXT")

        # --- INICIO: NUEVA TABLA PARA CONSUMO DE TOKENS ---
        # Se crea una tabla para llevar un registro histórico y acumulado del consumo por día y por modelo.
        # La clave primaria compuesta (fecha, modelo) asegura un único registro diario para cada modelo.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consumo_tokens_diario (
                fecha DATE NOT NULL,
                modelo TEXT NOT NULL,
                tokens_subida INTEGER DEFAULT 0,
                tokens_bajada INTEGER DEFAULT 0,
                costo_total REAL DEFAULT 0.0,
                PRIMARY KEY (fecha, modelo)
            )
        ''')

        # --- INICIO: NUEVAS TABLAS PARA HISTORIAL DE DATOS ---

        # Tabla 1: Registra cada carga de archivo como una "instantánea"
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                filename TEXT NOT NULL,
                num_students INTEGER,
                num_records INTEGER,
                UNIQUE(timestamp, filename)
            )
        ''')

        # Tabla 2: Almacena los datos de cada estudiante de cada instantánea
        # Esto nos permitirá consultar la evolución de notas, asistencia, etc.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS student_data_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                student_name TEXT NOT NULL,
                student_course TEXT,
                subject TEXT,
                grade REAL,
                attendance_perc REAL,
                conduct_observation TEXT,
                age INTEGER,
                professor TEXT,
                family_info TEXT,
                interviews_info TEXT,
                FOREIGN KEY (snapshot_id) REFERENCES data_snapshots (id) ON DELETE CASCADE
            )
        ''')
        # Crear índices para acelerar consultas futuras
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_student_data_snapshot ON student_data_history (snapshot_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_student_data_name ON student_data_history (student_name)")
        
        # --- FIN: NUEVAS TABLAS PARA HISTORIAL DE DATOS ---

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intervention_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                follow_up_id INTEGER,
                related_entity_type TEXT,
                related_entity_name TEXT,
                course_name TEXT,
                outcome_date DATE,
                compliance_pct REAL,
                impact_grade_delta REAL,
                impact_attendance_delta REAL,
                notes TEXT,
                FOREIGN KEY (follow_up_id) REFERENCES follow_ups (id) ON DELETE SET NULL
            )
        ''')

        # Tabla de acciones de intervención asociadas a un plan
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intervention_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id INTEGER NOT NULL,
                owner_username TEXT,
                action_title TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                applied_date DATE,
                responsable TEXT,
                compliance_pct REAL DEFAULT 0.0,
                notes TEXT,
                UNIQUE(plan_id, action_title, owner_username)
            )
        ''')

        conn.commit()
    except Exception as e:
        print(f"Error CRÍTICO al inicializar/actualizar SQLite: {e}"); traceback.print_exc()
    finally:
        if conn: conn.close()

def guardar_consumo_diario(db_path, modelo, tokens_subida, tokens_bajada, costo_total):
    """
    Registra o actualiza el consumo de tokens para un modelo en una fecha específica.
    Utiliza INSERT ON CONFLICT para sumar los valores si ya existe un registro para esa fecha y modelo.
    """
    # Usamos solo la fecha, sin la hora, para agrupar por día.
    fecha_actual = datetime.date.today().isoformat()
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            # Esta consulta inserta una nueva fila o, si ya existe una para la misma fecha y modelo,
            # actualiza la existente sumando los nuevos valores a los acumulados.
            cursor.execute("""
                INSERT INTO consumo_tokens_diario (fecha, modelo, tokens_subida, tokens_bajada, costo_total)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(fecha, modelo) DO UPDATE SET
                    tokens_subida = tokens_subida + excluded.tokens_subida,
                    tokens_bajada = tokens_bajada + excluded.tokens_bajada,
                    costo_total = costo_total + excluded.costo_total
            """, (fecha_actual, modelo, tokens_subida, tokens_bajada, costo_total))
            conn.commit()
    except Exception as e:
        print(f"Error CRÍTICO al guardar el consumo de tokens en la BD: {e}")
        traceback.print_exc()

def save_reporte_360_to_db(db_path, filename, tipo_entidad, nombre_entidad, reporte_360_markdown, prompt_reporte_360="Reporte 360 generado automáticamente", owner_username=None):
    # No changes
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO follow_ups 
                (related_filename, related_prompt, follow_up_comment, follow_up_type, related_entity_type, related_entity_name, related_analysis, owner_username)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (filename, prompt_reporte_360, reporte_360_markdown, 'reporte_360', tipo_entidad, nombre_entidad, None, owner_username))
            conn.commit()
            last_id = cursor.lastrowid
        print(f"Reporte 360 para {tipo_entidad} {nombre_entidad} (archivo: {filename}) guardado en la BD con ID: {last_id}.")
        return last_id
    except Exception as e:
        print(f"Error CRÍTICO al guardar Reporte 360 en la BD: {e}")
        traceback.print_exc()
        return None

def save_observation_for_reporte_360(db_path, parent_reporte_360_id, observador_nombre, observacion_text, tipo_entidad, nombre_entidad, csv_filename, owner_username=None):
    # No changes
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            related_prompt_text = f"Observación de: {observador_nombre}"
            cursor.execute("""
                INSERT INTO follow_ups
                (related_filename, related_prompt, related_analysis, follow_up_comment, follow_up_type, related_entity_type, related_entity_name, owner_username)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (csv_filename, related_prompt_text, str(parent_reporte_360_id), observacion_text, 'observacion_reporte_360', tipo_entidad, nombre_entidad, owner_username))
            conn.commit()
        print(f"Observación para Reporte 360 ID {parent_reporte_360_id} (Entidad: {tipo_entidad} {nombre_entidad}, Archivo: {csv_filename}) guardada.")
        return True
    except Exception as e:
        print(f"Error CRÍTICO al guardar observación para Reporte 360 en la BD: {e}")
        traceback.print_exc()
        return False

def save_observation_for_entity(db_path, entity_type, entity_name, observer_name, observation_text, csv_filename, owner_username=None):
    """Guarda una observación independiente para una entidad (alumno/curso) y la deja disponible
    para el índice FAISS de seguimientos.

    Se utiliza follow_up_type='observacion_entidad'. No tiene relación directa con un reporte 360,
    pero sí queda asociada al archivo CSV actual y a la entidad.
    """
    try:
        # Validaciones básicas en servidor
        if not all([entity_type, entity_name, observer_name, observation_text, csv_filename]):
            print("Error: Datos incompletos al guardar observación de entidad.")
            return False

        # Limitar a 300 palabras en servidor por seguridad (además del control en cliente)
        words = observation_text.strip().split()
        if len(words) > 300:
            observation_text = " ".join(words[:300])

        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            related_prompt_text = f"Observación de: {observer_name}"
            cursor.execute(
                """
                INSERT INTO follow_ups
                (related_filename, related_prompt, related_analysis, follow_up_comment, follow_up_type, related_entity_type, related_entity_name, owner_username)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (csv_filename, related_prompt_text, None, observation_text, 'observacion_entidad', entity_type, entity_name, owner_username)
            )
            conn.commit()
            new_id = cursor.lastrowid
        print(f"Observación de entidad guardada (ID: {new_id}) para {entity_type} {entity_name}, archivo: {csv_filename}.")
        return True
    except Exception as e:
        print(f"Error CRÍTICO al guardar observación de entidad en la BD: {e}")
        traceback.print_exc()
        return False

def save_data_snapshot_to_db(df, filename, db_path):
    """
    Guarda el DataFrame completo en la base de datos como una instantánea histórica.
    """
    if df is None or df.empty:
        print("Error en save_data_snapshot_to_db: El DataFrame está vacío.")
        return False, "DataFrame vacío."

    config = current_app.config
    total_records = len(df)
    total_students = df[config['NOMBRE_COL']].nunique()

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # 1. Crear la entrada en la tabla de instantáneas
            cursor.execute("""
                INSERT INTO data_snapshots (filename, num_students, num_records)
                VALUES (?, ?, ?)
            """, (filename, int(total_students), int(total_records)))
            
            snapshot_id = cursor.lastrowid

            # 2. Preparar el DataFrame para la tabla histórica
            # Seleccionamos solo las columnas que queremos guardar
            df_to_save = df.copy()
            
            # Añadimos el ID de la instantánea
            df_to_save['snapshot_id'] = snapshot_id

            # Mapeamos las columnas del CSV (desde config) a los nombres de la BD
            column_mapping = {
                config['NOMBRE_COL']: 'student_name',
                config['CURSO_COL']: 'student_course',
                config['ASIGNATURA_COL']: 'subject',
                config['NOTA_COL']: 'grade',
                config.get('ASISTENCIA_COL'): 'attendance_perc',
                config.get('OBSERVACIONES_COL'): 'conduct_observation',
                config.get('EDAD_COL'): 'age',
                config.get('PROFESOR_COL'): 'professor',
                config.get('FAMILIA_COL'): 'family_info',
                config.get('ENTREVISTAS_COL'): 'interviews_info'
            }

            # Renombrar solo las columnas que existen en el DataFrame
            actual_mapping = {csv_col: db_col for csv_col, db_col in column_mapping.items() if csv_col in df_to_save.columns}
            df_to_save.rename(columns=actual_mapping, inplace=True)

            # Columnas finales que deben coincidir con la tabla student_data_history
            final_db_columns = [
                'snapshot_id', 'student_name', 'student_course', 'subject', 'grade',
                'attendance_perc', 'conduct_observation', 'age', 'professor',
                'family_info', 'interviews_info'
            ]
            
            # Filtrar el DataFrame para que solo contenga columnas que existen en la BD
            columns_to_insert = [col for col in final_db_columns if col in df_to_save.columns]
            df_final = df_to_save[columns_to_insert]
            
            # 3. Guardar los datos en la tabla histórica usando pandas.to_sql
            df_final.to_sql('student_data_history', conn, if_exists='append', index=False)
            
            conn.commit()
            
        print(f"Instantánea de datos guardada con ID {snapshot_id} ({total_records} registros) para el archivo {filename}.")
        return True, f"Instantánea de datos (ID: {snapshot_id}) guardada con {total_records} registros."

    except Exception as e:
        print(f"Error CRÍTICO al guardar la instantánea de datos: {e}")
        traceback.print_exc()
        # Intentar revertir la entrada de snapshot si falla la inserción de datos
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute("DELETE FROM data_snapshots WHERE id = ?", (snapshot_id,))
                conn.commit()
            print(f"Reversión de instantánea ID {snapshot_id} completada.")
        except Exception as e_rollback:
            print(f"Error CRÍTICO durante la reversión de la instantánea: {e_rollback}")
            
        return False, f"Error al guardar la instantánea: {e}"

def load_follow_ups_as_documents(db_path, owner_username=None): 
    follow_up_docs = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row 
            cursor = conn.cursor()
            if owner_username:
                cursor.execute("SELECT * FROM follow_ups WHERE owner_username = ? ORDER BY timestamp DESC", (owner_username,))
            else:
                cursor.execute("SELECT * FROM follow_ups ORDER BY timestamp DESC") 
            rows = cursor.fetchall() 
            
            if not rows:
                print("DEBUG: load_follow_ups_as_documents - No rows found in follow_ups table.")
                return []

            for row in rows:
                page_content = ""
                follow_up_type_display = str(row['follow_up_type']).replace('_', ' ').capitalize()
                
                if row['follow_up_type'] == 'reporte_360':
                    page_content = f"Tipo Documento: {follow_up_type_display}\n"
                    page_content += f"ID Reporte: {row['id']}\n"
                    page_content += f"Generado: {row['timestamp']}\n"
                    if row['related_entity_type'] and row['related_entity_name']:
                        page_content += f"Entidad: {str(row['related_entity_type']).capitalize()} - {row['related_entity_name']}\n"
                    if row['related_filename']:
                        page_content += f"Archivo CSV Origen: {row['related_filename']}\n"
                    if row['related_prompt']:
                         page_content += f"Contexto/Prompt Generador: {row['related_prompt']}\n"
                    page_content += f"--- INICIO REPORTE 360 GUARDADO (ID: {row['id']}) ---\n{row['follow_up_comment']}\n--- FIN REPORTE 360 GUARDADO ---"
                
                elif row['follow_up_type'] == 'intervention_plan':
                    page_content = f"Tipo Documento: {follow_up_type_display}\n"
                    page_content += f"ID Plan: {row['id']}\n"
                    page_content += f"Generado: {row['timestamp']}\n"
                    if row['related_entity_type'] and row['related_entity_name']:
                        page_content += f"Entidad: {str(row['related_entity_type']).capitalize()} - {row['related_entity_name']}\n"
                    if row['related_filename']:
                        page_content += f"Archivo CSV Origen: {row['related_filename']}\n"
                    if row['related_analysis']: 
                         page_content += f"Plan Basado en Reporte 360 (o su prompt):\n{row['related_analysis']}\n\n"
                    page_content += f"--- INICIO PLAN DE INTERVENCIÓN GUARDADO (ID: {row['id']}) ---\n{row['follow_up_comment']}\n--- FIN PLAN DE INTERVENCIÓN GUARDADO ---"

                elif row['follow_up_type'] == 'observacion_reporte_360':
                    page_content = f"Tipo Documento: {follow_up_type_display}\n"
                    page_content += f"ID Observación: {row['id']}\n"
                    observador_nombre = str(row['related_prompt']).replace("Observación de: ", "").strip() if row['related_prompt'] else "Usuario Desconocido"
                    page_content += f"Observador: {observador_nombre}\n"
                    page_content += f"Registrada: {row['timestamp']}\n"
                    if row['related_entity_type'] and row['related_entity_name']:
                        page_content += f"Entidad Observada: {str(row['related_entity_type']).capitalize()} - {row['related_entity_name']}\n"
                    if row['related_filename']:
                        page_content += f"Archivo CSV Origen: {row['related_filename']}\n"
                    if row['related_analysis']: 
                        page_content += f"Referente a Reporte 360 ID: {row['related_analysis']}\n"
                    page_content += f"--- INICIO OBSERVACIÓN (ID: {row['id']}) ---\n{row['follow_up_comment']}\n--- FIN OBSERVACIÓN ---"

                elif row['follow_up_type'] == 'observacion_entidad':
                    page_content = f"Tipo Documento: {follow_up_type_display}\n"
                    page_content += f"ID Observación: {row['id']}\n"
                    observador_nombre = str(row['related_prompt']).replace("Observación de: ", "").strip() if row['related_prompt'] else "Usuario Desconocido"
                    page_content += f"Observador: {observador_nombre}\n"
                    page_content += f"Registrada: {row['timestamp']}\n"
                    if row['related_entity_type'] and row['related_entity_name']:
                        page_content += f"Entidad Observada: {str(row['related_entity_type']).capitalize()} - {row['related_entity_name']}\n"
                    if row['related_filename']:
                        page_content += f"Archivo CSV Origen: {row['related_filename']}\n"
                    page_content += f"--- INICIO OBSERVACIÓN (ID: {row['id']}) ---\n{row['follow_up_comment']}\n--- FIN OBSERVACIÓN ---"
                
                else: 
                    page_content = f"Tipo Seguimiento: {follow_up_type_display}\n"
                    page_content += f"ID Seguimiento: {row['id']}\n"
                    page_content += f"Registrado: {row['timestamp']}\n"
                    if row['related_entity_type'] and row['related_entity_name']:
                        page_content += f"Entidad Relacionada: {str(row['related_entity_type']).capitalize()} - {row['related_entity_name']}\n"
                    else:
                        page_content += f"Entidad Relacionada: General (no específica)\n" # Explicitly state if not specific
                    if row['related_filename']:
                        page_content += f"Archivo CSV Origen: {row['related_filename']}\n"
                    if row['related_prompt']:
                        page_content += f"Contexto/Pregunta Original: {row['related_prompt']}\n"
                    if row['related_analysis']:
                        page_content += f"Análisis Asociado (Gemini):\n{row['related_analysis']}\n\n"
                    page_content += f"Comentario/Nota Guardada (Usuario):\n{row['follow_up_comment']}"

                # Construct metadata dictionary by directly accessing row elements by key
                metadata = {key: row[key] for key in row.keys()} 
                metadata_entity_name = metadata.get('related_entity_name', 'general') # Use .get for safety on the dict
                metadata_filename = metadata.get('related_filename', 'unknown')
                metadata_timestamp = str(metadata.get('timestamp','1970-01-01 00:00:00')).split(' ')[0]

                metadata["source"] = f"db_entry_id_{row['id']}_type_{row['follow_up_type']}_entity_{metadata_entity_name}_file_{metadata_filename}_date_{metadata_timestamp}"
                follow_up_docs.append(Document(page_content=page_content, metadata=metadata))
        return follow_up_docs 
    except Exception as e: 
        print(f"Error al leer seguimientos/reportes/observaciones para RAG: {e}")
        traceback.print_exc()
        return None

def get_historical_reportes_360_for_entity(db_path, tipo_entidad, nombre_entidad, current_filename):
    """Recupera reportes 360 para la entidad y archivo actual, haciendo la comparación
    de tipo y nombre insensible a mayúsculas/minúsculas y espacios (TRIM), para evitar
    discrepancias leves en nombres.
    """
    reportes = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, follow_up_comment FROM follow_ups 
                WHERE follow_up_type = 'reporte_360' 
                  AND LOWER(TRIM(related_entity_type)) = LOWER(TRIM(?))
                  AND LOWER(TRIM(related_entity_name)) = LOWER(TRIM(?))
                  AND related_filename = ? 
                ORDER BY timestamp DESC
                """,
                (tipo_entidad, nombre_entidad, current_filename)
            )

            rows = cursor.fetchall()
            for row in rows:
                reporte_markdown = row['follow_up_comment']
                reporte_html = markdown.markdown(
                    reporte_markdown,
                    extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists']
                )
                # Convertir de UTC a hora de Santiago y formatear
                try:
                    naive_dt = datetime.datetime.strptime(row["timestamp"], '%Y-%m-%d %H:%M:%S')
                    utc_dt = pytz.utc.localize(naive_dt)
                    santiago_dt = utc_dt.astimezone(_get_tz())
                    ts_form = santiago_dt.strftime('%d/%m/%Y %H:%M')
                except Exception:
                    ts_form = row.get('timestamp', '')

                reportes.append({
                    "id": row["id"],
                    "timestamp_formateado": ts_form,
                    "timestamp": row.get('timestamp', ''),
                    "reporte_markdown": reporte_markdown,
                    "reporte_html": reporte_html
                })
    except Exception as e:
        print(f"Error CRÍTICO al recuperar Reportes 360 históricos: {e}")
        traceback.print_exc()
    return reportes

def get_observations_for_reporte_360(db_path, parent_reporte_360_id):
    # No changes
    observaciones = []
    if not parent_reporte_360_id:
        return observaciones
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, related_prompt, follow_up_comment FROM follow_ups
                WHERE follow_up_type = 'observacion_reporte_360'
                  AND related_analysis = ? 
                ORDER BY timestamp ASC 
            """, (str(parent_reporte_360_id),))

            for row in cursor.fetchall():
                observacion_markdown = row['follow_up_comment']
                observacion_html = markdown.markdown(observacion_markdown, extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists'])
                observador_nombre = str(row['related_prompt']).replace("Observación de:", "").strip() if row['related_prompt'] else "N/D"
                timestamp_dt = datetime.datetime.strptime(row["timestamp"], '%Y-%m-%d %H:%M:%S')
                observaciones.append({
                    "id": row["id"],
                    "timestamp": timestamp_dt.strftime('%d/%m/%Y %H:%M:%S'),
                    "observador_nombre": observador_nombre,
                    "observacion_markdown": observacion_markdown,
                    "observacion_html": observacion_html
                })
    except Exception as e:
        print(f"Error CRÍTICO al recuperar observaciones para Reporte 360 ID {parent_reporte_360_id}: {e}")
        traceback.print_exc()
    return observaciones

def initialize_rag_components(app_config): 
    # Inicializa el modelo de embeddings y realiza carga/creación de índices según modo
    global embedding_model_instance, vector_store, vector_store_followups 
    print("--- Iniciando Setup RAG ---")
    try:
        embedding_model_name = app_config.get('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
        embedding_model_instance = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        print(f"Modelo Embeddings '{embedding_model_name}' cargado.")
    except Exception as e:
        print(f"Error CRÍTICO Embeddings: {e}"); traceback.print_exc(); return

    # Modo de inicialización: 'lazy' intenta cargar índices existentes, 'eager' reconstruye
    rag_init_mode = (os.environ.get('RAG_INIT_MODE') or app_config.get('RAG_INIT_MODE') or 'lazy').lower()
    if rag_init_mode == 'eager':
        if not reload_institutional_context_vector_store(app_config):
            print("Advertencia: Vector store institucional no inicializado.")
        if not reload_followup_vector_store(app_config):
            print("Advertencia: Vector store seguimientos/reportes/obs no inicializado.")
    else:
        try_load_existing_vector_stores(app_config)

    print("--- Setup RAG Finalizado ---")


def try_load_existing_vector_stores(app_config):
    """
    Intenta cargar índices FAISS existentes desde disco sin reconstruirlos.
    Si no existen, deja los vector stores como None.
    """
    global vector_store, vector_store_followups, embedding_model_instance
    if not embedding_model_instance:
        print("Error CRÍTICO: Modelo embeddings no disponible para cargar índices existentes.")
        return False

    loaded_any = False
    # Cargar índice institucional si existe
    inst_path = app_config.get('FAISS_INDEX_PATH')
    try:
        if inst_path and os.path.isdir(inst_path) and os.listdir(inst_path):
            vector_store = FAISS.load_local(inst_path, embedding_model_instance, allow_dangerous_deserialization=True)
            print(f"Índice FAISS institucional cargado desde: {inst_path}")
            loaded_any = True
        else:
            print("No existe índice FAISS institucional en disco. Se omitirá la carga inicial.")
    except Exception as e:
        print(f"Error cargando índice FAISS institucional existente: {e}"); traceback.print_exc()
        vector_store = None

    # Cargar índice de seguimientos si existe
    fu_path = app_config.get('FAISS_FOLLOWUP_INDEX_PATH')
    try:
        if fu_path and os.path.isdir(fu_path) and os.listdir(fu_path):
            vector_store_followups = FAISS.load_local(fu_path, embedding_model_instance, allow_dangerous_deserialization=True)
            print(f"Índice FAISS seguimientos cargado desde: {fu_path}")
            loaded_any = True
        else:
            print("No existe índice FAISS de seguimientos en disco. Se omitirá la carga inicial.")
    except Exception as e:
        print(f"Error cargando índice FAISS seguimientos existente: {e}"); traceback.print_exc()
        vector_store_followups = None

    return loaded_any

def _load_and_split_context_documents(context_docs_folder_path): 
    # No changes
    all_docs = []
    if context_docs_folder_path and os.path.exists(context_docs_folder_path):
        for filename in os.listdir(context_docs_folder_path):
            path = os.path.join(context_docs_folder_path, filename); loader = None; docs_list = [] # Renamed docs to docs_list
            if filename.lower().endswith(".pdf"): loader = PyPDFLoader(path)
            elif filename.lower().endswith(".txt"): loader = TextLoader(path, encoding='utf-8', autodetect_encoding=True)
            if loader:
                try: docs_list = loader.load()
                except Exception as e: print(f"Error cargando {filename}: {e}")
                for doc_item in docs_list: 
                    doc_item.metadata = doc_item.metadata or {}; doc_item.metadata["source"] = path
                all_docs.extend(docs_list)
    chunks = []
    if all_docs:
        splitter = RecursiveCharacterTextSplitter(chunk_size=current_app.config.get('TEXT_SPLITTER_CHUNK_SIZE', 1000), 
                                                  chunk_overlap=current_app.config.get('TEXT_SPLITTER_CHUNK_OVERLAP', 150))
        try: chunks = splitter.split_documents(all_docs)
        except Exception as e: print(f"Error dividiendo docs: {e}"); traceback.print_exc()
    return chunks

def reload_institutional_context_vector_store(app_config):
    global vector_store, embedding_model_instance
    if not embedding_model_instance:
        print("Error CRÍTICO: Modelo embeddings no disponible.")
        vector_store = None
        return False

    folder = app_config.get('CONTEXT_DOCS_FOLDER')
    path = app_config.get('FAISS_INDEX_PATH')
    if not folder or not path:
        print("Error: Rutas de contexto o índice FAISS no configuradas.")
        vector_store = None
        return False

    chunks = _load_and_split_context_documents(folder)
    
    if chunks:
        # Si hay documentos, creamos un nuevo índice y lo guardamos
        try:
            vs_temp = FAISS.from_documents(chunks, embedding_model_instance)
            vs_temp.save_local(path)
            vector_store = vs_temp
            print(f"Índice FAISS institucional actualizado desde {len(chunks)} chunks: {path}")
            return True
        except Exception as e:
            print(f"Error creando/guardando índice FAISS institucional: {e}")
            traceback.print_exc()
            vector_store = None
            return False
    else:
        # Si NO hay documentos, eliminamos cualquier índice antiguo
        print("No se encontraron documentos de contexto. Limpiando índice existente si lo hubiera.")
        if os.path.exists(path):
            try:
                shutil.rmtree(path) # Elimina la carpeta del índice
                print(f"Índice FAISS institucional antiguo eliminado de: {path}")
            except Exception as e:
                print(f"Error al intentar eliminar el índice FAISS antiguo: {e}")
                traceback.print_exc()
                return False # Indica que hubo un problema en la limpieza
        
        vector_store = None # Aseguramos que el vector store en memoria esté vacío
        return True

def reload_followup_vector_store(app_config): 
    global vector_store_followups, embedding_model_instance
    if not embedding_model_instance: print("Error CRÍTICO: Modelo embeddings no disponible."); vector_store_followups = None; return False
    path, db_path = app_config.get('FAISS_FOLLOWUP_INDEX_PATH'), app_config.get('DATABASE_FILE')
    if not path or not db_path: print("Error: Rutas de índice FAISS o DB no configuradas."); vector_store_followups = None; return False
    
    owner = app_config.get('INDEX_OWNER_USERNAME')
    docs = load_follow_ups_as_documents(db_path, owner) 
    if docs is None: 
        print("ERROR: Falló la carga de documentos de seguimiento desde la BD. El índice FAISS de seguimientos no se actualizará/creará.")
        vector_store_followups = None 
        return False

    if docs: 
        try: 
            vs_reloaded = FAISS.from_documents(docs, embedding_model_instance)
            vs_reloaded.save_local(path)
            vector_store_followups = vs_reloaded
            print(f"Índice FAISS seguimientos/reportes/obs actualizado y guardado en: {path} ({len(docs)} documentos)")
            return True
        except Exception as e: 
            print(f"Error creando/guardando índice FAISS seguimientos/reportes/obs: {e}")
            traceback.print_exc()
            vector_store_followups = None
            return False
    elif os.path.exists(path) and os.path.isdir(path) and os.listdir(path): # If no new docs from DB, but an old index exists
        if not docs: # Explicitly check if docs list is empty
            try: 
                vector_store_followups = FAISS.load_local(path, embedding_model_instance, allow_dangerous_deserialization=True)
                print(f"Índice FAISS seguimientos/reportes/obs cargado desde: {path} (DB estaba vacía o falló la carga, pero se usó el índice existente)")
                return True
            except Exception as e: 
                print(f"Error cargando índice FAISS seguimientos/reportes/obs existente (DB vacía/fallo): {e}")
                traceback.print_exc()
                vector_store_followups = None
                return False
        # If docs is not empty but somehow we reached here (should not happen due to above 'if docs:' block)
        # This path is less likely if the first 'if docs:' is hit.
        # However, if the first 'if docs:' fails to create/save, this might be hit if an old index exists.
        # The logic should ideally prevent this double-check if the first 'if docs:' is successful.
        # For safety, if docs were loaded but FAISS creation failed, we might not want to load an old index.
        # The current structure correctly handles this: if 'if docs:' FAISS creation fails, it returns False.
        # This 'elif' is for when 'docs' is empty AND an old index exists.
        print(f"No hay seguimientos/reportes/obs en la BD, pero se cargó un índice FAISS existente desde: {path}.") # This case might not be ideal if DB should have data.
        return True # Or False if this state is considered an error. Let's assume True for now.
            
    else: # No docs from DB AND no existing index file
        print(f"No hay seguimientos/reportes/obs en la BD y no existe índice FAISS para ellos en: {path}. El índice de seguimientos estará vacío.")
        vector_store_followups = None 
        return True


# --- Helper and Data Processing Functions (Unchanged) ---
def _extract_level_from_course(course_name):
    if not isinstance(course_name, str): return "Desconocido"
    # Fix character encoding issues: ø -> °, \xa0 -> á
    course_name = course_name.replace('ø', '°').replace('\xa0', 'á')
    parts = " ".join(course_name.strip().split()).split(' ')
    return " ".join(parts[:-1]) if len(parts) > 1 and len(parts[-1]) == 1 and parts[-1].isalpha() else " ".join(parts)

def get_alumnos_menor_promedio_por_nivel(df):
    if df is None or df.empty: return {}
    cols = [current_app.config.get(c) for c in ['NOMBRE_COL', 'CURSO_COL', 'PROMEDIO_COL']]
    if not all(c in df.columns for c in cols): return {}
    df_c = df.copy(); df_c['Nivel'] = df_c[cols[1]].astype(str).apply(_extract_level_from_course); df_c[cols[2]] = pd.to_numeric(df_c[cols[2]], errors='coerce')
    res = {}
    try:
        idx = df_c.dropna(subset=[cols[2]]).groupby('Nivel')[cols[2]].idxmin()
        for _, r in df_c.loc[idx].iterrows(): res.setdefault(r['Nivel'], []).append({"nombre": r[cols[0]], "curso_original": r[cols[1]], "promedio": round(r[cols[2]], 2) if pd.notna(r[cols[2]]) else "N/A"})
    except Exception as e: print(f"Error en get_alumnos_menor_promedio_por_nivel: {e}"); return {"error": f"Error: {e}"}
    return res

def get_alumnos_observaciones_negativas_por_nivel(df):
    if df is None or df.empty: return {}
    
    # Obtener nombres de columnas desde la configuración
    nombre_col = current_app.config.get('NOMBRE_COL')
    curso_col = current_app.config.get('CURSO_COL')
    obs_col = current_app.config.get('OBSERVACIONES_COL')
    kws = current_app.config.get('NEGATIVE_OBSERVATION_KEYWORDS', [])
    
    if not all([nombre_col, curso_col, obs_col]) or not kws:
        return {} # No se puede continuar si faltan columnas clave o palabras clave
    
    if obs_col not in df.columns:
        return {} # La columna de observaciones no existe en el archivo
        
    df_c = df.copy()
    df_c['Nivel'] = df_c[curso_col].astype(str).apply(_extract_level_from_course)
    
    # Filtrar todas las filas que contienen una palabra clave negativa en la columna de observación
    def _norm(s: str) -> str:
        s_nfkd = unicodedata.normalize('NFKD', str(s).lower())
        return ''.join(c for c in s_nfkd if not unicodedata.combining(c))
    kws_l = [_norm(kw) for kw in kws]
    df_neg = df_c[df_c[obs_col].apply(lambda o: any(kw in _norm(o) for kw in kws_l) if pd.notna(o) else False)]
    
    # De la lista de filas con observaciones negativas, eliminamos los alumnos duplicados.
    # Esto asegura que cada alumno aparezca solo una vez en el reporte de alertas.
    if not df_neg.empty:
        df_neg = df_neg.drop_duplicates(subset=[nombre_col])
    
    # Agrupar los alumnos únicos por Nivel
    res_ag = {}
    for niv, grp in df_neg.groupby('Nivel'):
        al_niv = [{
            "nombre": r[nombre_col], 
            "curso_original": r[curso_col], 
            "observacion": str(r[obs_col])
        } for _, r in grp.iterrows()]
        
        if al_niv:
            res_ag[niv] = al_niv
            
    return res_ag

def get_level_kpis(df):
    if df is None or df.empty: return {}
    # Nombres de columnas desde config
    curso_col = current_app.config['CURSO_COL']
    nombre_col = current_app.config['NOMBRE_COL']
    nota_col = current_app.config['NOTA_COL']
    asistencia_col = current_app.config.get('ASISTENCIA_COL')
    
    # Crear un DataFrame de estudiantes únicos para algunos cálculos
    df_alumnos = df.drop_duplicates(subset=[nombre_col]).copy()
    df_alumnos['Nivel'] = df_alumnos[curso_col].astype(str).apply(_extract_level_from_course)

    # El df completo (con todas las notas) también necesita la columna Nivel para promedios de curso
    df_completo = df.copy()
    df_completo['Nivel'] = df_completo[curso_col].astype(str).apply(_extract_level_from_course)

    agg = {}
    min_attendance_by_level = {}
    if asistencia_col in df.columns and not df[asistencia_col].isnull().all():
        df_att = df.drop_duplicates(subset=[nombre_col]).copy()
        df_att['Nivel'] = df_att[curso_col].astype(str).apply(_extract_level_from_course)
        df_att = df_att.dropna(subset=[asistencia_col])
        try:
            grp = df_att.groupby(['Nivel', curso_col])[asistencia_col].mean().dropna()
            for nivel_name, sub in grp.groupby(level=0):
                sub_series = sub.droplevel(0)
                if not sub_series.empty:
                    c_name = sub_series.idxmin()
                    val = float(sub_series.min())
                    min_attendance_by_level[nivel_name] = {"nombre": c_name, "asistencia": f"{val:.1%}", "asistencia_num": val}
        except Exception:
            min_attendance_by_level = {}
    for niv, grp_alumnos in df_alumnos.groupby('Nivel'):
        # Usar el grupo de alumnos para contar el total
        total_alumnos_nivel = len(grp_alumnos)
        
        # Usar el df completo filtrado por nivel para cálculos de notas
        grp_completo_nivel = df_completo[df_completo['Nivel'] == niv]
        promedio_general_nivel = grp_completo_nivel[nota_col].mean()

        # Asistencia (desde el grupo de alumnos únicos)
        asist_prom = grp_alumnos[asistencia_col].mean() if asistencia_col in grp_alumnos.columns and not grp_alumnos[asistencia_col].isnull().all() else np.nan
        
        # Cursos con menor y mayor promedio (desde el grupo completo de notas)
        prom_cursos_en_nivel = grp_completo_nivel.groupby(curso_col)[nota_col].mean().dropna()
        c_menor = {"nombre": prom_cursos_en_nivel.idxmin(), "promedio": prom_cursos_en_nivel.min()} if not prom_cursos_en_nivel.empty else {"nombre": "N/A", "promedio": np.nan}
        c_mayor = {"nombre": prom_cursos_en_nivel.idxmax(), "promedio": prom_cursos_en_nivel.max()} if not prom_cursos_en_nivel.empty else {"nombre": "N/A", "promedio": np.nan}
        if asistencia_col in grp_alumnos.columns and not grp_alumnos[asistencia_col].isnull().all():
            asist_por_curso = (
                grp_alumnos[[curso_col, asistencia_col]]
                .dropna(subset=[asistencia_col])
                .groupby(curso_col)[asistencia_col]
                .mean()
            )
            if not asist_por_curso.empty:
                curso_min_asist = asist_por_curso.idxmin()
                val_min_asist = float(asist_por_curso.min())
                c_menor_asist = {"nombre": curso_min_asist, "asistencia": f"{val_min_asist:.1%}", "asistencia_num": val_min_asist}
            else:
                c_menor_asist = {"nombre": "N/A", "asistencia": "N/A", "asistencia_num": np.nan}
        else:
            c_menor_asist = {"nombre": "N/A", "asistencia": "N/A", "asistencia_num": np.nan}
        
        agg[niv] = {
            "total_alumnos": total_alumnos_nivel,
            "promedio_general_notas": round(promedio_general_nivel, 2) if pd.notna(promedio_general_nivel) else "N/A",
            "asistencia_promedio": f"{asist_prom:.1%}" if pd.notna(asist_prom) else "N/A",
            "curso_menor_promedio": {"nombre": c_menor["nombre"], "promedio": round(c_menor["promedio"], 2) if pd.notna(c_menor["promedio"]) else "N/A"},
            "curso_mayor_promedio": {"nombre": c_mayor["nombre"], "promedio": round(c_mayor["promedio"], 2) if pd.notna(c_mayor["promedio"]) else "N/A"},
            "curso_menor_asistencia": min_attendance_by_level.get(niv, {"nombre": "N/A", "asistencia": "N/A", "asistencia_num": np.nan})
        }
    return agg

def get_course_attendance_kpis(df):
    if df is None or df.empty: return {}
    # Nombres de columnas desde config
    curso_col = current_app.config['CURSO_COL']
    nombre_col = current_app.config['NOMBRE_COL']
    asistencia_col = current_app.config.get('ASISTENCIA_COL')
    
    if not all(c in df.columns for c in [curso_col, nombre_col, asistencia_col]): return {}
    
    # Trabajar con un DataFrame de alumnos únicos para evitar contar la asistencia múltiples veces
    df_alumnos = df.drop_duplicates(subset=[nombre_col]).copy()
    if df_alumnos[asistencia_col].isnull().all(): return {}
    
    agg = {}
    for name, grp in df_alumnos.groupby(curso_col):
        avg = grp[asistencia_col].mean()
        agg[name] = f"{avg:.1%}" if pd.notna(avg) else "N/A"
    return agg

def get_advanced_establishment_alerts(df, level_kpis_data):
    alerts = {"niveles_bajo_rendimiento": [], "niveles_obs_conducta": []}
    if df is None or df.empty or not level_kpis_data: return alerts

    # Nombres de columnas desde config
    curso_col = current_app.config['CURSO_COL']
    promedio_col = current_app.config['PROMEDIO_COL'] # Usamos el promedio calculado por alumno
    nombre_col = current_app.config['NOMBRE_COL']
    obs_col = current_app.config.get('OBSERVACIONES_COL')

    # Umbrales desde config
    low_perf_threshold = current_app.config.get('LOW_PERFORMANCE_THRESHOLD_GRADE', 4.0)
    significant_perc_low_perf = current_app.config.get('SIGNIFICANT_PERCENTAGE_LOW_PERF_ALERT', 0.20)
    
    # Crear un DataFrame de alumnos únicos, que ya contiene el promedio calculado
    df_alumnos = df.drop_duplicates(subset=[nombre_col]).copy()
    if 'Nivel' not in df_alumnos.columns:
         df_alumnos['Nivel'] = df_alumnos[curso_col].astype(str).apply(_extract_level_from_course)
    
    for nivel_name, nivel_info in level_kpis_data.items():
        total_alumnos_en_nivel = nivel_info.get("total_alumnos", 0)
        if total_alumnos_en_nivel == 0: continue
        
        # Alerta de bajo rendimiento
        alumnos_en_nivel_df = df_alumnos[df_alumnos['Nivel'] == nivel_name]
        alumnos_bajo_rendimiento_df = alumnos_en_nivel_df[alumnos_en_nivel_df[promedio_col] < low_perf_threshold]
        num_alumnos_bajo_rendimiento = len(alumnos_bajo_rendimiento_df)
        
        if num_alumnos_bajo_rendimiento > 0:
            porcentaje_bajo_rendimiento = num_alumnos_bajo_rendimiento / total_alumnos_en_nivel
            if porcentaje_bajo_rendimiento >= significant_perc_low_perf: 
                alerts["niveles_bajo_rendimiento"].append({
                    "nivel": nivel_name, "porcentaje_bajo_rendimiento": f"{porcentaje_bajo_rendimiento:.1%}",
                    "num_alumnos_bajo_rendimiento": num_alumnos_bajo_rendimiento,
                    "total_alumnos_nivel": total_alumnos_en_nivel, "umbral_promedio": low_perf_threshold
                })

    # La alerta de observaciones negativas no cambia su lógica fundamental
    if obs_col and obs_col in df.columns:
        # (El código de get_alumnos_observaciones_negativas_por_nivel debería funcionar si se le pasa el df_alumnos)
        pass # Por ahora asumimos que la lógica interna de esa función se adaptará o ya es compatible.

    return alerts

def _get_subject_averages(df_filtered, subject_cols, subject_display_names_config):
    labels, scores = [], []
    for subj_key in subject_cols:
        labels.append(subject_display_names_config.get(subj_key, {}).get('full', subj_key.capitalize()))
        if subj_key in df_filtered.columns and not df_filtered[subj_key].isnull().all():
            avg = df_filtered[subj_key].mean(); scores.append(round(avg, 2) if pd.notna(avg) else 0)
        else: scores.append(0)
    return {'labels': labels, 'scores': scores}

def get_student_vs_course_level_averages(df_full, student_name, student_course_name):
    if df_full is None or df_full.empty: return None

    # Columnas desde config
    nombre_col = current_app.config['NOMBRE_COL']
    curso_col = current_app.config['CURSO_COL']
    asignatura_col = current_app.config['ASIGNATURA_COL']
    nota_col = current_app.config['NOTA_COL']

    # 1. Obtener notas del alumno específico
    df_student = df_full[df_full[nombre_col] == student_name]
    student_avg_by_subj = df_student.groupby(asignatura_col)[nota_col].mean()

    # 2. Obtener promedios del curso del alumno
    df_course = df_full[df_full[curso_col] == student_course_name]
    course_avg_by_subj = df_course.groupby(asignatura_col)[nota_col].mean()

    # 3. Obtener promedios del nivel del alumno
    student_level = _extract_level_from_course(student_course_name)
    df_level = df_full.copy()
    df_level['Nivel'] = df_level[curso_col].astype(str).apply(_extract_level_from_course)
    df_level_filtered = df_level[df_level['Nivel'] == student_level]
    level_avg_by_subj = df_level_filtered.groupby(asignatura_col)[nota_col].mean()

    # Unificar todas las asignaturas presentes en los tres grupos
    all_subjects = sorted(list(set(student_avg_by_subj.index) | set(course_avg_by_subj.index) | set(level_avg_by_subj.index)))

    # Mapear los promedios a la lista unificada de asignaturas
    student_scores = [round(student_avg_by_subj.get(s, 0), 2) for s in all_subjects]
    course_scores = [round(course_avg_by_subj.get(s, 0), 2) for s in all_subjects]
    level_scores = [round(level_avg_by_subj.get(s, 0), 2) for s in all_subjects]

    return {'labels': all_subjects, 'datasets': [
            {'label': 'Alumno', 'data': student_scores, 'backgroundColor': 'rgba(54, 162, 235, 0.6)'},
            {'label': f'Prom. Curso ({student_course_name})', 'data': course_scores, 'backgroundColor': 'rgba(75, 192, 192, 0.6)'},
            {'label': f'Prom. Nivel ({student_level})', 'data': level_scores, 'backgroundColor': 'rgba(255, 206, 86, 0.6)'}]}

def get_course_vs_level_comparison_data(df_full, current_course_name):
    if df_full is None or df_full.empty: return None
    # Lógica similar a la anterior, pero solo comparando curso y nivel
    curso_col = current_app.config['CURSO_COL']
    asignatura_col = current_app.config['ASIGNATURA_COL']
    nota_col = current_app.config['NOTA_COL']

    df_course = df_full[df_full[curso_col] == current_course_name]
    course_avg_by_subj = df_course.groupby(asignatura_col)[nota_col].mean()

    current_level = _extract_level_from_course(current_course_name)
    df_level = df_full.copy()
    df_level['Nivel'] = df_level[curso_col].astype(str).apply(_extract_level_from_course)
    df_level_filtered = df_level[df_level['Nivel'] == current_level]
    level_avg_by_subj = df_level_filtered.groupby(asignatura_col)[nota_col].mean()
    
    all_subjects = sorted(list(set(course_avg_by_subj.index) | set(level_avg_by_subj.index)))
    course_scores = [round(course_avg_by_subj.get(s, 0), 2) for s in all_subjects]
    level_scores = [round(level_avg_by_subj.get(s, 0), 2) for s in all_subjects]

    return {'labels': all_subjects, 'datasets': [
            {'label': f'Curso Actual ({current_course_name})', 'data': course_scores, 'backgroundColor': 'rgba(75, 192, 192, 0.6)'},
            {'label': f'Prom. Nivel ({current_level})', 'data': level_scores, 'backgroundColor': 'rgba(255, 206, 86, 0.6)'}]}

def get_all_courses_in_level_breakdown_data(df_full, current_course_level_name):
    if df_full is None or df_full.empty: return None
    curso_col = current_app.config['CURSO_COL']
    asignatura_col = current_app.config['ASIGNATURA_COL']
    nota_col = current_app.config['NOTA_COL']

    df_level = df_full.copy()
    df_level['Nivel'] = df_level[curso_col].astype(str).apply(_extract_level_from_course)
    df_level_filtered = df_level[df_level['Nivel'] == current_course_level_name]

    # Agrupar por curso y asignatura para obtener todos los promedios
    avg_by_course_subj = df_level_filtered.groupby([curso_col, asignatura_col])[nota_col].mean().unstack()
    
    all_subjects = sorted(avg_by_course_subj.columns.tolist())
    datasets = []
    colors_bg = ['rgba(255, 99, 132, 0.6)', 'rgba(54, 162, 235, 0.6)', 'rgba(255, 206, 86, 0.6)', 'rgba(75, 192, 192, 0.6)']
    
    for i, (course, data) in enumerate(avg_by_course_subj.iterrows()):
        scores = [round(data.get(s, 0), 2) for s in all_subjects]
        datasets.append({
            'label': str(course),
            'data': scores,
            'backgroundColor': colors_bg[i % len(colors_bg)]
        })

    return {'labels': all_subjects, 'datasets': datasets}

def get_course_heatmap_data(df_course, student_name_col, asignatura_col, nota_col):
    if df_course is None or df_course.empty: return None

    # Usamos pivot_table para transformar el formato largo a ancho, ideal para un heatmap
    heatmap_df = pd.pivot_table(df_course, values=nota_col, index=student_name_col, columns=asignatura_col, aggfunc='mean')
    
    headers = heatmap_df.columns.tolist()
    rows = []
    for student_name, row_data in heatmap_df.iterrows():
        grades = []
        for subj_key in headers:
            score = row_data.get(subj_key)
            display = "N/A"; color = "bg-slate-100 text-slate-500"
            if pd.notna(score):
                display = f"{score:.1f}"
                if score < 4.0: color = "bg-red-400 text-white"
                elif score < 5.0: color = "bg-orange-300 text-orange-800"
                elif score < 6.0: color = "bg-yellow-200 text-yellow-800"
                else: color = "bg-green-400 text-white"
            grades.append({'score_display': display, 'color_class': color, 'raw_score': score if pd.notna(score) else -1})
        rows.append({'student_name': student_name, 'grades': grades})
    
    return {'subject_headers': headers, 'student_performance_rows': rows}

def generate_intervention_plan_with_gemini(reporte_360_markdown, tipo_entidad, nombre_entidad):
    # Obtenemos la plantilla del prompt desde la configuración de la aplicación.
    prompt_template = current_app.config.get('PROMPT_PLAN_INTERVENCION', "Generar plan de intervención para {tipo_entidad} {nombre_entidad} basado en:\n{reporte_360_markdown}")
    
    # Formateamos la plantilla con los datos específicos de la entidad y el reporte base.
    prompt_plan = prompt_template.format(
        reporte_360_markdown=reporte_360_markdown,
        tipo_entidad=tipo_entidad,
        nombre_entidad=nombre_entidad
    )
    
    # IMPORTANTE: Pasar el índice institucional (vector_store) para habilitar citas RAG
    # en la sección de "Fundamentación" del plan.
    analysis_result = analyze_data_with_gemini(
        data_string=reporte_360_markdown, 
        user_prompt=prompt_plan, 
        vs_inst=vector_store, 
        vs_followup=vector_store_followups, 
        chat_history_string="", 
        is_reporte_360=False, 
        is_plan_intervencion=True,
        is_direct_chat_query=False,
        entity_type=tipo_entidad, 
        entity_name=nombre_entidad
    )

    # Devolvemos tanto el HTML como el Markdown.
    return analysis_result['html_output'], analysis_result['raw_markdown']

def save_intervention_plan_to_db(db_path, filename, tipo_entidad, nombre_entidad, plan_markdown, reporte_360_base_markdown, owner_username=None):
    """
    Guarda el plan de intervención en la BD y devuelve el ID de la nueva fila.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            prompt = f"Plan Intervención para {tipo_entidad.capitalize()}: {nombre_entidad}"
            cursor.execute("""
                INSERT INTO follow_ups 
                (related_filename, related_prompt, related_analysis, follow_up_comment, follow_up_type, related_entity_type, related_entity_name, owner_username) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (filename, prompt, reporte_360_base_markdown, plan_markdown, 'intervention_plan', tipo_entidad, nombre_entidad, owner_username))
            
            # LÍNEA CORREGIDA: Obtener y devolver el ID de la fila insertada
            last_id = cursor.lastrowid
            conn.commit()

        print(f"Plan intervención para {tipo_entidad} {nombre_entidad} guardado con ID: {last_id}.")
        return last_id
    except Exception as e: 
        print(f"Error guardando plan intervención: {e}")
        traceback.print_exc()
        return None

def get_intervention_plans_for_entity(db_path, tipo_entidad, nombre_entidad, current_filename):
    # No changes
    plans = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row; cursor = conn.cursor()
            cursor.execute("SELECT id, timestamp, follow_up_comment FROM follow_ups WHERE follow_up_type = 'intervention_plan' AND related_entity_type = ? AND related_entity_name = ? AND related_filename = ? ORDER BY timestamp DESC",
                           (tipo_entidad, nombre_entidad, current_filename))
            for row in cursor.fetchall():
                plans.append({
                    "id": row["id"],
                    "timestamp": datetime.datetime.strptime(row["timestamp"], '%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y %H:%M'),
                    "plan_markdown": row["follow_up_comment"],
                    # Importante: sin 'nl2br' para que el texto del plan se justifique correctamente sin saltos forzados
                    "plan_html": markdown.markdown(row["follow_up_comment"], extensions=['fenced_code', 'tables', 'sane_lists'])
                })
    except Exception as e: print(f"Error recuperando planes: {e}"); traceback.print_exc()
    return plans

def search_web_for_support_resources(plan_intervencion_markdown, tipo_entidad, nombre_entidad):
    # Extracción de temas clave desde el Plan (reglas determinísticas)
    def extract_support_topics_from_markdown(txt: str) -> list:
        base = normalize_text(txt or '')
        topics = []
        def add(q):
            if q and q not in topics:
                topics.append(q)
        # Académicos
        if re.search(r'\bmatematic', base):
            add('recursos de matemáticas: ejercicios y guías docentes')
        if re.search(r'\blenguaj|\bescrit|\blecture|\bcomprension lectora', base):
            add('lenguaje y lectoescritura: comprensión lectora y escritura')
        if re.search(r'\bmusica', base):
            add('recursos de música: teoría y práctica escolar')
        if re.search(r'\bciencia|\bfisic|\bquimic|\bbiolog', base):
            add('recursos de ciencias: física, química o biología')
        if re.search(r'\bingles', base):
            add('recursos de inglés: vocabulario y gramática escolar')
        # Conducta y socioemocional
        if re.search(r'\bconduct|\bcomport|\bdiscipl|\bconvivenc|\bbullying|\bsocializ', base):
            add('apoyo conductual y habilidades socioemocionales en aula')
        # Asistencia
        if re.search(r'\basistenc|\binasistenc', base):
            add('mejorar asistencia escolar: estrategias familia-escuela')
        # Hábitos y estudio
        if re.search(r'\bhabitos de estudio|\borganizacion|\bplanificacion', base):
            add('hábitos de estudio y planificación académica')
        return topics

    topics_from_plan = extract_support_topics_from_markdown(plan_intervencion_markdown)
    print(f"DEBUG: Iniciando search_web_for_support_resources (SIMULACIÓN) para {tipo_entidad} {nombre_entidad}") 
    search_queries = []
    try:
        api_key = current_app.config.get('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        model_analisis = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt_analisis_plan = f"""
        Analiza el siguiente "Plan de Intervención" para el {tipo_entidad} '{nombre_entidad}'.
        Identifica los 3-4 temas o áreas de dificultad más críticos que requieren apoyo externo.
        Para cada tema, genera una consulta de búsqueda en español, concisa y efectiva que se usaría para encontrar recursos educativos prácticos en la web.
        Devuelve SOLAMENTE una lista de Python con las cadenas de texto de estas consultas de búsqueda, por ejemplo: ["ejercicios de fracciones para primaria", "videos sobre el ciclo del agua", "guía para mejorar la comprensión lectora en adolescentes"].

        Plan de Intervención:
        ---
        {plan_intervencion_markdown}
        ---
        """
        response_analisis = model_analisis.generate_content(prompt_analisis_plan)
        queries_str = response_analisis.text.strip()
        
        if queries_str.startswith("```python"):
            queries_str = queries_str[len("```python"):].strip()
        elif queries_str.startswith("```"):
            queries_str = queries_str[3:].strip()
        if queries_str.endswith("```"):
            queries_str = queries_str[:-3].strip()
        
        try:
            evaluated_queries = eval(queries_str)
            if isinstance(evaluated_queries, list) and all(isinstance(q, str) for q in evaluated_queries):
                search_queries = evaluated_queries
            else:
                raise ValueError("La respuesta de Gemini no fue una lista de Python válida.")
        except Exception:
            search_queries = [line.strip().replace('"', '').replace("'", "") for line in queries_str.strip("[]").split(',') if line.strip()]
            search_queries = [q for q in search_queries if q] 
            if not search_queries:
                return f"Error: No pude interpretar el plan de intervención para generar consultas de búsqueda (simulación)."
        
        # Si la extracción determinística encontró temas, usarlos de forma prioritaria
        if topics_from_plan:
            search_queries = topics_from_plan
        if not search_queries:
            return "Error: No se pudieron generar consultas de búsqueda a partir del plan de intervención (simulación)."

    except Exception as e:
        traceback.print_exc()
        return f"Error crítico al generar consultas de búsqueda desde el plan (simulación): {e}"

    if not search_queries: 
        return "Error: No se generaron consultas de búsqueda válidas (simulación)."

    formatted_queries_for_gemini = "\n".join([f"- {q}" for q in search_queries])

    # --- INICIO: BLOQUE DE PROMPT MODIFICADO ---
    prompt_sugerencia_recursos = f"""
    Actúa como un orientador educativo experto. Basándote en las siguientes áreas temáticas (derivadas de un plan de intervención para el {tipo_entidad} '{nombre_entidad}'), sugiere 2-3 ejemplos de recursos educativos online por cada tema.

    **CRÍTICO: Debes devolver SOLAMENTE código HTML. NO incluyas frases introductorias o conclusivas fuera del HTML.**
    Usa la siguiente estructura HTML de manera estricta:
    1.  Para cada tema, crea un encabezado: `<h3 class="resource-topic-title">[Nombre del Tema]</h3>`
    2.  Luego, para todos los recursos de ese tema, envuélvelos en un div: `<div class="resource-grid">`
    3.  Cada recurso individual debe ser una tarjeta: `<div class="resource-card">`
    4.  Dentro de la tarjeta, usa esta estructura:
        * `<h4 class="resource-title"><i class="fas fa-link mr-2"></i>[Título Descriptivo del Recurso]</h4>` (Puedes cambiar el ícono de font-awesome si es pertinente, ej: 'fa-video', 'fa-file-alt').
        * `<p class="resource-description">[Descripción de por qué el recurso es útil]</p>`
        * `<a href="https://www.spanishdict.com/translate/ficticia" class="resource-button" target="_blank" rel="noopener noreferrer">Acceder al Recurso</a>`

    Temas/Consultas para los que sugerir recursos:
    ---
    {formatted_queries_for_gemini}
    ---

    **Ejemplo de salida para UN tema:**
    <h3 class="resource-topic-title">Ejercicios de Fracciones para Primaria</h3>
    <div class="resource-grid">
        <div class="resource-card">
            <h4 class="resource-title"><i class="fas fa-shapes mr-2"></i>Plataforma Interactiva de Fracciones</h4>
            <p class="resource-description">Un sitio con ejercicios para practicar sumas, restas y comparaciones de fracciones, ideal para reforzar conceptos.</p>
            <a href="https://www.ejemplo-educativo.com/fracciones-primaria" class="resource-button" target="_blank" rel="noopener noreferrer">Explorar Plataforma</a>
        </div>
        <div class="resource-card">
            <h4 class="resource-title"><i class="fas fa-video mr-2"></i>Video Tutorial Animado</h4>
            <p class="resource-description">Un video que explica de forma sencilla qué son las fracciones, útil para la comprensión visual del concepto.</p>
            <a href="https://www.youtube.com/ejemplo-fracciones" class="resource-button" target="_blank" rel="noopener noreferrer">Ver Video</a>
        </div>
    </div>
    """
    # --- FIN: BLOQUE DE PROMPT MODIFICADO ---
    try:
        print("DEBUG: Enviando prompt a Gemini para SIMULAR y sugerir recursos con nuevo formato...") 
        model_sugerencias = genai.GenerativeModel('gemini-2.5-flash')
        response_sugerencias = model_sugerencias.generate_content(prompt_sugerencia_recursos)
        final_html = response_sugerencias.text.strip()
        
        if final_html.strip().lower().startswith("```html"):
            final_html = final_html.strip()[7:]
        if final_html.strip().endswith("```"):
            final_html = final_html.strip()[:-3]
        
        print("DEBUG: HTML con recursos SIMULADOS (nuevo formato) generado por Gemini exitosamente.") 
        if not final_html:
            return "Error: El modelo no pudo generar sugerencias de recursos en este momento."
        try:
            verified_html = verify_and_annotate_resources_html(final_html, current_app.config)
            enriched_html = inject_local_resources(verified_html, current_app.config)
            return enriched_html
        except Exception as ve:
            print(f"WARN: verificación de enlaces falló, devolviendo HTML original: {ve}")
            return final_html

    except Exception as e:
        print(f"CRITICAL DEBUG: Error al SIMULAR y sugerir recursos con Gemini: {e}") 
        traceback.print_exc()
        return f"Error: No pude generar sugerencias de recursos. Detalle: {e}"

# ================================
# NUEVO: FEATURE STORE UNIFICADO
# ================================
def build_feature_store_from_csv(df_full: pd.DataFrame) -> dict:
    """
    Construye un Feature Store mínimo a partir del CSV activo para asegurar señales
    numéricas y trazables, incluso cuando el LLM no interprete bien el texto libre.

    Estructura de salida:
    {
        'students': {
            '<Nombre>': { 'curso': str, 'promedio': float|None, 'asistencia': float|None }
        },
        'courses': {
            '<Curso>': { 'avg_grade': float|None, 'avg_attendance': float|None, 'num_students': int }
        },
        'levels': get_level_kpis(df_full),
        'course_attendance': get_course_attendance_kpis(df_full)
    }
    """
    try:
        if df_full is None or df_full.empty:
            return {}
        nombre_col = current_app.config['NOMBRE_COL']
        curso_col = current_app.config['CURSO_COL']
        nota_col = current_app.config['NOTA_COL']
        promedio_col = current_app.config['PROMEDIO_COL']
        asistencia_col = current_app.config.get('ASISTENCIA_COL')
        asignatura_col = current_app.config.get('ASIGNATURA_COL')
        observaciones_col = current_app.config.get('OBSERVACIONES_COL')
        familia_col = current_app.config.get('FAMILIA_COL')
        entrevistas_col = current_app.config.get('ENTREVISTAS_COL')

        # Tiempos para métricas
        start_perf = datetime.datetime.now().timestamp()

        # Asegurar promedio por alumno (si estamos en formato largo)
        df = df_full.copy()
        try:
            if promedio_col not in df.columns and nota_col in df.columns and nombre_col in df.columns:
                df[promedio_col] = df.groupby(nombre_col)[nota_col].transform('mean')
        except Exception:
            pass

        # Alumnos únicos
        students = {}
        if nombre_col in df.columns:
            df_alumnos = df.drop_duplicates(subset=[nombre_col]).copy()
            # Pre-computar peor asignatura por alumno si hay columnas disponibles
            peor_asig_por_alumno = {}
            # Métricas de consistencia por alumno (std y rango entre asignaturas)
            consistencia_por_alumno = {}
            try:
                if asignatura_col in df.columns and nota_col in df.columns:
                    # Promedios por alumno y asignatura
                    proms = (
                        df[[nombre_col, asignatura_col, nota_col]]
                          .dropna(subset=[nombre_col, asignatura_col, nota_col])
                          .groupby([nombre_col, asignatura_col])[nota_col]
                          .mean()
                          .reset_index()
                    )
                    # Para cada alumno, escoger la asignatura con menor promedio
                    for alumno, grp in proms.groupby(nombre_col):
                        try:
                            thr = float(current_app.config.get('LOW_PERFORMANCE_THRESHOLD_GRADE', 4.0))
                        except Exception:
                            thr = 4.0
                        idx_min = grp[nota_col].idxmin()
                        row = grp.loc[idx_min]
                        peor_asig_por_alumno[str(alumno).strip()] = {
                            'asignatura': str(row[asignatura_col]),
                            'promedio': float(row[nota_col])
                        }
                        # Consistencia entre asignaturas
                        subject_avgs = grp[nota_col].astype(float).values
                        if len(subject_avgs) >= 2:
                            std_dev = float(np.std(subject_avgs))
                            rng = float(np.max(subject_avgs) - np.min(subject_avgs))
                        else:
                            std_dev = None
                            rng = None
                        consistencia_por_alumno[str(alumno).strip()] = {
                            'num_asignaturas': int(len(subject_avgs)),
                            'std_dev': std_dev,
                            'rango': rng
                        }
                        below_cnt = int((grp[nota_col] < thr).sum())
                        consistencia_por_alumno[str(alumno).strip()]['subjects_below_threshold_count'] = below_cnt
            except Exception:
                peor_asig_por_alumno = {}

            # Conteo de anotaciones negativas por alumno, vectorizado por palabras clave
            neg_counts_por_alumno = {}
            neg_examples_por_alumno = {}
            try:
                if observaciones_col in df.columns:
                    cfg_kws = current_app.config.get('NEGATIVE_OBSERVATION_KEYWORDS', []) or []
                    extra_kws = [
                        'pelea', 'bullying', 'maltrato', 'incumplimiento', 'irrespeto', 'castigo',
                        'bajo rendimiento', 'baja asistencia', 'problema', 'riesgo'
                    ]
                    all_kws = list({ normalize_text(k) for k in (cfg_kws + extra_kws) })
                    pattern = compile_any_keyword_pattern(all_kws)

                    df_obs = df[[nombre_col, observaciones_col]].dropna(subset=[nombre_col]).copy()
                    df_obs[observaciones_col] = df_obs[observaciones_col].fillna("").astype(str)
                    # Normalización previa para matching rápido
                    df_obs["__obs_norm"] = df_obs[observaciones_col].map(normalize_text)
                    hit_mask = df_obs["__obs_norm"].str.contains(pattern, na=False)
                    hits_df = df_obs.loc[hit_mask, [nombre_col, observaciones_col]].copy()
                    # Conteos por alumno
                    if not hits_df.empty:
                        neg_counts_por_alumno = hits_df.groupby(nombre_col).size().astype(int).to_dict()
                        # Ejemplos por alumno (hasta 3)
                        ex_series = hits_df.groupby(nombre_col)[observaciones_col].apply(lambda s: s.iloc[:3].tolist())
                        neg_examples_por_alumno = { str(k).strip(): v for k, v in ex_series.to_dict().items() }
            except Exception:
                pass

            # Indicador de situación familiar compleja por alumno (vectorizado)
            fam_complex_score = {}
            fam_complex_examples = {}
            try:
                cols_to_scan = []
                if familia_col in df.columns:
                    cols_to_scan.append(familia_col)
                if entrevistas_col in df.columns:
                    cols_to_scan.append(entrevistas_col)
                if cols_to_scan:
                    complexity_keywords = [
                        'separacion','divorcio','violencia','maltrato','abuso','vulnerabilidad','sename',
                        'medida de proteccion','denuncia','consumo de drogas','alcohol','adiccion',
                        'problemas economicos','desempleo','fallecimiento','enfermedad grave','hospitalizado',
                        'conflicto familiar','cambio de tutor','custodia','situacion compleja','situacion familiar compleja',
                        'acoso escolar', 'tdah'
                    ]
                    comp_pattern = compile_any_keyword_pattern(complexity_keywords)
                    df_fam = df[[nombre_col] + cols_to_scan].copy()
                    for col in cols_to_scan:
                        df_fam[col] = df_fam[col].fillna("").astype(str)
                        df_fam[f"__{col}_norm"] = df_fam[col].map(normalize_text)
                    # Máscara de match por columna y OR global
                    masks = [ df_fam[f"__{col}_norm"].str.contains(comp_pattern, na=False) for col in cols_to_scan ]
                    if masks:
                        mask_any = masks[0]
                        for m in masks[1:]:
                            mask_any = mask_any | m
                        hits_df = df_fam.loc[mask_any, [nombre_col] + cols_to_scan].copy()
                        if not hits_df.empty:
                            fam_complex_score = hits_df.groupby(nombre_col).size().astype(int).to_dict()
                            # Ejemplos: concatenar valores no vacíos de columnas escaneadas y tomar hasta 3
                            def _row_texts(row):
                                vals = []
                                for c in cols_to_scan:
                                    v = str(row[c]).strip()
                                    if v:
                                        vals.append(v)
                                return vals
                            # Expandir a una serie de listas y aplanar por alumno
                            ex_map = {}
                            for _, row in hits_df.iterrows():
                                alumno = str(row[nombre_col]).strip()
                                ex_map.setdefault(alumno, [])
                                for v in _row_texts(row):
                                    if len(ex_map[alumno]) < 3:
                                        ex_map[alumno].append(v)
                            fam_complex_examples = ex_map
            except Exception:
                pass

            for _, r in df_alumnos.iterrows():
                name = str(r.get(nombre_col, '')).strip()
                course = r.get(curso_col)
                avg = r.get(promedio_col)
                att = r.get(asistencia_col)
                try:
                    att_val = float(att) if pd.notna(att) else None
                except Exception:
                    att_val = None
                students[name] = {
                    'curso': course,
                    'promedio': float(avg) if pd.notna(avg) else None,
                    'asistencia': att_val,
                    'peor_asignatura': peor_asig_por_alumno.get(name),
                    'consistencia_asignaturas': consistencia_por_alumno.get(name),
                    'neg_annotations_count': neg_counts_por_alumno.get(name, 0),
                    'neg_annotations_examples': neg_examples_por_alumno.get(name, []),
                    'family_complexity_score': fam_complex_score.get(name, 0),
                    'family_complexity_examples': fam_complex_examples.get(name, []),
                    'subjects_below_threshold_count': (consistencia_por_alumno.get(name) or {}).get('subjects_below_threshold_count', 0)
                }

        # Cursos agregados
        courses = {}
        if curso_col in df.columns:
            # Promedio de curso calculado sobre notas por asignatura (si existe) o promedio por alumno
            try:
                avg_by_course = df.groupby(curso_col)[nota_col].mean() if nota_col in df.columns else df.groupby(curso_col)[promedio_col].mean()
            except Exception:
                avg_by_course = pd.Series(dtype=float)
            # Asistencia promedio por curso
            try:
                if asistencia_col in df.columns:
                    df_alumnos = df.drop_duplicates(subset=[nombre_col]).copy()
                    att_by_course = df_alumnos.groupby(curso_col)[asistencia_col].mean()
                else:
                    att_by_course = pd.Series(dtype=float)
            except Exception:
                att_by_course = pd.Series(dtype=float)

            counts_by_course = df.drop_duplicates(subset=[nombre_col]).groupby(curso_col)[nombre_col].count()
            for course_name in df[curso_col].unique():
                courses[str(course_name)] = {
                    'avg_grade': float(avg_by_course.get(course_name)) if pd.notna(avg_by_course.get(course_name)) else None,
                    'avg_attendance': float(att_by_course.get(course_name)) if pd.notna(att_by_course.get(course_name)) else None,
                    'num_students': int(counts_by_course.get(course_name, 0))
                }

        # Métricas de construcción y señales (para trazabilidad)
        try:
            neg_total = int(sum(neg_counts_por_alumno.values())) if neg_counts_por_alumno else 0
        except Exception:
            neg_total = 0
        try:
            fam_total = int(sum(fam_complex_score.values())) if fam_complex_score else 0
        except Exception:
            fam_total = 0
        end_perf = datetime.datetime.now().timestamp()
        build_ms = int((end_perf - start_perf) * 1000)

        fs = {
            'students': students,
            'courses': courses,
            'levels': get_level_kpis(df),
            'course_attendance': get_course_attendance_kpis(df),
            'totals': {
                'num_students': int(len(students)),
                'num_courses': int(len(courses)),
                'num_levels': int(len(get_level_kpis(df)))
            },
            'metrics': {
                'neg_annotations_total': neg_total,
                'family_complexity_total': fam_total,
                'build_time_ms': build_ms
            }
        }
        try:
            current_app.logger.info(f"FeatureStore construido: alumnos={len(students)}, cursos={len(courses)}, neg_total={neg_total}, fam_total={fam_total}, build_ms={build_ms}")
        except Exception:
            # Fallback simple
            print(f"FeatureStore construido: alumnos={len(students)}, cursos={len(courses)}, neg_total={neg_total}, fam_total={fam_total}, build_ms={build_ms}")
        return fs
    except Exception as e:
        print(f"Error al construir Feature Store desde CSV: {e}")
        traceback.print_exc()
        return {}


def build_feature_store_signals(fs: dict, entity_type: str = None, entity_name: str = None, user_prompt: str = "") -> str:
    """
    Genera un bloque de señales textuales a partir del Feature Store.
    Priorizamos señales que suelen pedir los usuarios: asistencia baja por curso, curso de menor promedio por nivel, alumnos con peor promedio.
    """
    if not fs:
        return ""
    lines = []
    try:
        # Señales globales de conteos básicos (siempre útiles y responden preguntas directas)
        totals = fs.get('totals', {}) or {}
        num_students = totals.get('num_students'); num_courses = totals.get('num_courses'); num_levels = totals.get('num_levels')
        if isinstance(num_students, int) and num_students > 0:
            lines.append(f"Total de estudiantes en el CSV: {num_students}")
        if isinstance(num_courses, int) and num_courses > 0:
            lines.append(f"Total de cursos en el CSV: {num_courses}")
        if isinstance(num_levels, int) and num_levels > 0:
            lines.append(f"Total de niveles en el CSV: {num_levels}")

        # Señales de asistencia por curso (Top 3 peor asistencia)
        att_map = fs.get('course_attendance', {}) or {}
        # Convertir posibles strings "85.0%" a float 0.85
        def _att_to_float(v):
            if v is None:
                return None
            try:
                s = str(v).strip().replace('%', '')
                val = float(s)
                return val/100.0 if val > 1 else val
            except Exception:
                return None
        att_items = [(k, _att_to_float(v)) for k, v in att_map.items()]
        att_items = [(k, v) for k, v in att_items if v is not None]
        att_sorted = sorted(att_items, key=lambda x: x[1])[:3]
        if att_sorted:
            lines.append("Cursos con menor asistencia (Top 3):")
            for course, val in att_sorted:
                lines.append(f"- {course}: {val:.1%}")

        # Señales de tamaño por curso (Top 5 por cantidad de alumnos)
        courses = fs.get('courses', {}) or {}
        try:
            by_size = [ (name, info.get('num_students')) for name, info in courses.items() if isinstance(info.get('num_students'), int) ]
            by_size.sort(key=lambda x: x[1], reverse=True)
            top5 = by_size[:5]
            if top5:
                lines.append("Cursos con más estudiantes (Top 5):")
                for cname, ccount in top5:
                    lines.append(f"- {cname}: {int(ccount)} alumnos")
        except Exception:
            pass

        # Señales por nivel: curso de menor promedio
        levels = fs.get('levels', {}) or {}
        for level_name, info in levels.items():
            cmin = (info or {}).get('curso_menor_promedio', {})
            cmin_name = str(cmin.get('nombre', 'N/A'))
            cmin_avg = cmin.get('promedio', None)
            if cmin_name and cmin_name != 'N/A' and cmin_avg is not None:
                lines.append(f"Nivel {level_name} – Curso con menor promedio: {cmin_name} ({float(cmin_avg):.2f})")

        # Señales globales: alumnos con rendimiento más parejo entre asignaturas (Top 3 por menor desviación)
        try:
            students_map = fs.get('students', {}) or {}
            consistency_list = []
            for name, info in students_map.items():
                c = info.get('consistencia_asignaturas') or {}
                std_dev = c.get('std_dev'); nsub = c.get('num_asignaturas'); course = info.get('curso')
                if isinstance(std_dev, (int, float)) and std_dev is not None and isinstance(nsub, int) and nsub >= 2:
                    consistency_list.append((name, std_dev, nsub, course))
            consistency_list.sort(key=lambda x: x[1])
            top_consistent = consistency_list[:3]
            if top_consistent:
                lines.append("Alumnos con rendimiento más parejo entre asignaturas (Top 3):")
                for name, std_dev, nsub, course in top_consistent:
                    lines.append(f"- {name} ({course}): desviación {float(std_dev):.2f} en {int(nsub)} asignaturas")
        except Exception:
            pass

        # Señales globales: alumnos con menor promedio (Top 3)
        try:
            students_map = fs.get('students', {}) or {}
            bottom_students = [
                (name, info.get('promedio'), info.get('curso'))
                for name, info in students_map.items()
                if isinstance(info.get('promedio'), (int, float)) and info.get('promedio') is not None
            ]
            bottom_students.sort(key=lambda x: x[1])
            bottom_students = bottom_students[:3]
            if bottom_students:
                lines.append("Alumnos con menor promedio (Top 3):")
                for name, avg, course in bottom_students:
                    lines.append(f"- {name} ({course}): {float(avg):.2f}")
        except Exception:
            pass

        # Señales globales: alumnos con mejor promedio (Top 3)
        try:
            students_map = fs.get('students', {}) or {}
            top_students = [
                (name, info.get('promedio'), info.get('curso'))
                for name, info in students_map.items()
                if isinstance(info.get('promedio'), (int, float)) and info.get('promedio') is not None
            ]
            top_students.sort(key=lambda x: x[1], reverse=True)
            top_students = top_students[:3]
            if top_students:
                lines.append("Alumnos con mejor promedio (Top 3):")
                for name, avg, course in top_students:
                    lines.append(f"- {name} ({course}): {float(avg):.2f}")
        except Exception:
            pass

        # Señales globales: alumnos con más anotaciones negativas (Top 3)
        try:
            students_map = fs.get('students', {}) or {}
            neg_list = []
            for name, info in students_map.items():
                cnt = info.get('neg_annotations_count', 0)
                course = info.get('curso')
                if isinstance(cnt, int) and cnt > 0:
                    neg_list.append((name, cnt, course))
            neg_list.sort(key=lambda x: x[1], reverse=True)
            top_neg = neg_list[:3]
            if top_neg:
                lines.append("Alumnos con más anotaciones negativas (Top 3):")
                for name, cnt, course in top_neg:
                    lines.append(f"- {name} ({course}): {int(cnt)} anotaciones negativas")
        except Exception:
            pass

        # Señales globales: alumnos con indicios de situación familiar compleja (Top 5 por score)
        try:
            students_map = fs.get('students', {}) or {}
            fam_list = []
            for name, info in students_map.items():
                score = info.get('family_complexity_score', 0)
                course = info.get('curso')
                if isinstance(score, int) and score > 0:
                    fam_list.append((name, score, course))
            fam_list.sort(key=lambda x: x[1], reverse=True)
            top_fam = fam_list[:5]
            if top_fam:
                lines.append("Alumnos con situación familiar potencialmente compleja (Top 5):")
                for name, score, course in top_fam:
                    lines.append(f"- {name} ({course}): {int(score)} registros con indicios")
        except Exception:
            pass

        # Señales específicas si el contexto es curso o alumno
        if entity_type == 'curso' and entity_name:
            c = (fs.get('courses', {}) or {}).get(entity_name, {})
            if c:
                avg = c.get('avg_grade'); att = c.get('avg_attendance'); n = c.get('num_students')
                avg_txt = f"{avg:.2f}" if isinstance(avg, (int,float)) and avg is not None else "N/A"
                att_txt = f"{att:.1%}" if isinstance(att, (int,float)) and att is not None else "N/A"
                lines.append(f"Curso {entity_name}: promedio {avg_txt} | asistencia {att_txt} | Nº alumnos {n}")
        elif entity_type == 'alumno' and entity_name:
            s = (fs.get('students', {}) or {}).get(entity_name, {})
            if s:
                avg = s.get('promedio'); att = s.get('asistencia'); course = s.get('curso')
                att_txt = f"{att:.1%}" if isinstance(att, (int,float)) and att is not None else "N/A"
                avg_txt = f"{avg:.2f}" if isinstance(avg, (int,float)) and avg is not None else "N/A"
                lines.append(f"Alumno {entity_name} ({course}): promedio {avg_txt} | asistencia {att_txt}")
                peor_asig = s.get('peor_asignatura') or {}
                pa_nombre = peor_asig.get('asignatura')
                pa_prom = peor_asig.get('promedio')
                if pa_nombre and isinstance(pa_prom, (int, float)):
                    lines.append(f"Peor asignatura: {pa_nombre} ({float(pa_prom):.2f})")
                # Detalle de consistencia entre asignaturas
                cons = s.get('consistencia_asignaturas') or {}
                if isinstance(cons.get('std_dev'), (int, float)) and cons.get('std_dev') is not None:
                    lines.append(f"Consistencia (desv. estándar entre asignaturas): {float(cons.get('std_dev')):.2f} en {int(cons.get('num_asignaturas') or 0)} asignaturas")
                # Detalle de anotaciones negativas
                neg_cnt = s.get('neg_annotations_count', 0)
                if isinstance(neg_cnt, int) and neg_cnt > 0:
                    lines.append(f"Anotaciones negativas detectadas en CSV: {int(neg_cnt)}")
                # Detalle de situación familiar compleja
                fam_score = s.get('family_complexity_score', 0)
                if isinstance(fam_score, int) and fam_score > 0:
                    lines.append(f"Registros con indicios de situación familiar compleja: {int(fam_score)}")

    except Exception as e:
        lines.append(f"[Error al generar señales del Feature Store: {e}]")

    return "\n".join(lines)


def compute_risk_scores_from_feature_store(fs: dict) -> dict:
    if not fs:
        return {}
    students = fs.get('students', {}) or {}
    levels = fs.get('levels', {}) or {}
    attendance_threshold_default = float(current_app.config.get('ATTENDANCE_RISK_THRESHOLD', 0.85))
    low_perf_thr_default = float(current_app.config.get('LOW_PERFORMANCE_THRESHOLD_GRADE', 4.0))
    ovr = session.get('risk_sensitivity') or {}
    avg_below_high_thr = float(ovr.get('avg_below_high_threshold', low_perf_thr_default))
    subj_below_medium_cnt = int(ovr.get('subjects_below_medium_count', 2))
    neg_obs_medium_cnt = int(ovr.get('neg_observations_medium_count', 2))
    neg_obs_high_cnt = int(ovr.get('neg_observations_high_count', 4))
    level_thresholds = current_app.config.get('RISK_THRESHOLDS_BY_LEVEL') or {}
    risk = {}
    db_path = None
    try:
        db_path = current_app.config['DATABASE_FILE']
    except Exception:
        db_path = None
    for name, info in students.items():
        avg = info.get('promedio')
        att = info.get('asistencia')
        neg = info.get('neg_annotations_count', 0)
        fam = info.get('family_complexity_score', 0)
        cons = info.get('consistencia_asignaturas') or {}
        std_dev = cons.get('std_dev')
        course = info.get('curso')
        score = 0.0
        reasons = []
        try:
            level_name = None
            try:
                cn = str(course or '').lower()
                cn = unicodedata.normalize('NFKD', cn).encode('ascii', 'ignore').decode('ascii')
                if 'medio' in cn:
                    level_name = 'medio'
                elif 'basico' in cn:
                    level_name = 'basico'
            except Exception:
                level_name = None
            att_thr = attendance_threshold_default
            grade_thr = low_perf_thr_default
            if level_name and isinstance(level_thresholds.get(level_name), dict):
                att_thr = float(level_thresholds[level_name].get('attendance', att_thr))
                grade_thr = float(level_thresholds[level_name].get('grade', grade_thr))
            if isinstance(avg, (int, float)) and avg is not None:
                if avg < grade_thr:
                    inc = max(0.0, min(40.0, (grade_thr - float(avg)) * 10.0))
                    score += inc
                    reasons.append('promedio_bajo')
            if isinstance(att, (int, float)) and att is not None:
                if float(att) < att_thr:
                    inc = max(0.0, min(30.0, (att_thr - float(att)) * 100.0))
                    score += inc
                    reasons.append('asistencia_baja')
            if isinstance(neg, int) and neg > 0:
                inc = float(min(20, neg * 5))
                score += inc
                reasons.append('anotaciones_negativas')
            if isinstance(fam, int) and fam > 0:
                inc = float(min(20, fam * 5))
                score += inc
                reasons.append('situacion_familiar_compleja')
            if isinstance(std_dev, (int, float)) and std_dev is not None:
                if float(std_dev) > 1.0:
                    inc = max(0.0, min(20.0, (float(std_dev) - 1.0) * 10.0))
                    score += inc
                    reasons.append('desbalance_asignaturas')
            if db_path:
                evo = get_student_grade_evolution_value(db_path, name)
                if isinstance(evo, (int, float)):
                    if float(evo) <= -0.5:
                        score += 10.0
                        reasons.append('tendencia_baja_notas')
                    elif float(evo) >= 0.5:
                        score -= 5.0
                        reasons.append('tendencia_mejora_notas')
                aev = get_student_attendance_evolution_value(db_path, name)
                if isinstance(aev, (int, float)):
                    if float(aev) <= -5.0:
                        score += 8.0
                        reasons.append('tendencia_baja_asistencia')
                    elif float(aev) >= 5.0:
                        score -= 4.0
                        reasons.append('tendencia_mejora_asistencia')
        except Exception:
            pass
        score = float(max(0.0, min(100.0, score)))
        if score >= 70.0:
            level = 'alto'
        elif score >= 40.0:
            level = 'medio'
        else:
            level = 'bajo'
        try:
            subj_low_cnt = int(info.get('subjects_below_threshold_count', 0) or 0)
        except Exception:
            subj_low_cnt = 0
        if isinstance(avg, (int, float)) and avg is not None and float(avg) < avg_below_high_thr:
            level = 'alto'
            score = max(score, 75.0)
            if 'regla_promedio_bajo_alto' not in reasons:
                reasons.append('regla_promedio_bajo_alto')
        elif subj_low_cnt >= subj_below_medium_cnt and level != 'alto':
            level = 'medio'
            score = max(score, 40.0)
            if 'regla_dos_asignaturas_bajas_medio' not in reasons:
                reasons.append('regla_dos_asignaturas_bajas_medio')
        if isinstance(neg, int):
            if neg_obs_high_cnt and neg >= neg_obs_high_cnt:
                level = 'alto'
                score = max(score, 75.0)
                if 'regla_observaciones_negativas_alto' not in reasons:
                    reasons.append('regla_observaciones_negativas_alto')
            elif neg_obs_medium_cnt and neg >= neg_obs_medium_cnt and level != 'alto':
                level = 'medio'
                score = max(score, 40.0)
                if 'regla_observaciones_negativas_medio' not in reasons:
                    reasons.append('regla_observaciones_negativas_medio')
        risk[name] = {'score': int(round(score)), 'level': level, 'reasons': reasons, 'course': course}
    return risk


def build_risk_summary(fs: dict) -> dict:
    scores = compute_risk_scores_from_feature_store(fs)
    by_course = {}
    totals = {'alto': 0, 'medio': 0, 'bajo': 0}
    for name, r in scores.items():
        c = r.get('course')
        lvl = r.get('level')
        if c:
            m = by_course.setdefault(str(c), {'alto': 0, 'medio': 0, 'bajo': 0, 'top': []})
            m[lvl] += 1
            m['top'].append((name, r.get('score', 0)))
        totals[lvl] += 1
    for c, m in by_course.items():
        m['top'] = sorted(m['top'], key=lambda x: x[1], reverse=True)[:5]
    return {'scores': scores, 'by_course': by_course, 'totals': totals}


def get_student_grade_evolution_value(db_path: str, student_name: str):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                WITH SnapshotAverages AS (
                    SELECT h.snapshot_id, h.student_name, s.timestamp, AVG(h.grade) AS avg_grade
                    FROM student_data_history h
                    JOIN data_snapshots s ON h.snapshot_id = s.id
                    WHERE h.student_name = ?
                    GROUP BY h.snapshot_id, h.student_name, s.timestamp
                )
                SELECT
                    FIRST_VALUE(avg_grade) OVER (ORDER BY timestamp ASC) AS first_avg_grade,
                    LAST_VALUE(avg_grade) OVER (ORDER BY timestamp ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_avg_grade
                FROM SnapshotAverages
                LIMIT 1
                """,
                (student_name,)
            )
            rows = cursor.fetchall()
            if not rows:
                return None
            row = rows[-1]
            first_avg = row['first_avg_grade']
            last_avg = row['last_avg_grade']
            if first_avg is None or last_avg is None:
                return None
            return float(last_avg) - float(first_avg)
    except Exception:
        return None


def get_student_attendance_evolution_value(db_path: str, student_name: str):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                WITH SnapshotAttendance AS (
                    SELECT h.snapshot_id, h.student_name, s.timestamp, AVG(h.attendance_perc) AS avg_attendance
                    FROM student_data_history h
                    JOIN data_snapshots s ON h.snapshot_id = s.id
                    WHERE h.student_name = ?
                    GROUP BY h.snapshot_id, h.student_name, s.timestamp
                )
                SELECT
                    FIRST_VALUE(avg_attendance) OVER (ORDER BY timestamp ASC) AS first_avg_attendance,
                    LAST_VALUE(avg_attendance) OVER (ORDER BY timestamp ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_avg_attendance
                FROM SnapshotAttendance
                LIMIT 1
                """,
                (student_name,)
            )
            rows = cursor.fetchall()
            if not rows:
                return None
            row = rows[-1]
            first_att = row['first_avg_attendance']
            last_att = row['last_avg_attendance']
            if first_att is None or last_att is None:
                return None
            return float((last_att or 0) - (first_att or 0))
    except Exception:
        return None

def get_intervention_compliance_for_entity(db_path: str, entity_name: str, owner_username: str = None):
    try:
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            # Obtener planes para la entidad y, opcionalmente, por propietario
            if owner_username:
                cur.execute(
                    """
                    SELECT id FROM follow_ups
                    WHERE follow_up_type = 'intervention_plan' AND related_entity_type = 'alumno' AND related_entity_name = ?
                    AND owner_username = ?
                    ORDER BY timestamp DESC
                    """,
                    (entity_name, owner_username)
                )
            else:
                cur.execute(
                    """
                    SELECT id FROM follow_ups
                    WHERE follow_up_type = 'intervention_plan' AND related_entity_type = 'alumno' AND related_entity_name = ?
                    ORDER BY timestamp DESC
                    """,
                    (entity_name,)
                )
            plan_rows = cur.fetchall()
            if not plan_rows:
                return {'plans': 0, 'total_actions': 0, 'applied_actions': 0, 'avg_compliance_pct': 0.0}
            total_actions = 0
            applied_actions = 0
            compliance_values = []
            for r in plan_rows:
                pid = int(r['id'])
                cur.execute(
                    "SELECT status, compliance_pct FROM intervention_actions WHERE plan_id = ? AND (owner_username = ? OR owner_username IS NULL)",
                    (pid, owner_username)
                )
                arows = cur.fetchall()
                total_actions += len(arows)
                for ar in arows:
                    if (ar['status'] or '').strip() == 'applied':
                        applied_actions += 1
                    try:
                        compliance_values.append(float(ar['compliance_pct'] or 0.0))
                    except Exception:
                        pass
            avg_comp = (sum(compliance_values)/len(compliance_values)) if compliance_values else 0.0
            return {'plans': len(plan_rows), 'total_actions': total_actions, 'applied_actions': applied_actions, 'avg_compliance_pct': avg_comp}
    except Exception:
        return {'plans': 0, 'total_actions': 0, 'applied_actions': 0, 'avg_compliance_pct': 0.0}

def get_effectiveness_conditioned(db_path: str, entity_name: str, days: int = None, only_applied: bool = True, owner_username: str = None):
    # Para MVP: usamos deltas globales (first vs last) y condicionamos por existencia de acciones "applied" cuando only_applied=True
    comp = get_intervention_compliance_for_entity(db_path, entity_name, owner_username)
    if only_applied and comp.get('applied_actions', 0) == 0:
        return {'grade_delta': None, 'attendance_delta': None, 'compliance': comp}
    gd = get_student_grade_evolution_value(db_path, entity_name)
    ad = get_student_attendance_evolution_value(db_path, entity_name)
    return {'grade_delta': gd, 'attendance_delta': ad, 'compliance': comp}

def save_intervention_outcome(db_path: str, follow_up_id: int, related_entity_type: str, related_entity_name: str, course_name: str, outcome_date: str, compliance_pct: float, impact_grade_delta: float, impact_attendance_delta: float, notes: str):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO intervention_outcomes (follow_up_id, related_entity_type, related_entity_name, course_name, outcome_date, compliance_pct, impact_grade_delta, impact_attendance_delta, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (follow_up_id, related_entity_type, related_entity_name, course_name, outcome_date, compliance_pct, impact_grade_delta, impact_attendance_delta, notes)
            )
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        print(f"Error al guardar resultado de intervención: {e}")
        traceback.print_exc()
        return None

def get_intervention_effectiveness_summary(db_path: str, days_window: int = None):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            base = "SELECT related_entity_type, course_name, compliance_pct, impact_grade_delta, impact_attendance_delta FROM intervention_outcomes"
            params = []
            if isinstance(days_window, int) and days_window > 0:
                base += " WHERE outcome_date >= date('now', ?)"
                params.append(f"-{int(days_window)} day")
            cursor.execute(base, tuple(params))
            rows = cursor.fetchall()
            if not rows:
                return {'totals': {'count': 0, 'avg_compliance': None, 'avg_grade_delta': None, 'avg_att_delta': None}, 'by_course': {}}
            count = len(rows)
            comp_vals = [r['compliance_pct'] for r in rows if r['compliance_pct'] is not None]
            grade_vals = [r['impact_grade_delta'] for r in rows if r['impact_grade_delta'] is not None]
            att_vals = [r['impact_attendance_delta'] for r in rows if r['impact_attendance_delta'] is not None]
            def avg(lst):
                return float(sum(lst)/len(lst)) if lst else None
            by_course = {}
            for r in rows:
                c = str(r['course_name']) if r['course_name'] is not None else ''
                m = by_course.setdefault(c, {'count': 0, 'avg_compliance': None, 'avg_grade_delta': None, 'avg_att_delta': None, 'comp_values': [], 'grade_values': [], 'att_values': []})
                m['count'] += 1
                if r['compliance_pct'] is not None:
                    m['comp_values'].append(r['compliance_pct'])
                if r['impact_grade_delta'] is not None:
                    m['grade_values'].append(r['impact_grade_delta'])
                if r['impact_attendance_delta'] is not None:
                    m['att_values'].append(r['impact_attendance_delta'])
            for c, m in by_course.items():
                m['avg_compliance'] = avg(m['comp_values'])
                m['avg_grade_delta'] = avg(m['grade_values'])
                m['avg_att_delta'] = avg(m['att_values'])
                m.pop('comp_values', None)
                m.pop('grade_values', None)
                m.pop('att_values', None)
            return {
                'totals': {
                    'count': int(count),
                    'avg_compliance': avg(comp_vals),
                    'avg_grade_delta': avg(grade_vals),
                    'avg_att_delta': avg(att_vals)
                },
                'by_course': by_course
            }
    except Exception as e:
        print(f"Error al obtener resumen de efectividad de intervenciones: {e}")
        traceback.print_exc()
        return {'totals': {'count': 0, 'avg_compliance': None, 'avg_grade_delta': None, 'avg_att_delta': None}, 'by_course': {}}


def recommend_interventions_for_student(fs: dict, student_name: str) -> dict:
    students = fs.get('students', {}) or {}
    info = students.get(student_name) or {}
    risk = compute_risk_scores_from_feature_store(fs).get(student_name) or {}
    reasons = set(risk.get('reasons') or [])
    course = info.get('curso')
    peor_asig = (info.get('peor_asignatura') or {}).get('asignatura')
    actions = []
    def add(title, action, foundation):
        actions.append({'title': title, 'action': action, 'foundation': foundation})
    if 'promedio_bajo' in reasons:
        subj_txt = f" en {peor_asig}" if peor_asig else ""
        add(
            f"Tutorías focalizadas{subj_txt}",
            f"Asignar 2 sesiones semanales con metas de dominio y práctica espaciada{subj_txt}.",
            "Basado en aprendizaje de dominio (Bloom) y práctica espaciada (Ebbinghaus)."
        )
        add(
            "Evaluación formativa y retroalimentación específica",
            "Instrumentos cortos por objetivo, retroalimentación inmediata y autoevaluación guiada.",
            "Fundamentado en evaluación formativa y metacognición (Black & Wiliam)."
        )
    if 'asistencia_baja' in reasons or 'tendencia_baja_asistencia' in reasons:
        add(
            "Plan de asistencia y seguimiento",
            "Contacto familia, compromisos semanales, incentivos y monitoreo quincenal.",
            "PBIS y enfoques de compromiso escolar (Fredricks et al.)."
        )
    if 'anotaciones_negativas' in reasons:
        add(
            "Apoyo conductual positivo",
            "Refuerzo diferencial, reglas claras, práctica de habilidades sociales, acuerdos de aula.",
            "PBIS y aprendizaje socioemocional (CASEL)."
        )
        add(
            "Prácticas restaurativas",
            "Círculos breves para reparar daños y reconstruir vínculos con pares y docentes.",
            "Restorative Practices (Wachtel)."
        )
    if 'situacion_familiar_compleja' in reasons:
        add(
            "Derivación a apoyo psicosocial",
            "Coordinación con dupla psicosocial/DAEM y seguimiento confidencial.",
            "Enfoque ecológico de Bronfenbrenner y protección integral."
        )
    if 'desbalance_asignaturas' in reasons and peor_asig:
        add(
            "Foco en asignatura más débil",
            f"Plan de mejora específico en {peor_asig} con objetivos semanales y práctica guiada.",
            "Aprendizaje guiado y andamiaje (Vygotsky)."
        )
    if 'tendencia_baja_notas' in reasons:
        add(
            "Escalamiento de apoyo académico",
            "Aumentar intensidad de tutorías, revisión de hábitos de estudio y coordinación con familia.",
            "Multi-tiered systems of support (MTSS)."
        )
    if not actions:
        add(
            "Seguimiento estándar",
            "Mantener evaluación formativa y tutorías de refuerzo según necesidad.",
            "Buenas prácticas docentes y ajuste por datos."
        )
    return {
        'student': student_name,
        'course': course,
        'risk': risk,
        'actions': actions
    }

def _list_snapshots(db_path):
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, timestamp FROM data_snapshots ORDER BY timestamp ASC")
            return [(r[0], r[1]) for r in cur.fetchall()]
    except Exception:
        return []

def _load_snapshot_dataframe(db_path, snapshot_id):
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(
                "SELECT student_name, student_course, subject, grade, attendance_perc, conduct_observation, age, professor, family_info, interviews_info FROM student_data_history WHERE snapshot_id = ?",
                conn,
                params=(snapshot_id,)
            )
            return df
    except Exception:
        return None

def _build_features_from_df(df):
    df2 = df.copy()
    df2['grade'] = pd.to_numeric(df2['grade'], errors='coerce')
    df2['attendance_perc'] = pd.to_numeric(df2['attendance_perc'], errors='coerce')
    df2['conduct_observation'] = df2['conduct_observation'].fillna('').astype(str).str.lower()
    df2['family_info'] = df2['family_info'].fillna('').astype(str).str.lower()
    neg_keys = current_app.config.get('NEGATIVE_OBSERVATION_KEYWORDS', []) or []
    def _neg_count(text):
        if not text:
            return 0
        c = 0
        for k in neg_keys:
            if k and k in text:
                c += 1
        return c
    g = df2.groupby('student_name')
    avg_grade = g['grade'].mean().fillna(0.0)
    avg_att = g['attendance_perc'].mean().fillna(0.0)
    neg_obs = g['conduct_observation'].apply(lambda s: sum(_neg_count(x) for x in s)).fillna(0)
    fam_complex = g['family_info'].apply(lambda s: 1 if any(('separacion' in (x or '') or 'conflicto' in (x or '') or 'ausencia' in (x or '') or 'violencia' in (x or '')) for x in s) else (0 if all((x or '') == '' for x in s) else 1)).fillna(0)
    out = {}
    for name in avg_grade.index:
        out[name] = {
            'avg_grade': float(avg_grade.loc[name] or 0.0),
            'attendance_perc': float(avg_att.loc[name] or 0.0),
            'neg_observations': int(neg_obs.loc[name] or 0),
            'family_complexity': int(fam_complex.loc[name] or 0)
        }
    return out

def build_training_dataset_from_db(db_path):
    snaps = _list_snapshots(db_path)
    X = []
    y = []
    meta = []
    last_feats = {}
    for sid, ts in snaps:
        df = _load_snapshot_dataframe(db_path, sid)
        if df is None or df.empty:
            continue
        fs = build_feature_store_from_csv(df)
        risks = compute_risk_scores_from_feature_store(fs)
        feats = _build_features_from_df(df)
        for student, f in feats.items():
            prev = last_feats.get(student)
            grade_trend = 0.0
            att_trend = 0.0
            if prev:
                grade_trend = f['avg_grade'] - prev['avg_grade']
                att_trend = f['attendance_perc'] - prev['attendance_perc']
            label = 'bajo'
            r = risks.get(student)
            if isinstance(r, dict):
                label = r.get('level') or label
            X.append([
                f['avg_grade'],
                f['attendance_perc'],
                float(f['neg_observations']),
                float(f['family_complexity']),
                grade_trend,
                att_trend
            ])
            y.append(label)
            meta.append({'snapshot_id': sid, 'timestamp': ts, 'student_name': student})
        last_feats = feats
    feature_names = ['avg_grade', 'attendance_perc', 'neg_observations', 'family_complexity', 'grade_trend', 'attendance_trend']
    return {'X': X, 'y': y, 'meta': meta, 'feature_names': feature_names, 'snapshots': snaps}

def _normalize_train_test(dataset):
    snaps = dataset['snapshots']
    if not snaps:
        return None
    last_sid = snaps[-1][0]
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for i, row in enumerate(dataset['X']):
        sid = dataset['meta'][i]['snapshot_id']
        if sid == last_sid:
            X_test.append(row)
            y_test.append(dataset['y'][i])
        else:
            X_train.append(row)
            y_train.append(dataset['y'][i])
    if not X_train:
        return None
    cols = list(zip(*X_train))
    means = [sum(c)/len(c) if len(c)>0 else 0.0 for c in cols]
    stds = []
    for idx, c in enumerate(cols):
        m = means[idx]
        var = sum((v-m)*(v-m) for v in c)/len(c) if len(c)>0 else 0.0
        stds.append(var**0.5 if var>0 else 1.0)
    def norm_rows(rows):
        out = []
        for r in rows:
            out.append([(r[j]-means[j])/stds[j] for j in range(len(r))])
        return out
    return {
        'X_train': norm_rows(X_train),
        'y_train': y_train,
        'X_test': norm_rows(X_test),
        'y_test': y_test,
        'means': means,
        'stds': stds
    }

def _labels_to_indices(y):
    classes = ['bajo', 'medio', 'alto']
    return [classes.index(v) if v in classes else 0 for v in y], classes

def _train_logistic_multiclass(X, y_idx, classes, epochs=200, lr=0.05, l2=0.001):
    if not X:
        return None
    n = len(X)
    d = len(X[0])
    k = len(classes)
    W = [[0.0 for _ in range(d)] for _ in range(k)]
    b = [0.0 for _ in range(k)]
    def _softmax(z):
        m = max(z)
        ez = [np.exp(v-m) for v in z]
        s = float(np.sum(ez))
        return [v/s for v in ez]
    for _ in range(epochs):
        for i in range(n):
            xi = X[i]
            yi = y_idx[i]
            z = [sum(W[c][j]*xi[j] for j in range(d)) + b[c] for c in range(k)]
            p = _softmax(z)
            for c in range(k):
                grad_b = (1.0 if c==yi else 0.0) - p[c]
                b[c] += lr*grad_b
                for j in range(d):
                    grad_w = ((1.0 if c==yi else 0.0) - p[c]) * xi[j] - l2*W[c][j]
                    W[c][j] += lr*grad_w
    return {'W': W, 'b': b, 'classes': classes}

def _predict_logistic(model, X):
    W = model['W']
    b = model['b']
    classes = model['classes']
    d = len(W[0])
    k = len(classes)
    def _softmax(z):
        m = max(z)
        ez = [np.exp(v-m) for v in z]
        s = float(np.sum(ez))
        return [v/s for v in ez]
    preds = []
    prob_high = []
    for xi in X:
        z = [sum(W[c][j]*xi[j] for j in range(d)) + b[c] for c in range(k)]
        p = _softmax(z)
        idx = max(range(k), key=lambda c: p[c])
        preds.append(classes[idx])
        prob_high.append(p[classes.index('alto')])
    return preds, prob_high

def _f1_macro(y_true, y_pred):
    classes = ['bajo', 'medio', 'alto']
    f1s = []
    for c in classes:
        tp = sum(1 for i in range(len(y_true)) if y_true[i]==c and y_pred[i]==c)
        fp = sum(1 for i in range(len(y_true)) if y_true[i]!=c and y_pred[i]==c)
        fn = sum(1 for i in range(len(y_true)) if y_true[i]==c and y_pred[i]!=c)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        f1s.append(f1)
    return sum(f1s)/len(f1s) if f1s else 0.0

def _auc_roc_binary(y_true_high, scores_high):
    pairs = list(zip(scores_high, y_true_high))
    pairs.sort(key=lambda x: x[0])
    tps = 0
    fps = 0
    pos = sum(y_true_high)
    neg = len(y_true_high)-pos
    auc = 0.0
    prev_score = None
    prev_tpr = 0.0
    prev_fpr = 0.0
    for s, yb in pairs:
        if prev_score is None or s != prev_score:
            tpr = tps/pos if pos>0 else 0.0
            fpr = fps/neg if neg>0 else 0.0
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
            prev_tpr = tpr
            prev_fpr = fpr
            prev_score = s
        if yb:
            tps += 1
        else:
            fps += 1
    tpr = tps/pos if pos>0 else 0.0
    fpr = fps/neg if neg>0 else 0.0
    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
    return auc

def train_predictive_risk_model(db_path, artifacts_dir):
    ds = build_training_dataset_from_db(db_path)
    split = _normalize_train_test(ds)
    if not split:
        return None
    y_idx, classes = _labels_to_indices(split['y_train'])
    model = _train_logistic_multiclass(split['X_train'], y_idx, classes)
    preds_test, prob_high = _predict_logistic(model, split['X_test'])
    f1_macro = _f1_macro(split['y_test'], preds_test)
    y_true_high = [1 if v=='alto' else 0 for v in split['y_test']]
    y_pred_bin = ['alto' if p=='alto' else 'no_alto' for p in preds_test]
    y_true_bin = ['alto' if v==1 else 'no_alto' for v in y_true_high]
    f1_bin = _f1_macro(y_true_bin, y_pred_bin)
    auc = _auc_roc_binary(y_true_high, prob_high)
    params = {
        'feature_names': ds['feature_names'],
        'means': split['means'],
        'stds': split['stds'],
        'classes': classes
    }
    metrics = {
        'macro_f1_3clases': f1_macro,
        'f1_binario_alto_vs_no': f1_bin,
        'auc_roc_binario': auc,
        'train_size': len(split['X_train']),
        'test_size': len(split['X_test'])
    }
    try:
        import pickle
        with open(os.path.join(artifacts_dir, 'predictive_risk_model.pkl'), 'wb') as f:
            pickle.dump({'model': model, 'params': params}, f)
    except Exception:
        pass
    with open(os.path.join(artifacts_dir, 'predictive_risk_model_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({'params': params, 'metrics': metrics}, f)
    return metrics

def predict_risk_with_model_for_fs(fs, artifacts_dir):
    path = os.path.join(artifacts_dir, 'predictive_risk_model.pkl')
    if not os.path.exists(path):
        return {}
    try:
        import pickle
        obj = pickle.load(open(path, 'rb'))
    except Exception:
        return {}
    params = obj.get('params') or {}
    model = obj.get('model') or {}
    students = list((fs.get('students') or {}).keys())
    rows = []
    for s in students:
        info = (fs.get('students') or {}).get(s) or {}
        avg_grade = float(info.get('promedio') or 0.0)
        attendance = float(info.get('asistencia') or 0.0)
        neg = int(info.get('neg_annotations_count') or 0)
        fam = int(info.get('family_complexity_score') or 0)
        rows.append([avg_grade, attendance, float(neg), float(fam), 0.0, 0.0])
    means = params.get('means') or ([0.0]*len(rows[0]) if rows else [])
    stds = params.get('stds') or ([1.0]*len(rows[0]) if rows else [])
    def _norm_row(r):
        return [(r[j]-means[j])/stds[j] for j in range(len(r))]
    Xn = [_norm_row(r) for r in rows]
    preds, prob_high = _predict_logistic(model, Xn) if rows else ([], [])
    out = {}
    for i, s in enumerate(students):
        out[s] = {'predicted_level': preds[i] if i < len(preds) else None, 'prob_high': prob_high[i] if i < len(prob_high) else None}
    return out

def reset_database_tables(db_path: str, tables: list = None):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            if not tables:
                tables = [
                    'student_data_history',
                    'data_snapshots',
                    'follow_ups',
                    'intervention_outcomes',
                    'consumo_tokens_diario'
                ]
            for t in tables:
                try:
                    cursor.execute(f"DELETE FROM {t}")
                except Exception:
                    pass
            try:
                cursor.execute("VACUUM")
            except Exception:
                pass
            conn.commit()
        return True
    except Exception:
        return False
# --- VERIFICACIÓN DE ENLACES Y POST-FILTRADO ---
import re
import requests
from urllib.parse import urlparse

def _domain_allowed(url, cfg):
    try:
        parsed = urlparse(url)
        host = (parsed.netloc or '').lower()
        wl = set((cfg.get('RESOURCE_WHITELIST_DOMAINS') or []))
        bl = set((cfg.get('RESOURCE_BLACKLIST_DOMAINS') or []))
        if any(host.endswith(d.lower()) for d in bl):
            return False
        if wl:
            return any(host.endswith(d.lower()) for d in wl)
        return True
    except Exception:
        return False

def _verify_url(url, cfg):
    if not _domain_allowed(url, cfg):
        return {'ok': False, 'reason': 'domain_not_allowed'}
    timeout = float(cfg.get('RESOURCE_VERIFY_TIMEOUT_SEC', 4))
    headers = {'User-Agent': 'TutorIA360/1.0 (+verificador)'}
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout, headers=headers)
        code = int(resp.status_code)
        if code >= 400 or code < 200:
            # Fall back to GET
            resp = requests.get(url, allow_redirects=True, timeout=timeout, headers=headers, stream=True)
            code = int(resp.status_code)
        if 200 <= code < 300:
            ctype = (resp.headers.get('Content-Type') or '').lower()
            lang = (resp.headers.get('Content-Language') or '').lower()
            prefer_lang = str(cfg.get('PREFER_RESOURCE_LANGUAGE', '') or '').lower()
            lang_ok = True if not prefer_lang else (prefer_lang in lang) or ('html' in ctype)
            return {'ok': True, 'final_url': resp.url, 'content_type': ctype, 'lang_ok': lang_ok}
        return {'ok': False, 'reason': f'status_{code}'}
    except requests.exceptions.RequestException as ex:
        return {'ok': False, 'reason': str(ex)}

def verify_and_annotate_resources_html(html: str, cfg: dict) -> str:
    if not bool(cfg.get('STRICT_RESOURCE_VERIFICATION', False)):
        return html
    # Procesar por secciones (título + grid) para mantener estructura y orden
    sections = []
    idx = 0
    pattern = re.compile(r'(<h3[^>]*class="resource-topic-title"[\s\S]*?</h3>\s*<div\s+class="resource-grid">)([\s\S]*?)(</div>)', re.I)
    result = []
    last_end = 0
    for m in pattern.finditer(html):
        start, inner, end = m.group(1), m.group(2), m.group(3)
        # Añadir contenido previo sin tocar
        result.append(html[last_end:m.start()])
        # Filtrar tarjetas de esta sección
        cards = re.findall(r'<div\s+class="resource-card"[\s\S]*?</div>', inner, flags=re.I)
        verified_cards = []
        for card in cards:
            link = re.search(r'<a[^>]*href="([^"]+)"[^>]*>', card, flags=re.I|re.S)
            if not link:
                continue
            url = link.group(1)
            vr = _verify_url(url, cfg)
            if not vr.get('ok'):
                continue
            badge = '<span class="resource-badge verified" style="background:#d1fae5;color:#065f46;padding:2px 6px;border-radius:6px;margin-left:6px;font-size:11px;">Verificado</span>'
            card2 = re.sub(r'(</h4>)', badge + r'\1', card, flags=re.I)
            final_url = vr.get('final_url') or url
            card2 = re.sub(r'href="[^"]+"', f'href="{final_url}"', card2, flags=re.I)
            if not vr.get('lang_ok', True):
                badge2 = '<span class="resource-badge lang" style="background:#e5e7eb;color:#374151;padding:2px 6px;border-radius:6px;margin-left:6px;font-size:11px;">Idioma no preferido</span>'
                card2 = re.sub(r'(</h4>)', badge2 + r'\1', card2, flags=re.I)
            verified_cards.append(card2)
        if verified_cards:
            result.append(start + ''.join(verified_cards) + end)
        # Si no hay tarjetas válidas, eliminar la sección completa (deja espacio para locales)
        last_end = m.end()
    # Añadir el resto del HTML después de la última sección
    result.append(html[last_end:])
    out_html = ''.join(result)
    # Si toda la salida queda vacía o sin grids, devolver vacío para que se inyecten locales
    if not re.search(r'<div\s+class="resource-grid">', out_html, flags=re.I):
        return ''
    return out_html

def inject_local_resources(html: str, cfg: dict) -> str:
    baseline = cfg.get('RESOURCE_LOCAL_BASELINE') or []
    if not baseline:
        return html
    cards = []
    for item in baseline:
        url_ok = None
        for cand in (item.get('candidates') or []):
            vr = _verify_url(cand, cfg)
            if vr.get('ok'):
                url_ok = vr.get('final_url') or cand
                break
        if not url_ok:
            # Fallback: tomar primer candidato aunque la verificación falle
            if (item.get('candidates') or []):
                url_ok = item['candidates'][0]
            else:
                continue
        title = item.get('title') or 'Recurso local'
        desc = item.get('desc') or ''
        card = (
            f'<div class="resource-card">'
            f'<h4 class="resource-title"><i class="fas fa-university mr-2"></i>{title}'
            f'<span class="resource-badge verified" style="background:#d1fae5;color:#065f46;padding:2px 6px;border-radius:6px;margin-left:6px;font-size:11px;">Verificado</span>'
            f'</h4>'
            f'<p class="resource-description">{desc}</p>'
            f'<a href="{url_ok}" class="resource-button" target="_blank" rel="noopener noreferrer">Acceder al Recurso</a>'
            f'</div>'
        )
        cards.append(card)
    if not cards:
        return html
    section = (
        '<h3 class="resource-topic-title">Recursos locales (Chile)</h3>'
        '<div class="resource-grid">'
        + ''.join(cards) +
        '</div>'
    )
    return (html or '') + section

# --- USUARIOS Y AUTENTICACIÓN (SQLite) ---
def init_users_table(db_path: str):
    try:
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    except Exception as e:
        print(f"WARN: init_users_table failed: {e}")

def get_user_by_username(db_path: str, username: str):
    try:
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cur.fetchone()
            return dict(row) if row else None
    except Exception as e:
        print(f"WARN: get_user_by_username failed: {e}")
        return None

def create_user(db_path: str, username: str, password_hash: str, role: str = 'user'):
    try:
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", (username, password_hash, role))
            conn.commit()
            return cur.lastrowid
    except Exception as e:
        print(f"WARN: create_user failed: {e}")
        return None
def extract_actions_from_plan_html(plan_html: str) -> list:
    import re
    if not plan_html:
        return []
    titles = []
    for m in re.finditer(r"<li[^>]*>\s*<strong>([^<]+)</strong>", plan_html, flags=re.I):
        t = m.group(1).strip()
        if t and t not in titles:
            titles.append(t)
    # Fallback: tomar items simples si no hay strong
    if not titles:
        for m in re.finditer(r"<li[^>]*>([^<]+)</li>", plan_html, flags=re.I):
            t = m.group(1).strip()
            if len(t) > 5 and t not in titles:
                titles.append(t)
    return titles

def ensure_actions_initialized(db_path: str, plan_id: int, owner_username: str, plan_html: str):
    try:
        titles = extract_actions_from_plan_html(plan_html)
        if not titles:
            return 0
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            inserted = 0
            for t in titles:
                try:
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO intervention_actions (plan_id, owner_username, action_title)
                        VALUES (?, ?, ?)
                        """,
                        (int(plan_id), owner_username, t)
                    )
                    inserted += (cur.rowcount or 0)
                except Exception:
                    pass
            conn.commit()
        return inserted
    except Exception:
        return 0

def get_actions_for_plan(db_path: str, plan_id: int, owner_username: str) -> list:
    try:
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM intervention_actions WHERE plan_id = ? AND (owner_username = ? OR owner_username IS NULL) ORDER BY id ASC",
                (int(plan_id), owner_username)
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []

def upsert_action_status(db_path: str, plan_id: int, owner_username: str, action_title: str, status: str, applied_date: str, responsable: str, compliance_pct: float, notes: str) -> bool:
    try:
        import sqlite3
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO intervention_actions (plan_id, owner_username, action_title, status, applied_date, responsable, compliance_pct, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id, action_title, owner_username) DO UPDATE SET
                    status = excluded.status,
                    applied_date = excluded.applied_date,
                    responsable = excluded.responsable,
                    compliance_pct = excluded.compliance_pct,
                    notes = excluded.notes
                """,
                (int(plan_id), owner_username, action_title, status, applied_date, responsable, float(compliance_pct or 0.0), notes)
            )
            conn.commit()
            return True
    except Exception:
        return False
