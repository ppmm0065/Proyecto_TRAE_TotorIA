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

embedding_model_instance = None
vector_store = None # For institutional context
vector_store_followups = None # Global FAISS for general follow-up search

def get_dataframe_from_session_file():
    current_file_path = session.get('current_file_path')
    if not current_file_path or not os.path.exists(current_file_path):
        print("app_logic.get_dataframe_from_session_file: No se encontró current_file_path en sesión o el archivo no existe.")
        return None
    
    try:
        df = None
        # --- LÓGICA DE LECTURA MEJORADA CON DOBLE INTENTO DE DELIMITADOR ---
        try:
            # Intento 1: Leer con el delimitador estándar (coma) y la codificación que maneja el BOM.
            df = pd.read_csv(current_file_path, skipinitialspace=True, encoding='utf-8-sig', sep=',')
            
            # Si después de leer con comas solo hay una columna, es probable que el delimitador sea punto y coma.
            if df.shape[1] == 1:
                print("Advertencia: El CSV parece tener una sola columna. Reintentando con delimitador de punto y coma (;).")
                df = pd.read_csv(current_file_path, skipinitialspace=True, encoding='utf-8-sig', sep=';')

        except UnicodeDecodeError:
            # Fallback de codificación si utf-8-sig falla
            df = pd.read_csv(current_file_path, skipinitialspace=True, encoding='latin-1', sep=';')
        except Exception as e:
            print(f"Error crítico al leer CSV en get_dataframe_from_session_file: {e}")
            traceback.print_exc()
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
        processed_lines = []
        try:
            with open(full_filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line.startswith('"') and stripped_line.endswith('"'): processed_lines.append(stripped_line[1:-1])
                    else: processed_lines.append(stripped_line)
        except UnicodeDecodeError:
            processed_lines = []
            with open(full_filepath, 'r', encoding='latin-1') as f:
                for line in f:
                    stripped_line = line.strip()
                    if stripped_line.startswith('"') and stripped_line.endswith('"'): processed_lines.append(stripped_line[1:-1])
                    else: processed_lines.append(stripped_line)
        if not processed_lines: return "Error: Archivo CSV vacío o no se pudo leer."
        csv_string_io = io.StringIO("\n".join(processed_lines))
        df = pd.read_csv(csv_string_io, skipinitialspace=True)
        # Aquí también añadimos index=False por consistencia.
        return df.to_string(index=False) if df is not None else "Error: No se pudo cargar el DataFrame."
    except FileNotFoundError: return "Error: Archivo CSV de datos no encontrado."
    except Exception as e: return f"Error al procesar CSV: {e}"

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

def analyze_data_with_gemini(data_string, user_prompt, vs_inst, vs_followup, 
                             chat_history_string="", is_reporte_360=False, 
                             is_plan_intervencion=False, is_direct_chat_query=False,
                             entity_type=None, entity_name=None):
    api_key = current_app.config.get('GEMINI_API_KEY')
    num_relevant_chunks_config = current_app.config.get('NUM_RELEVANT_CHUNKS', 7)
    model_name = 'gemini-2.5-flash' # NUEVO: Definir el nombre del modelo para usarlo en cálculos

    # NUEVO: Estructura de retorno de error estandarizada
    def create_error_response(error_message):
        return {
            'html_output': f"<div class='text-red-700 p-3 bg-red-100 border rounded-md'><strong>Error:</strong> {error_message}</div>",
            'raw_markdown': f"Error: {error_message}", 'error': error_message,
            'model_name': model_name, 'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0,
            'input_cost': 0, 'output_cost': 0, 'total_cost': 0
        }

    if not api_key:
        return create_error_response("Configuración: Falta la clave API de Gemini.")
    if not embedding_model_instance:
        return create_error_response("Crítico: Modelo de embeddings no cargado.")


    retrieved_context_inst = "Contexto Institucional no disponible o no buscado."
    retrieved_context_followup_header = "Historial de Seguimiento Relevante de la Entidad"
    retrieved_context_followup = "Historial de Seguimiento de la Entidad no disponible o no buscado."
    
    default_analysis_prompt = current_app.config.get('DEFAULT_ANALYSIS_PROMPT', "Realiza un análisis general de los datos.")
    final_user_instruction = user_prompt

    if entity_type and entity_name:
        retrieved_context_followup_header = f"Historial de Seguimiento Específico para {entity_type.capitalize()} '{entity_name}'"

    # --- Institutional Context Retrieval (General) ---
    if not is_reporte_360 and not is_plan_intervencion: 
        if not final_user_instruction: 
             final_user_instruction = default_analysis_prompt
        if vs_inst: # Institutional context vector store
            try:
                relevant_docs_inst = vs_inst.similarity_search(final_user_instruction, k=num_relevant_chunks_config)
                if relevant_docs_inst:
                    context_list = [f"--- Contexto Inst. {i+1} (Fuente: {os.path.basename(doc.metadata.get('source', 'Unknown'))}) ---\n{doc.page_content}\n" for i, doc in enumerate(relevant_docs_inst)]
                    retrieved_context_inst = "\n".join(context_list)
                else: retrieved_context_inst = "No se encontró contexto institucional relevante para esta consulta."
            except Exception as e: retrieved_context_inst = f"Error al buscar contexto institucional: {e}"
        
        # --- Follow-up Context Retrieval ---
        relevant_docs_fu_final = []
        try:
            if entity_type and entity_name: # Entity-specific query
                print(f"DEBUG: RAG - Entity specific search for {entity_type} '{entity_name}'. Loading fresh from DB.")
                
                all_db_followups_as_lc_docs = load_follow_ups_as_documents(current_app.config['DATABASE_FILE'])
                if all_db_followups_as_lc_docs is None: 
                    all_db_followups_as_lc_docs = [] 
                    print("ERROR: RAG - load_follow_ups_as_documents returned None. Cannot proceed with entity-specific RAG.")
                
                entity_specific_docs_from_db = []
                for doc_candidate in all_db_followups_as_lc_docs:
                    if isinstance(doc_candidate, Document) and hasattr(doc_candidate, 'metadata'):
                        meta_entity_type = doc_candidate.metadata.get('related_entity_type')
                        meta_entity_name = doc_candidate.metadata.get('related_entity_name')
                        if meta_entity_type == entity_type and meta_entity_name == entity_name:
                            entity_specific_docs_from_db.append(doc_candidate)
                
                print(f"DEBUG: RAG - Found {len(entity_specific_docs_from_db)} documents from DB matching entity: {entity_type} '{entity_name}'.")

                if entity_specific_docs_from_db:
                    try:
                        temp_vs_entity_docs = FAISS.from_documents(entity_specific_docs_from_db, embedding_model_instance)
                        k_temp_search = min(len(entity_specific_docs_from_db), num_relevant_chunks_config)
                        if k_temp_search > 0:
                            relevant_docs_fu_final = temp_vs_entity_docs.similarity_search(final_user_instruction, k=k_temp_search)
                            print(f"DEBUG: RAG - Re-ranked {len(entity_specific_docs_from_db)} entity-specific DB docs. Selected top {len(relevant_docs_fu_final)} by similarity to prompt.")
                        else:
                            relevant_docs_fu_final = []
                    except Exception as e_temp_vs:
                        print(f"DEBUG: RAG - Error creating/searching temp VS for entity docs: {e_temp_vs}. Fallback to most recent N.")
                        entity_specific_docs_from_db_sorted = sorted(
                            entity_specific_docs_from_db, 
                            key=lambda d: d.metadata.get('timestamp', '1970-01-01 00:00:00'), 
                            reverse=True
                        )
                        relevant_docs_fu_final = entity_specific_docs_from_db_sorted[:num_relevant_chunks_config]
                        
                if not relevant_docs_fu_final:
                    retrieved_context_followup = f"No se encontraron seguimientos específicos para {entity_type.capitalize()} '{entity_name}' que sean relevantes para la consulta actual."
                    print(f"DEBUG: RAG - No specific follow-ups relevant to prompt for {entity_type} {entity_name} after DB filter/re-rank.")
            
            else: # General search (no entity specified) - use the global vs_followup
                if vs_followup:
                    relevant_docs_fu_final = vs_followup.similarity_search(final_user_instruction, k=num_relevant_chunks_config)
                    if not relevant_docs_fu_final:
                        retrieved_context_followup = "No se encontraron Reportes 360, Observaciones, u otros seguimientos previos relevantes para esta consulta general."
                    print(f"DEBUG: RAG - General followup search using global FAISS. Found {len(relevant_docs_fu_final)} docs.")
                else:
                    print("DEBUG: RAG - Global vs_followup is None for general search or not initialized.")
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
        
    # --- Construct final prompt for Gemini ---
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name) # NUEVO: Usar la variable model_name
        
        prompt_parts = []
        prompt_parts.append(current_app.config.get('GEMINI_SYSTEM_ROLE_PROMPT', "Eres un asistente útil."))

        if not is_reporte_360 and not is_plan_intervencion: 
            if chat_history_string:
                prompt_parts.append(f"\n--- INICIO HISTORIAL DE CONVERSACIÓN PREVIA ---\n{chat_history_string}\n--- FIN HISTORIAL DE CONVERSACIÓN PREVIA ---\n")
            
            prompt_parts.extend([
                "Contexto Institucional Relevante (Información general de la institución, si se encontró para la consulta):\n```\n", retrieved_context_inst, "\n```\n---",
                f"{retrieved_context_followup_header} (Reportes 360 previos, Observaciones registradas, etc., si se encontraron para la consulta):\n```\n", retrieved_context_followup, "\n```\n---",
            ])
        
        prompt_parts.extend([
            "Contexto Principal de Datos para la Consulta Actual (Datos del CSV para la entidad, o Reporte 360 base para el plan):\n```\n", data_string, "\n```\n---",
            "Instrucción Específica del Usuario (Pregunta Actual o Solicitud):\n", f'"{final_user_instruction}"', "\n---",
        ])

        if not is_direct_chat_query and not is_reporte_360 and not is_plan_intervencion:
            prompt_parts.append(current_app.config.get('GEMINI_FORMATTING_INSTRUCTIONS', "Formatea tu respuesta claramente en Markdown."))
        elif is_direct_chat_query:
            pass 

        final_prompt_string = "\n".join(filter(None, prompt_parts)) 

        response = model.generate_content(final_prompt_string)
        
        # --- INICIO: NUEVO BLOQUE DE CONTEO DE TOKENS Y CÁLCULO DE COSTOS ---
        input_tokens, output_tokens, total_tokens = 0, 0, 0
        cost_input, cost_output, total_cost = 0.0, 0.0, 0.0
        
        try:
            # Extraer conteo de tokens desde la metadata de la respuesta
            usage_info = response.usage_metadata
            input_tokens = usage_info.prompt_token_count
            output_tokens = usage_info.candidates_token_count
            total_tokens = usage_info.total_token_count
            
            # Calcular costos
            pricing_info = current_app.config.get('MODEL_PRICING', {}).get(model_name, {})
            if pricing_info:
                input_cost_per_million = pricing_info.get('input_per_million', 0)
                output_cost_per_million = pricing_info.get('output_per_million', 0)
                cost_input = (input_tokens / 1_000_000) * input_cost_per_million
                cost_output = (output_tokens / 1_000_000) * output_cost_per_million
                total_cost = cost_input + cost_output
            
            # Guardar en la base de datos el consumo diario
            db_path = current_app.config['DATABASE_FILE']
            guardar_consumo_diario(db_path, model_name, input_tokens, output_tokens, total_cost)

        except Exception as e:
            print(f"Advertencia: No se pudo calcular el consumo de tokens/costos: {e}")
            traceback.print_exc()
        # --- FIN: NUEVO BLOQUE DE CONTEO Y CÁLCULO ---

        raw_response_text, html_output, error_message = "", "", None
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
                html_output = markdown.markdown(raw_response_text, extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists'])
        
        if error_message: 
            print(f"Error en Gemini: {error_message}")
            return create_error_response(error_message)
        
        # NUEVO: Devolver un diccionario con toda la información
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
        # --- FIN: NUEVA TABLA PARA CONSUMO DE TOKENS ---
        
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

def save_reporte_360_to_db(db_path, filename, tipo_entidad, nombre_entidad, reporte_360_markdown, prompt_reporte_360="Reporte 360 generado automáticamente"):
    # No changes
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO follow_ups 
                (related_filename, related_prompt, follow_up_comment, follow_up_type, related_entity_type, related_entity_name, related_analysis)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (filename, prompt_reporte_360, reporte_360_markdown, 'reporte_360', tipo_entidad, nombre_entidad, None))
            conn.commit()
            last_id = cursor.lastrowid
        print(f"Reporte 360 para {tipo_entidad} {nombre_entidad} (archivo: {filename}) guardado en la BD con ID: {last_id}.")
        return last_id
    except Exception as e:
        print(f"Error CRÍTICO al guardar Reporte 360 en la BD: {e}")
        traceback.print_exc()
        return None

def save_observation_for_reporte_360(db_path, parent_reporte_360_id, observador_nombre, observacion_text, tipo_entidad, nombre_entidad, csv_filename):
    # No changes
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            related_prompt_text = f"Observación de: {observador_nombre}"
            cursor.execute("""
                INSERT INTO follow_ups
                (related_filename, related_prompt, related_analysis, follow_up_comment, follow_up_type, related_entity_type, related_entity_name)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (csv_filename, related_prompt_text, str(parent_reporte_360_id), observacion_text, 'observacion_reporte_360', tipo_entidad, nombre_entidad))
            conn.commit()
        print(f"Observación para Reporte 360 ID {parent_reporte_360_id} (Entidad: {tipo_entidad} {nombre_entidad}, Archivo: {csv_filename}) guardada.")
        return True
    except Exception as e:
        print(f"Error CRÍTICO al guardar observación para Reporte 360 en la BD: {e}")
        traceback.print_exc()
        return False

def load_follow_ups_as_documents(db_path): 
    follow_up_docs = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row 
            cursor = conn.cursor()
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
    # No changes
    reportes = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, follow_up_comment FROM follow_ups 
                WHERE follow_up_type = 'reporte_360' 
                  AND related_entity_type = ? 
                  AND related_entity_name = ? 
                  AND related_filename = ? 
                ORDER BY timestamp DESC
            """, (tipo_entidad, nombre_entidad, current_filename))
            
            for row in cursor.fetchall():
                reporte_markdown = row['follow_up_comment']
                reporte_html = markdown.markdown(reporte_markdown, extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists'])
                timestamp_dt = datetime.datetime.strptime(row["timestamp"], '%Y-%m-%d %H:%M:%S')
                reportes.append({
                    "id": row["id"], 
                    "timestamp": timestamp_dt.strftime('%d/%m/%Y %H:%M:%S'),
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
    # No changes
    global embedding_model_instance, vector_store, vector_store_followups 
    print("--- Iniciando Setup RAG ---")
    try:
        embedding_model_name = app_config.get('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')
        embedding_model_instance = SentenceTransformerEmbeddings(model_name=embedding_model_name)
        print(f"Modelo Embeddings '{embedding_model_name}' cargado.")
    except Exception as e: print(f"Error CRÍTICO Embeddings: {e}"); traceback.print_exc(); return
    if not reload_institutional_context_vector_store(app_config): print("Advertencia: Vector store institucional no inicializado.")
    if not reload_followup_vector_store(app_config): print("Advertencia: Vector store seguimientos/reportes/obs no inicializado.")
    print("--- Setup RAG Finalizado ---")

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
    # No changes
    global vector_store, embedding_model_instance 
    if not embedding_model_instance: print("Error CRÍTICO: Modelo embeddings no disponible."); vector_store = None; return False
    folder, path = app_config.get('CONTEXT_DOCS_FOLDER'), app_config.get('FAISS_INDEX_PATH')
    if not folder or not path: print("Error: Rutas de contexto o índice FAISS no configuradas."); vector_store = None; return False 
    chunks = _load_and_split_context_documents(folder)
    if chunks: 
        try: vs_temp = FAISS.from_documents(chunks, embedding_model_instance); vs_temp.save_local(path); vector_store = vs_temp; print(f"Índice FAISS institucional actualizado: {path}"); return True
        except Exception as e: print(f"Error creando/guardando índice FAISS institucional: {e}"); traceback.print_exc(); vector_store = None; return False
    elif os.path.exists(path) and os.path.isdir(path) and os.listdir(path): 
        try: vector_store = FAISS.load_local(path, embedding_model_instance, allow_dangerous_deserialization=True); print(f"Índice FAISS institucional cargado: {path}"); return True
        except Exception as e: print(f"Error cargando índice FAISS institucional: {e}"); traceback.print_exc(); vector_store = None; return False
    else: print(f"No hay docs de contexto y no existe índice FAISS institucional: {path}."); vector_store = None; return True

def reload_followup_vector_store(app_config): 
    global vector_store_followups, embedding_model_instance
    if not embedding_model_instance: print("Error CRÍTICO: Modelo embeddings no disponible."); vector_store_followups = None; return False
    path, db_path = app_config.get('FAISS_FOLLOWUP_INDEX_PATH'), app_config.get('DATABASE_FILE')
    if not path or not db_path: print("Error: Rutas de índice FAISS o DB no configuradas."); vector_store_followups = None; return False
    
    docs = load_follow_ups_as_documents(db_path) 
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
    kws_l = [kw.lower() for kw in kws]
    df_neg = df_c[df_c[obs_col].apply(lambda o: any(kw in str(o).lower() for kw in kws_l) if pd.notna(o) else False)]
    
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
        
        agg[niv] = {
            "total_alumnos": total_alumnos_nivel,
            "promedio_general_notas": round(promedio_general_nivel, 2) if pd.notna(promedio_general_nivel) else "N/A",
            "asistencia_promedio": f"{asist_prom:.1%}" if pd.notna(asist_prom) else "N/A",
            "curso_menor_promedio": {"nombre": c_menor["nombre"], "promedio": round(c_menor["promedio"], 2) if pd.notna(c_menor["promedio"]) else "N/A"},
            "curso_mayor_promedio": {"nombre": c_mayor["nombre"], "promedio": round(c_mayor["promedio"], 2) if pd.notna(c_mayor["promedio"]) else "N/A"}
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
    
    analysis_result = analyze_data_with_gemini(
        data_string=reporte_360_markdown, 
        user_prompt=prompt_plan, 
        vs_inst=None, 
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

def save_intervention_plan_to_db(db_path, filename, tipo_entidad, nombre_entidad, plan_markdown, reporte_360_base_markdown):
    """
    Guarda el plan de intervención en la BD y devuelve el ID de la nueva fila.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            prompt = f"Plan Intervención para {tipo_entidad.capitalize()}: {nombre_entidad}"
            cursor.execute("""
                INSERT INTO follow_ups 
                (related_filename, related_prompt, related_analysis, follow_up_comment, follow_up_type, related_entity_type, related_entity_name) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (filename, prompt, reporte_360_base_markdown, plan_markdown, 'intervention_plan', tipo_entidad, nombre_entidad))
            
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
                plans.append({"id": row["id"], "timestamp": datetime.datetime.strptime(row["timestamp"], '%Y-%m-%d %H:%M:%S').strftime('%d/%m/%Y %H:%M'),
                              "plan_markdown": row["follow_up_comment"], "plan_html": markdown.markdown(row["follow_up_comment"], extensions=['fenced_code', 'tables', 'nl2br', 'sane_lists'])})
    except Exception as e: print(f"Error recuperando planes: {e}"); traceback.print_exc()
    return plans

def search_web_for_support_resources(plan_intervencion_markdown, tipo_entidad, nombre_entidad):
    # No changes to the query generation part
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
        return final_html

    except Exception as e:
        print(f"CRITICAL DEBUG: Error al SIMULAR y sugerir recursos con Gemini: {e}") 
        traceback.print_exc()
        return f"Error: No pude generar sugerencias de recursos. Detalle: {e}"