# config.py
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde .env (especialmente para GEMINI_API_KEY)
load_dotenv()

class Config:
    """Clase base de configuración."""
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'una-clave-secreta-muy-dificil-de-adivinar')
    DEBUG = False
    TESTING = False

    # Configuraciones específicas de la aplicación
    UPLOAD_FOLDER = 'uploads'
    CONTEXT_DOCS_FOLDER = 'context_docs' 
    DATABASE_FILE = 'seguimiento.db' 
    OBSERVACIONES_COL = 'Observacion de conducta' # Normalizado

    # Configuraciones de RAG y Modelos
    FAISS_INDEX_PATH = "./faiss_index_multi" 
    FAISS_FOLLOWUP_INDEX_PATH = "./faiss_index_followups"
    EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
    NUM_RELEVANT_CHUNKS = 7 # Aumentado para más contexto
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    # Zona horaria centralizada para conversiones de tiempo
    TIMEZONE_NAME = os.environ.get('TIMEZONE_NAME', 'America/Santiago')

# --- INICIO: CONFIGURACIÓN DE TARIFAS DE MODELOS (NUEVO) ---
    # Estructura para almacenar las tarifas de los modelos de IA en USD por cada 1,000,000 de tokens.
    # Esto permite una fácil actualización y la adición de nuevos modelos en el futuro.
    # Nota: Las tarifas pueden cambiar. Verificar siempre los precios oficiales del proveedor.
    # Para Gemini 1.5 Flash, los precios para prompts > 128k tokens son los indicados.
    MODEL_PRICING = {
        'gemini-2.5-flash': {
            'input_per_million': 0.35,  # USD
            'output_per_million': 1.05, # USD
            'provider': 'Google'
        },
        # Ejemplo para un futuro modelo de OpenAI
        'gpt-4o': {
            'input_per_million': 5.00,
            'output_per_million': 15.00,
            'provider': 'OpenAI'
        },
        # Ejemplo para un futuro modelo de Anthropic
        'claude-3-haiku': {
            'input_per_million': 0.25,
            'output_per_million': 1.25,
            'provider': 'Anthropic'
        }
    }
    # --- FIN: CONFIGURACIÓN DE TARIFAS DE MODELOS ---

    # --- INICIO: CONFIGURACIONES PARA PROMPTS DE GEMINI (ACTUALIZADAS) ---

    GEMINI_SYSTEM_ROLE_PROMPT = """Eres TutorIA360, un asistente experto en pedagogía y análisis de datos educativos. Tu misión es proporcionar estrategias y consejos ACCIONABLES y PERSONALIZADOS para mejorar el desempeño académico y conductual de los estudiantes.

**CRÍTICO: Fundamenta todas tus recomendaciones en teorías pedagógicas reconocidas y estudios de investigación educativa validados a nivel mundial, tales como los modelos educativos de paises como Finlandia, Canada, Singapur, Inglaterra y Corea del Sur. Cuando sea pertinente, menciona brevemente el concepto o teoría que respalda tu sugerencia (ej., "basado en los principios del aprendizaje constructivista...", "considerando la teoría de las inteligencias múltiples de Gardner...", "aplicando técnicas de refuerzo positivo Skinneriano...").**

**MODO DE RESPUESTA:**
* **Para Preguntas Directas (Chat):** Responde de manera concisa y directa a la pregunta del usuario. Sintetiza la información relevante del contexto (CSV, institucional, historial de reportes y registro de observaciones) para informar tu respuesta sin replicar documentos completos. El objetivo es una conversación fluida y útil que informe al usuario de los avances o cambios en el desarrollo del estudiante.
* **Para Solicitudes de Análisis/Reportes Estructurados:** Sigue las directrices de formato específicas para cada tipo de solicitud (Reporte 360, Plan de Intervención, etc.).

Utiliza la siguiente información para formular tus respuestas, integrándola de manera coherente:
1.  **Contexto Institucional Relevante**: Documentos proporcionados por la institución (prioriza esta información para alinear tus respuestas con las políticas y recursos existentes).
2.  **Historial de Seguimiento Relevante de la Entidad (Alumno/Curso)**: Este historial es crucial y puede incluir: Reportes 360 Previos, Observaciones Registradas por Usuarios, Planes de Intervención Anteriores y otros comentarios. Presta atención a la evolución temporal.
3.  **Contexto de Datos Proporcionado (Estudiantes CSV)**: Analiza detalladamente los datos actuales del o los estudiantes (notas, observaciones del CSV, asistencia, etc.) para identificar patrones. **Si los datos del estudiante incluyen una columna llamada 'materias_debiles', presta especial atención a su contenido, ya que indica áreas específicas de dificultad reportadas.**
4.  **Instrucción Específica del Usuario**: Responde directamente a la pregunta del usuario, considerando el modo de respuesta apropiado (directo o estructurado).
5.  **Resumen de Datos Históricos (Evolución Cuantitativa)**: (SOLO SI SE PROPORCIONA) Este es un resumen de SQL precalculado que muestra la evolución de las *notas* de un estudiante.
6.  **Resumen de Datos Cualitativos Históricos (Evolución de Comportamiento)**: (SOLO SI SE PROPORCIONA) Este es un listado cronológico de *todas* las observaciones de conducta, entrevistas y datos familiares registrados para un estudiante en cargas de datos anteriores. Úsalo para realizar un análisis interpretativo de su evolución conductual y actitudinal.

Sé claro, conciso y empático. Evita la jerga excesiva. Tu objetivo es empoderar a los docentes y profesionales de la educación con herramientas prácticas."""

    # NUEVO: Prompt específico para el Reporte 360
    PROMPT_REPORTE_360 = """
    Genera un "Reporte de Aprendizaje y Conducta" para el {tipo_entidad} '{nombre_entidad}'.
    El reporte debe ser conciso, estructurado y fácil de leer, con un máximo de 250 palabras en total.
    Utiliza estrictamente el siguiente formato Markdown:

    ### Fortalezas Destacables
    * [Lista de 2-4 fortalezas clave observadas en los datos, cada una en un ítem]
    * ...

    ### Desafíos Significativos
    * **Académicos:** [Menciona 1-3 desafíos académicos. Si la columna 'materias_debiles' existe y tiene contenido, úsalo para informar esta sección.]
    * **Conductuales:** [Menciona 1-3 desafíos conductuales basados en las observaciones.]

    ### Sugerencia Clave
    * [Basado en los desafíos, proporciona una recomendación principal o un próximo paso sugerido, fundamentado en un modelo pedagógico reconocido.]

    Sigue el rol y las directrices generales definidas en GEMINI_SYSTEM_ROLE_PROMPT.
    """

    # NUEVO: Prompt específico para el Plan de Intervención
    PROMPT_PLAN_INTERVENCION = """
    Basado en el siguiente Reporte 360 para el {tipo_entidad} '{nombre_entidad}':
    ```markdown
    {reporte_360_markdown}
    ```
    Genera un "Plan de Intervención" breve y concreto. El plan debe estar claramente estructurado en Markdown, utilizando encabezados y listas.
    Sigue estrictamente este formato:

    ### Objetivos Claros
    1.  **[Título del Objetivo 1]:** [Descripción breve del objetivo]
    2.  **[Título del Objetivo 2]:** [Descripción breve del objetivo]
    3.  ...

    ### Acciones y Estrategias Sugeridas
    * **[Nombre de la Estrategia 1]:**
        * **Acción:** [Paso concreto y práctico a implementar].
        * **Fundamentación:** [Breve mención al estudio o teoría pedagógica que la respalda. Ej: Modelo Finlandés, Teoría de Gardner, etc.].
    * **[Nombre de la Estrategia 2]:**
        * **Acción:** [Paso concreto y práctico a implementar].
        * **Fundamentación:** [Breve mención al estudio o teoría pedagógica que la respalda].
    * ...
    """
    GEMINI_FORMATTING_INSTRUCTIONS = """**SOLO CUANDO SE SOLICITE UN ANÁLISIS ESTRUCTURADO (NO PARA PREGUNTAS DIRECTAS EN CHAT)**, formatea tu respuesta utilizando Markdown de la siguiente manera:

Para análisis individuales de estudiantes:
### Análisis y Estrategias para [Nombre del Estudiante]

**1. Diagnóstico Resumido:**
   - Breve descripción de la situación actual basada en los datos y el contexto, con una extensión máxima de 300 palabras. Si existe información en 'materias_debiles', incorpórala aquí. Considera la evolución si hay reportes u observaciones previas.

**2. Objetivos de Mejora:**
   - Lista de 2-3 objetivos claros y medibles, considerando las 'materias_debiles' si aplica y la información histórica.

**3. Estrategias de Apoyo Sugeridas:**
   - **Estrategia 1:** [Descripción de la estrategia, idealmente abordando alguna 'materia_debil' si es relevante y fundamentada en el historial]
     - *Fundamentación:* [Breve mención al estudio o teoría pedagógica que la respalda]
   - **Estrategia 2:** [Descripción de la estrategia]
     - *Fundamentación:* [Breve mención al estudio o teoría pedagógica que la respalda]
   - ... (más estrategias si es necesario)

**4. Indicadores de Seguimiento:**
   - ¿Cómo se medirá el progreso?

Para análisis grupales o de tendencias:
### Análisis de [Grupo/Tendencia Específica]

**1. Observaciones Clave:**
   - Patrones identificados en el grupo o tendencia. Si se analizan datos individuales que incluyen 'materias_debiles', busca patrones también en esta columna. Considera tendencias observadas en el historial.

**2. Posibles Causas Raíz (basadas en datos, historial y conocimiento experto):**
   - Hipótesis sobre los factores que contribuyen.

**3. Estrategias de Intervención Grupal:**
   - **Estrategia A:** [Descripción]
     - *Fundamentación:* [Base teórica/estudio]
   - **Estrategia B:** [Descripción]
     - *Fundamentación:* [Base teórica/estudio]

**4. Consideraciones Adicionales:**
   - Cualquier otro punto relevante.

Utiliza encabezados (##, ###), listas con viñetas (-) o numeradas (1.), y **negritas** para destacar puntos importantes. Asegúrate de que la respuesta sea fácil de leer y esté bien organizada."""
    # --- FIN: CONFIGURACIONES PARA PROMPTS DE GEMINI ---

    # Constantes de Chat y Sesión
    MAX_CHAT_HISTORY_DISPLAY_ON_ANALYZE = 3
    MAX_CHAT_HISTORY_TURNS_FOR_GEMINI_PROMPT = 3
    MAX_CHAT_HISTORY_SESSION_STORAGE = 10

    # Nombres de Columnas del CSV (importante mantenerlos consistentes)
    NOMBRE_COL = 'Nombre' 
    CURSO_COL = 'curso'
    # La columna 'Promedio' ahora será calculada, no leída directamente. La mantenemos para uso interno.
    PROMEDIO_COL = 'Promedio' 
    
    # --- NUEVAS COLUMNAS PARA EL FORMATO "LARGO" ---
    ASIGNATURA_COL = 'Asignatura'
    NOTA_COL = 'Nota'
    
    # Columnas opcionales que enriquecen el análisis
    MATERIAS_DEBILES_COL = 'materias_debiles' 
    ASISTENCIA_COL = 'Asistencia' 
    OBSERVACIONES_COL = 'Observacion de conducta'
    EDAD_COL = 'edad' 
    PROFESOR_COL = 'profesor' 
    FAMILIA_COL = 'Familia' 
    ENTREVISTAS_COL = 'Entrevistas'

    # Palabras/expresiones clave para detectar observaciones de conducta NEGATIVAS en el CSV.
    # Se usa coincidencia por subcadenas y posteriormente se normaliza (sin tildes) en app_logic.
    # FIX: se corrige el error de coma faltante entre "insulta" y "agrede" y se amplían sinónimos comunes.
    NEGATIVE_OBSERVATION_KEYWORDS = [
        "copia en la prueba",
        "es suspendido de clases",
        "golpea a un companero",
        "golpea a una companera",
        "es agresivo",
        "agresion",
        "agresiones",
        "ofende al profesor",
        "ofende a sus companeros",
        "interrumpe en clases",
        "interrumpe la clase",
        "interrumpe",
        "insulta",
        "agrede",
        "molesta",
        "falta de respeto",
        "amonestacion",
        "llamado de atencion"
    ]
    
    DEFAULT_ANALYSIS_PROMPT = "Realiza un analisis general de los datos y sugiere posibles areas de enfoque o intervencion." 
    
    TEXT_SPLITTER_CHUNK_SIZE = 1000
    TEXT_SPLITTER_CHUNK_OVERLAP = 150

    # Umbrales para alertas (asegurarse que sean numéricos)
    LOW_PERFORMANCE_THRESHOLD_GRADE = 4.0 
    SIGNIFICANT_PERCENTAGE_LOW_PERF_ALERT = 0.20
    MIN_STUDENTS_FOR_CONDUCT_ALERT = 3
    SIGNIFICANT_PERCENTAGE_CONDUCT_ALERT = 0.15
    EDAD_COL = 'edad' 
    PROFESOR_COL = 'profesor' 
    FAMILIA_COL = 'Familia' 
    ENTREVISTAS_COL = 'Entrevistas' 
    ASISTENCIA_COL = 'Asistencia' 

    # --- NUEVA CONFIGURACIÓN DE FEATURES Y TRAZABILIDAD ---
    # Flags para activar/desactivar características de resumen y contexto
    ENABLE_ATTENDANCE_SUMMARY = True
    ENABLE_QUALITATIVE_SUMMARY = True
    INCLUDE_KEY_DOCS_IN_PROMPT = True
    # Control del bloque de Documentos Clave recientes
    KEY_DOCS_MAX_ITEMS = 2
    KEY_DOCS_INCLUDE_EXCERPTS = True
    KEY_DOCS_EXCERPT_LINES = 8
    KEY_DOCS_EXCERPT_CHARS = 1000

    # Política de manejo de ambigüedades de cursos
    # 'ask' -> Solicita letra/paralelo cuando hay múltiples opciones
    # 'default' -> Selecciona por defecto usando DEFAULT_PARALLEL si existe
    COURSE_AMBIGUITY_POLICY = 'ask'
    DEFAULT_PARALLEL = 'A'

    # Logging adicional para trazabilidad de secciones del prompt
    LOG_PROMPT_SECTIONS = True

    # Presupuesto de prompt y secciones (en caracteres)
    ENABLE_PROMPT_BUDGETING = True
    PROMPT_MAX_CHARS = 24000
    PROMPT_SECTION_CHAR_BUDGETS = {
        'chat_history': 0.10,
        'key_docs_and_qualitative': 0.25,
        'rag_institutional': 0.25,
        'rag_followups': 0.20,
        'historical_quantitative': 0.10,
        'csv_or_base_context': 0.10
    }
    # Presupuestos máximos específicos para RAG (aplicados además del total)
    RAG_INST_MAX_CHARS = 8000
    RAG_FOLLOWUP_MAX_CHARS = 8000

    # Listas de keywords configurables (se pueden sobreescribir por entorno)
    EVOLUTION_KEYWORDS = [
        'evolución', 'historial', 'progreso', 'rendimiento histórico',
        'variación', 'tendencia', 'cambio'
    ]
    NEGATIVE_EVOLUTION_KEYWORDS = [
        'empeoramiento', 'peor', 'bajado', 'disminuido', 'caída', 'regresión'
    ]
    ATTENDANCE_KEYWORDS = [
        'asistencia', 'inasistencia', 'faltas', 'presentismo', 'ausentismo',
        'evolución de asistencia', 'histórico de asistencia', 'tendencia de asistencia'
    ]
    QUALITATIVE_KEYWORDS = [
        'cualitativo', 'conducta', 'comportamiento', 'observaciones', 'entrevistas', 'familia',
        'resumen cualitativo', 'historial cualitativo', 'evolución conductual'
    ]

    # Nivel de logging de la aplicación (INFO por defecto)
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')


class DevelopmentConfig(Config):
    """Configuración para desarrollo."""
    DEBUG = True

class ProductionConfig(Config):
    """Configuración para producción."""
    DEBUG = False

config_by_name = dict(
    dev=DevelopmentConfig,
    prod=ProductionConfig
)

key = Config.SECRET_KEY
