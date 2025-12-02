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
    NUM_RELEVANT_CHUNKS_INST = 15 # Fragmentos para índice institucional
    NUM_RELEVANT_CHUNKS_FU = 10  # Fragmentos para índice de seguimientos
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    GEMINI_MODEL_NAME = os.environ.get('GEMINI_MODEL_NAME', 'gemini-2.5-flash')
    # Zona horaria centralizada para conversiones de tiempo
    TIMEZONE_NAME = os.environ.get('TIMEZONE_NAME', 'America/Santiago')

# --- INICIO: CONFIGURACIÓN DE TARIFAS DE MODELOS (NUEVO) ---
    # Estructura para almacenar las tarifas de los modelos de IA en USD por cada 1,000,000 de tokens.
    # Esto permite una fácil actualización y la adición de nuevos modelos en el futuro.
    # Nota: Las tarifas pueden cambiar. Verificar siempre los precios oficiales del proveedor.
    # Para Gemini 1.5 Flash, los precios para prompts > 128k tokens son los indicados.
    MODEL_PRICING = {
        'gemini-2.5-flash': {
            'input_per_million': 0.30,  # USD
            'output_per_million': 2.50, # USD
            'provider': 'Google'
        },
        'gemini-3-pro-preview': {
            'provider': 'Google',
            'tiers': [
                {
                    'max_prompt_tokens': 200000,
                    'input_per_million': 2.00,
                    'output_per_million': 12.00
                },
                {
                    'min_prompt_tokens': 200001,
                    'input_per_million': 4.00,
                    'output_per_million': 18.00
                }
            ]
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

    GEMINI_SYSTEM_ROLE_PROMPT = """Eres un consultor internacional de reconocimiento mundial, experto en pedagogía y análisis de datos educativos. Tu misión es proporcionar estrategias y consejos ACCIONABLES y PERSONALIZADOS para mejorar el desempeño académico y conductual de los estudiantes.

**CRÍTICO: Fundamenta todas tus recomendaciones en teorías pedagógicas reconocidas, basadas en la ciencia pedagógica, (por ejemplo: teoría de la carga cognitiva y de estrategias para la captura de atención y almacenamiento de información en la memoria de largo plazo)
y estudios de investigación educativa validados a nivel mundial, tales como los modelos educativos de paises como Finlandia, Canada, Singapur, Inglaterra y Corea del Sur. Cuando sea pertinente, menciona brevemente el concepto o teoría
que respalda tu sugerencia (ej., "basado en los principios del aprendizaje...", "considerando la teoría de...", "aplicando técnicas de...").**

Debes poner atención especialmente a las siguientes consideraciones:
* **Dificultades Específicas del Estudiante**: Asegúrate de adaptar tus recomendaciones a las necesidades individuales del estudiante, considerando su nivel de habilidad, desafíos educativos y patrones de comportamiento.
* **Contexto Institucional**: Utiliza la información proporcionada por la institución para informar tus recomendaciones, considerando sus políticas, recursos y prácticas educativas.
* **Historial de Seguimiento**: Presta especial atención al historial de seguimiento del estudiante, incluyendo reportes 360 previos, observaciones registradas por usuarios y planes de intervención anteriores.

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

    # NUEVO: Prompt específico para el Reporte 360 (alineado al GEMINI_SYSTEM_ROLE_PROMPT)
    PROMPT_REPORTE_360 = """
    Genera un "Reporte de Aprendizaje y Conducta" para el {tipo_entidad} '{nombre_entidad}'.
    El reporte debe ser conciso, estructurado y fácil de leer (máx. 250 palabras) y debe estar explícitamente fundamentado en ciencia pedagógica.

    Utiliza estrictamente el siguiente formato Markdown:

    ### Fortalezas Destacables
    * [Lista de 2-4 fortalezas clave observadas en los datos]

    ### Desafíos Significativos
    * **Académicos:** [1-3 desafíos; si 'materias_debiles' existe y tiene contenido, incorpóralo]
    * **Conductuales:** [1-3 desafíos a partir de observaciones y registros]

    ### Sugerencia Clave
    * [Un próximo paso accionable y personalizado, fundamentado en teorías pedagógicas reconocidas (p. ej., carga cognitiva, estrategias de codificación en memoria de largo plazo, aprendizaje colaborativo), citando brevemente el concepto]

    ### Fundamentación y Referencias Institucionales
    * Verifica y utiliza el contexto institucional recuperado por RAG para respaldar la sugerencia.
    * Si los documentos institucionales mencionan enfoques de Finlandia, Canadá, Singapur, Inglaterra o Corea del Sur, cítalos brevemente indicando nombre de archivo y un extracto.
    * Solo si es pertinente al caso específico (por ejemplo, tareas excesivas, materiales poco estructurados, distracciones), incluye una cita breve de la Teoría de la Carga Cognitiva indicando el tipo y la acción de mitigación. Ejemplos: "Reducir carga extrínseca con guías paso a paso", "Gestionar carga intrínseca segmentando contenidos", "Optimizar carga germana con práctica graduada".

    Alinea el reporte con el rol y las directrices de GEMINI_SYSTEM_ROLE_PROMPT.
    """

    # NUEVO: Prompt específico para el Plan de Intervención (alineado al GEMINI_SYSTEM_ROLE_PROMPT)
    PROMPT_PLAN_INTERVENCION = """
    Basado en el siguiente Reporte 360 para el {tipo_entidad} '{nombre_entidad}':
    ```markdown
    {reporte_360_markdown}
    ```
    Genera un "Plan de Intervención" breve y concreto, claramente estructurado en Markdown (encabezados y listas), y fundamentado en ciencia pedagógica.

    ### Objetivos Claros
    1.  **[Título del Objetivo 1]:** [Descripción breve del objetivo]
    2.  **[Título del Objetivo 2]:** [Descripción breve del objetivo]
    3.  ...

    ### Acciones y Estrategias Sugeridas
    * **[Nombre de la Estrategia 1]:**
        * **Acción:** [Paso concreto y práctico a implementar]
        * **Fundamentación:** [Referencia explícita a principios/teorías pedagógicas (p. ej., carga cognitiva, codificación en memoria de largo plazo, aprendizaje colaborativo), y cita RELEVANTE de documentos institucionales recuperados por RAG (incluye nombre de archivo y breve extracto)]
        * **Cita de Carga Cognitiva (solo si aplica al caso):** [Menciona el tipo (intrínseca, extrínseca, germana) y la acción concreta de mitigación. Ejemplos: "Reducir carga extrínseca eliminando elementos irrelevantes de la guía", "Segmentar contenido para gestionar carga intrínseca", "Diseñar andamiaje para potenciar carga germana"]
    * **[Nombre de la Estrategia 2]:**
        * **Acción:** [Paso concreto y práctico a implementar]
        * **Fundamentación:** [Referencia explícita a teoría/modelo y respaldo institucional por RAG]
        * **Cita de Carga Cognitiva (solo si aplica al caso):** [Tipo y acción concreta, como en el ejemplo anterior]
    * ...

    Alinea el plan con el rol y las directrices de GEMINI_SYSTEM_ROLE_PROMPT.
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
    PROMPT_MAX_CHARS = 36000
    PROMPT_SECTION_CHAR_BUDGETS = {
        'chat_history': 0.08,
        'key_docs_and_qualitative': 0.20,
        'rag_institutional': 0.30,
        'rag_followups': 0.28,
        'historical_quantitative': 0.07,
        'csv_or_base_context': 0.07,
        # Añadido para controlar el presupuesto de señales del feature store dentro del total
        'feature_store_signals': 0.05,
        'external_refs': 0.05
    }
    # Presupuestos máximos específicos para RAG (aplicados además del total)
    RAG_INST_MAX_CHARS = 15000
    RAG_FOLLOWUP_MAX_CHARS = 12000

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
    
    ATTENDANCE_RISK_THRESHOLD = float(os.environ.get('ATTENDANCE_RISK_THRESHOLD', 0.85))
    RISK_THRESHOLDS_BY_LEVEL = {
        'basico': {'attendance': 0.85, 'grade': 4.0},
        'medio': {'attendance': 0.90, 'grade': 4.5}
    }

    ENABLE_PREDICTIVE_MODEL_VIS = False
    MODEL_ARTIFACTS_SUBDIR = 'model_artifacts'

    # Verificación de recursos de apoyo
    STRICT_RESOURCE_VERIFICATION = True
    RESOURCE_VERIFY_TIMEOUT_SEC = 4
    RESOURCE_WHITELIST_DOMAINS = [
        'khanacademy.org', 'youtube.com', 'youtu.be', 'edutopia.org',
        'unesco.org', 'phet.colorado.edu', 'es.khanacademy.org',
        'ineedapencil.com', 'ocw.mit.edu',
        'mineduc.cl', 'gob.cl', 'oecd.org', 'wikipedia.org'
    ]
    RESOURCE_BLACKLIST_DOMAINS = [
        'example.com', 'invalid.com'
    ]
    PREFER_RESOURCE_LANGUAGE = 'es'
    ENABLE_PERPLEXITY_BACKFILL = False
    RESOURCE_LOCAL_BASELINE = [
        {
            'title': 'Aprendo en Línea (Mineduc)',
            'desc': 'Plataforma oficial con materiales y actividades por nivel y asignatura.',
            'candidates': [
                'https://aprendoenlinea.mineduc.cl/',
                'https://www.mineduc.cl/aprendo-en-linea/'
            ]
        },
        {
            'title': 'Biblioteca Digital Escolar (Mineduc)',
            'desc': 'Libros y recursos digitales gratuitos para estudiantes y docentes.',
            'candidates': [
                'https://bdescolar.mineduc.cl/',
                'https://www.mineduc.cl/biblioteca-digital-escolar/'
            ]
        },
        {
            'title': 'Aprendo TV',
            'desc': 'Capsulas audiovisuales educativas para reforzar contenidos escolares.',
            'candidates': [
                'https://aprendoenlinea.mineduc.cl/aprendotv',
                'https://www.mineduc.cl/aprendo-tv/'
            ]
        },
        {
            'title': 'Aprendo FM',
            'desc': 'Programas radiales educativos y material complementario.',
            'candidates': [
                'https://aprendoenlinea.mineduc.cl/aprendofm',
                'https://www.mineduc.cl/aprendo-fm/'
            ]
        },
        {
            'title': 'Plan de Lectoescritura Digital',
            'desc': 'Recursos y estrategias para fortalecer la lectoescritura.',
            'candidates': [
                'https://www.mineduc.cl/lectoescritura-digital/',
                'https://lectoescritura.mineduc.cl/'
            ]
        },
        {
            'title': 'Chile Presente (Seguimiento Estudiantil)',
            'desc': 'Iniciativa para monitorear la trayectoria y asistencia de estudiantes.',
            'candidates': [
                'https://www.mineduc.cl/chile-presente/',
                'https://chilepresente.mineduc.cl/'
            ]
        },
        {
            'title': 'ChatSP (Asistente para Matemáticas)',
            'desc': 'Asistente virtual para docentes de matemáticas.',
            'candidates': [
                'https://www.mineduc.cl/chatsp/',
                'https://chatsp.mineduc.cl/'
            ]
        }
    ]

    ENABLE_EXTERNAL_PEDAGOGY_REFERENCES = True
    EXTERNAL_PEDAGOGY_KEY_TERMS = [
        'modelo finlandes', 'modelo finlandés', 'finlandia',
        'singapur', 'modelo singapur',
        'inglaterra', 'modelo britanico', 'modelo británico',
        'canada', 'canadá',
        'corea del sur', 'corea',
        'carga cognitiva', 'memoria largo plazo'
    ]
    EXTERNAL_PREFERRED_LANGUAGE = 'es'
    EXTERNAL_DOMAIN_PATH_PREFIXES = {
        'oecd.org': ['/education', '/education/', '/education-and-skills', '/education/skills']
    }

    # Autenticación básica
    ENABLE_LOGIN = True
    DEMO_LOGIN_ENABLED = True
    DEMO_LOGIN_USERNAME = 'demo'


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
