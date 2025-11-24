import pandas as pd
import random
import datetime
import json
import os
CHILE_FIRST_NAMES = [
    'María','Alicia','Claudia','Marisol','Lucía','Francisca','Paola','Paula','Pedro','Pablo',
    'Francisco','Julio','Esteban','Mario','Ana','Camila','Sofía','Valentina','José','Felipe',
    'Daniela','Andrea','Martina','Isabella','Diego','Carlos','Miguel','Luis','Javiera','Constanza',
    'Sebastián','Matías','Nicolás','Tomás','Benjamín','Antonia','Catalina','Fernanda','Ignacio','Gabriela'
]
CHILE_LAST_NAMES = [
    'Aguirre','Baltra','Miranda','Morales','Gutiérrez','Arteaga','Márquez','Prieto','López','González',
    'Muñoz','Rojas','Díaz','Pérez','Soto','Silva','Contreras','Sepúlveda','Herrera','Araya',
    'Vega','Tapia','Campos','Ramírez','Reyes','Castro','Hidalgo','Figueroa','Valenzuela','Pizarro',
    'Ibarra','Saavedra','Ortega','Riquelme','Escobar','Carrasco','Leiva','Romero','Vásquez','Salazar'
]
def fake_first():
    return random.choice(CHILE_FIRST_NAMES)
def fake_last():
    return random.choice(CHILE_LAST_NAMES)
def fake_name():
    return f"{fake_first()} {fake_last()}"

# --- CONFIGURACIÓN ---
NUM_ESTUDIANTES = 800
CURSOS_GRADOS = {
    "1° Básico": ["A", "B"], "2° Básico": ["A", "B"], "3° Básico": ["A", "B"],
    "4° Básico": ["A", "B"], "5° Básico": ["A", "B"], "6° Básico": ["A", "B"],
    "7° Básico": ["A", "B"], "8° Básico": ["A", "B"],
    "1° Medio": ["A", "B"], "2° Medio": ["A", "B"], "3° Medio": ["A", "B"], "4° Medio": ["A", "B"]
}
ASIGNATURAS_POR_NIVEL = {
    "Básico": ["Lenguaje", "Matemáticas", "Ciencias Naturales", "Historia", "Inglés", "Música", "Artes Visuales", "Educación Física"],
    "Medio": ["Lenguaje", "Matemáticas", "Física", "Química", "Biología", "Historia", "Inglés", "Filosofía", "Artes Visuales", "Educación Física"]
}
RANGO_NOTAS = (2.0, 7.0)
PROBABILIDAD_OBSERVACION = 0.4
OBSERVACIONES_POSITIVAS = [
    "Excelente participación en clases.", "Demuestra gran interés por la asignatura.",
    "Colabora activamente con sus compañeros.", "Muy responsable y puntual con sus entregas.",
    "Creativo y con gran potencial.", "Ha mejorado notablemente su rendimiento.",
    "Liderazgo positivo dentro del grupo."
]
OBSERVACIONES_NEGATIVAS = [
    "Falta de estudio y preparación para las evaluaciones.", "Dificultad para concentrarse en clases.",
    "Progreso inconsistente, alterna entre notas altas y bajas.",
    "Molesta a sus compañeros e interrumpe la clase.", "Falta de respeto hacia el profesor.",
    "Se distrae con facilidad.", "Tímido, le cuesta participar oralmente.",
    "Registra agresiones físicas a compañeros.", "Copia en la prueba."
]

# Lista de ejemplos para la columna Entrevistas
ENTREVISTAS_EJEMPLOS = [
    "Apoderado: Se observa con los apoderados una mejora en la intención de estudios.",
    "Apoderado: Padres preocupados porque se duerme muy tarde jugando en el computador.",
    "Apoderado: Padres manifiestan que están viviendo situación de separación que ha afectado a sus hijos.",
    "Los apoderados informan que están en un proceso de separación y que eso ha afectado a sus hijos.",
    "Los apoderados agradecen la actividad pastoral de la semana pasada.",
    "El colegio hace saber a los apoderados la preocupación por el mal comportamiento de su hijo.",
    "Los apoderados solicitan al colegio que su hijo sea incorporado a la selección de futbol.",
    "Se conversa con apoderado sobre la importancia de reforzar hábitos de estudio en casa.",
    "Apoderado justifica inasistencias por viaje familiar, presenta justificativo médico.",
    "Se felicita al apoderado por el notable avance del estudiante en conducta y responsabilidad.",
    "Apoderado informa que el estudiante presenta dificultades de concentración en casa y solicita estrategias de apoyo.",
    "Se cita a apoderado por reiteradas faltas de respeto a inspectores. Apoderado se compromete a conversar con su hijo.",
    "Apoderado informa diagnóstico reciente de TDAH del estudiante. Se acuerda derivar a equipo PIE para evaluación de apoyos.",
    "Se destaca al apoderado el excelente desempeño del estudiante en el debate de Historia, mostrando gran liderazgo.",
    "Apoderado consulta sobre el proceso de postulación a becas para el próximo año. Se entrega información y fechas relevantes.",
    "Conversación sobre la integración social del estudiante. Apoderados mencionan que es tímido; se acuerda fomentar participación en clases.",
    "Se informa al apoderado sobre la falta recurrente de entrega de tareas. Apoderado se compromete a revisar la agenda escolar diariamente.",
    "Apoderados expresan preocupación por posible situación de acoso escolar. Convivencia Escolar iniciará protocolo de investigación.",
    "Apoderado solicita reunión con UTP para revisar la cobertura curricular y los métodos de evaluación de la asignatura de Inglés.",
    "Reunión de seguimiento con apoderado. Se revisan los compromisos de la entrevista anterior y se constatan avances positivos en la conducta."
]
PROBABILIDAD_ENTREVISTA = 0.7 # 70% de los estudiantes tendrán un registro de entrevista

MATERIAS_DEBILES_EJEMPLOS = ["Matemáticas", "Lenguaje", "Física", "Inglés"]
RANGO_ASISTENCIA = (0.85, 1.0)
RANGO_EDAD_BASICA = (6, 14)
RANGO_EDAD_MEDIA = (14, 18)

ARCHIVO_SALIDA = "datos_completos_800_estudiantes.csv"
ROSTER_FILE = "roster_estudiantes.json"

def _generar_lista_promedios_objetivo():
    """
    Genera una lista de promedios objetivo forzando la distribución
    solicitada por el usuario (media ~5.9, percentiles específicos).
    """
    print("Generando distribución de promedios objetivo...")
    
    lista_promedios = []
    
    # 1. Calcular el número de estudiantes en cada tramo
    num_bajos = int(NUM_ESTUDIANTES * 0.10) # 10%
    num_top = int(NUM_ESTUDIANTES * 0.01)   # 1%
    num_altos = int(NUM_ESTUDIANTES * 0.25) - num_top # 24% (25% total - 1% top)
    num_medios = NUM_ESTUDIANTES - num_bajos - num_top - num_altos # 65%
    
    # 2. Generar promedios para cada tramo
    
    # Tramo Bajo (10%): <= 4.0
    for _ in range(num_bajos):
        lista_promedios.append(random.uniform(3.0, 4.0))
        
    # Tramo Top (1%): 6.9-7.0
    for _ in range(num_top):
        lista_promedios.append(random.uniform(6.9, 7.0))
        
    # Tramo Alto (24%): 6.5-6.89
    for _ in range(num_altos):
        lista_promedios.append(random.uniform(6.5, 6.89))
        
    # Tramo Medio (65%): 4.1 - 6.4 (Gaussiano)
    # Para que la media global sea 5.9, la media de este grupo debe ser ~5.95
    mu_medios = 5.95
    sigma_medios = 0.4 # Desviación para cubrir el rango (4.1 a 6.4)
    
    for _ in range(num_medios):
        prom = random.normalvariate(mu_medios, sigma_medios)
        # Forzar a los límites 4.1 y 6.4
        prom = max(4.1, min(6.4, prom))
        lista_promedios.append(prom)

    # 3. Barajar la lista y retornarla
    print(f"Distribución generada. Media: {sum(lista_promedios) / len(lista_promedios):.2f}")
    random.shuffle(lista_promedios)
    return lista_promedios

# --- INICIALIZACIÓN ---
fake = None

# --- LÍNEA AÑADIDA ---
# Forzamos una nueva semilla aleatoria basada en la hora actual
random.seed(datetime.datetime.now().timestamp())
print("Nueva semilla aleatoria inicializada.")

datos_completos = []

# 1. Generar la lista de promedios base ANTES de cargar o crear el roster
lista_promedios_base = _generar_lista_promedios_objetivo()
lista_promedios_base_copia = lista_promedios_base.copy() # Copia para el bloque IF

estudiantes = []
if os.path.exists(ROSTER_FILE):
    print(f"Cargando roster de estudiantes existente desde '{ROSTER_FILE}'...")
    with open(ROSTER_FILE, 'r', encoding='utf-8') as f:
        estudiantes = json.load(f)
    print(f"Se cargaron {len(estudiantes)} estudiantes.")
    
    # Flags para saber si hay que re-guardar
    hubo_actualizacion = False

    # Si el número de estudiantes en el roster no coincide, advertir.
    if len(estudiantes) != len(lista_promedios_base_copia):
        print(f"ADVERTENCIA: Roster tiene {len(estudiantes)} est. vs {len(lista_promedios_base_copia)} promedios generados.")
        # Ajustar lista de promedios por si acaso
        while len(lista_promedios_base_copia) < len(estudiantes):
            lista_promedios_base_copia.append(random.uniform(4.0, 5.0)) # Relleno
        lista_promedios_base_copia = lista_promedios_base_copia[:len(estudiantes)]
        
    print("Actualizando roster existente con nueva lógica (Entrevistas y Promedio Objetivo)...")
    
    # Mezclar la lista de promedios antes de asignar a estudiantes existentes
    random.shuffle(lista_promedios_base_copia)
    
    for i, est in enumerate(estudiantes):
        # Preservar identidad del estudiante (nombre, apoderado y profesor) y su asistencia
        # Solo se actualizarán notas, observaciones y entrevistas en los registros generados más abajo
        if "asistencia" not in est or est["asistencia"] in (None, ""):
            est["asistencia"] = round(random.uniform(RANGO_ASISTENCIA[0], RANGO_ASISTENCIA[1]), 2)
        if "profesor" not in est or not str(est["profesor"]).strip():
            est["profesor"] = f"{fake_first()} {fake_last()}"
        if "nombre" not in est or not str(est["nombre"]).strip():
            est["nombre"] = f"{fake_first()} {fake_last()} {fake_last()}"
        if "Familia" not in est or not str(est["Familia"]).strip():
            est["Familia"] = f"Apoderado: {fake_name()}"
        hubo_actualizacion = True
        
        # FIX Entrevistas (de la vez anterior)
        if "Entrevistas" not in est or est["Entrevistas"] == "Sin entrevistas registradas.":
            est["Entrevistas"] = random.choice(ENTREVISTAS_EJEMPLOS) if random.random() < PROBABILIDAD_ENTREVISTA else "Sin entrevistas registradas."
            hubo_actualizacion = True
            
        # NUEVA LÓGICA: Añadir Promedio Objetivo si no existe
        if "promedio_objetivo" not in est:
            est["promedio_objetivo"] = round(lista_promedios_base_copia[i], 2)
            hubo_actualizacion = True

    if hubo_actualizacion:
        print("Roster actualizado. Guardando cambios en 'roster_estudiantes.json'...")
        try:
            with open(ROSTER_FILE, 'w', encoding='utf-8') as f:
                json.dump(estudiantes, f, ensure_ascii=False, indent=4)
            print("Roster guardado con éxito.")
        except Exception as e:
            print(f"Error al re-guardar el roster: {e}")

else:
    # (El roster no existe, se genera de cero)
    print(f"No se encontró roster. Generando {NUM_ESTUDIANTES} estudiantes base nuevos...")
    
    # Usar la lista de promedios generada al inicio
    lista_promedios_para_generar = lista_promedios_base.copy()
    
    id_estudiante_actual = 1
    for grado, letras in CURSOS_GRADOS.items():
        for letra in letras:
            curso_completo = f"{grado} {letra}"
            est_por_curso = round(NUM_ESTUDIANTES / len(CURSOS_GRADOS.keys()) / len(letras))
            
            for _ in range(est_por_curso):
                if not lista_promedios_para_generar: # Seguridad por si el redondeo falla
                    print("Agotada lista de promedios, rellenando...")
                    lista_promedios_para_generar.append(random.uniform(4.0, 5.0))
                    
                nombre = f"{fake_first()} {fake_last()} {fake_last()}"
                
                if "Básico" in grado:
                    edad = random.randint(RANGO_EDAD_BASICA[0], RANGO_EDAD_BASICA[1])
                    asignaturas = ASIGNATURAS_POR_NIVEL["Básico"]
                else:
                    edad = random.randint(RANGO_EDAD_MEDIA[0], RANGO_EDAD_MEDIA[1])
                    asignaturas = ASIGNATURAS_POR_NIVEL["Medio"]
                
                estudiante_info = {
                    "id_estudiante": id_estudiante_actual,
                    "nombre": nombre,
                    "curso": curso_completo,
                    "edad": edad,
                    "asignaturas": asignaturas,
                    "materias_debiles": random.choice(MATERIAS_DEBILES_EJEMPLOS) if random.random() < 0.3 else "", # 30% tiene una materia débil
                    "asistencia": round(random.uniform(RANGO_ASISTENCIA[0], RANGO_ASISTENCIA[1]), 2),
                    "profesor": f"{fake_first()} {fake_last()}",
                    "Familia": f"Apoderado: {fake_name()}",
                    "Entrevistas": random.choice(ENTREVISTAS_EJEMPLOS) if random.random() < PROBABILIDAD_ENTREVISTA else "Sin entrevistas registradas.",
                    # --- LÍNEA AÑADIDA ---
                    "promedio_objetivo": round(lista_promedios_para_generar.pop(), 2)
                }
                estudiantes.append(estudiante_info)
                id_estudiante_actual += 1

    print(f"Se generaron {len(estudiantes)} estudiantes nuevos.")
    try:
        with open(ROSTER_FILE, 'w', encoding='utf-8') as f:
            json.dump(estudiantes, f, ensure_ascii=False, indent=4)
        print(f"Roster de estudiantes guardado en '{ROSTER_FILE}'.")
    except Exception as e:
        print(f"Error al guardar el roster: {e}")

# --- GENERACIÓN DE DATOS (NOTAS Y OBSERVACIONES) ---
print("Generando datos de asignaturas, notas y observaciones (nuevos)...")
for est in estudiantes:
    promedio_obj = est["promedio_objetivo"]
    texto_ent = str(est.get("Entrevistas", "")).lower()
    kw = ["separacion","violencia","maltrato","abuso","tdah","acoso","falta de respeto","preocupacion","diagnostico","protocolo"]
    es_familia_compleja = any(k in texto_ent for k in kw)
    base_att = est.get("asistencia")
    try:
        base_att = float(base_att) if base_att is not None else None
    except Exception:
        base_att = None
    if base_att is None:
        base_att = round(random.uniform(RANGO_ASISTENCIA[0], RANGO_ASISTENCIA[1]), 2)
    drift = random.uniform(-0.08, 0.08)
    asistencia_run = max(RANGO_ASISTENCIA[0], min(RANGO_ASISTENCIA[1], round(base_att + drift, 2)))
    est["asistencia"] = asistencia_run
    registros_estudiante = []
    for asignatura in est["asignaturas"]:
        nota_base = promedio_obj
        variacion_asignatura = random.uniform(-0.8, 0.8)
        nota = nota_base + variacion_asignatura
        if est["materias_debiles"] and est["materias_debiles"] in asignatura:
            nota -= 1.0
        if es_familia_compleja:
            nota -= 0.3
        nota = round(nota, 1)
        nota = max(RANGO_NOTAS[0], min(RANGO_NOTAS[1], nota))
        observacion = ""
        if random.random() < PROBABILIDAD_OBSERVACION:
            if nota < 4.0:
                observacion = random.choice(OBSERVACIONES_NEGATIVAS)
            elif nota > 6.0:
                observacion = random.choice(OBSERVACIONES_POSITIVAS)
            else:
                pos_prob = 0.5
                if promedio_obj >= 6.0:
                    pos_prob = 0.85
                elif promedio_obj <= 4.2:
                    pos_prob = 0.15
                observacion = random.choice(OBSERVACIONES_POSITIVAS) if random.random() < pos_prob else random.choice(OBSERVACIONES_NEGATIVAS)
        registros_estudiante.append({"Asignatura": asignatura, "Nota": nota, "Observacion de conducta": observacion})
    if promedio_obj < 5.0:
        idxs_bajo = [i for i, r in enumerate(registros_estudiante) if r["Nota"] < 4.0]
        n = len(registros_estudiante)
        objetivo = random.randint(1, 3)
        if objetivo >= n:
            objetivo = min(3, max(1, n - 1))
        if len(idxs_bajo) == 0:
            seleccion = random.sample(range(n), objetivo)
            for i in seleccion:
                registros_estudiante[i]["Nota"] = round(random.uniform(3.0, 3.9), 1)
        elif len(idxs_bajo) > objetivo:
            subir = len(idxs_bajo) - objetivo
            a_subir = random.sample(idxs_bajo, subir)
            for i in a_subir:
                registros_estudiante[i]["Nota"] = round(random.uniform(4.0, 4.5), 1)
        elif len(idxs_bajo) == n:
            ajustar = max(1, n - objetivo)
            a_ajustar = random.sample(range(n), ajustar)
            for i in a_ajustar:
                registros_estudiante[i]["Nota"] = round(random.uniform(4.0, 4.5), 1)
    if all(r["Nota"] < 4.0 for r in registros_estudiante):
        j = random.randrange(len(registros_estudiante))
        registros_estudiante[j]["Nota"] = round(random.uniform(4.0, 4.5), 1)
    for r in registros_estudiante:
        datos_completos.append({
            "ID Estudiante": est["id_estudiante"],
            "Nombre": est["nombre"],
            "curso": est["curso"],
            "edad": est["edad"],
            "Asignatura": r["Asignatura"],
            "Nota": r["Nota"],
            "Observacion de conducta": r["Observacion de conducta"],
            "materias_debiles": est["materias_debiles"],
            "Asistencia": asistencia_run,
            "profesor": est["profesor"],
            "Familia": est["Familia"],
            "Entrevistas": est["Entrevistas"]
        })

# --- CREACIÓN Y GUARDADO DEL DATAFRAME ---
print(f"Creando DataFrame y guardando en '{ARCHIVO_SALIDA}'...")
df = pd.DataFrame(datos_completos)

# Reordenar columnas para que las principales queden al inicio
columnas_ordenadas = [
    "ID Estudiante", "Nombre", "curso", "edad", "Asignatura", "Nota", 
    "Observacion de conducta", "materias_debiles", "Asistencia", 
    "profesor", "Familia", "Entrevistas"
]
df = df[columnas_ordenadas]

# Guardar en CSV
df.to_csv(ARCHIVO_SALIDA, index=False, sep=';', encoding='utf-8-sig')

print("¡Proceso completado! Archivo CSV generado exitosamente.")
