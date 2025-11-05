import pandas as pd
import random
from faker import Faker
import datetime
import json
import os

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

# --- INICIALIZACIÓN ---
fake = Faker('es_ES') # Nombres en español

# --- LÍNEA AÑADIDA ---
# Forzamos una nueva semilla aleatoria basada en la hora actual
random.seed(datetime.datetime.now().timestamp())
print("Nueva semilla aleatoria inicializada.")

datos_completos = []

# --- GENERACIÓN DE ROSTER DE ESTUDIANTES (LÓGICA MEJORADA) ---
estudiantes = []
if os.path.exists(ROSTER_FILE):
    # --- INICIO BLOQUE MODIFICADO (FIX 1) ---
    # (Reemplace este bloque 'if' completo, aprox. líneas 81-90)
    print(f"Cargando roster de estudiantes existente desde '{ROSTER_FILE}'...")
    with open(ROSTER_FILE, 'r', encoding='utf-8') as f:
        estudiantes = json.load(f)
    print(f"Se cargaron {len(estudiantes)} estudiantes.")
    
    # Actualizar datos que pueden variar (ej. asistencia, profesor)
    # Y ACTUALIZAR ENTREVISTAS SI FALTAN O SON ANTIGUAS
    print("Actualizando roster existente con nueva lógica de entrevistas...")
    hubo_actualizacion_entrevista = False
    for est in estudiantes:
        est["asistencia"] = round(random.uniform(RANGO_ASISTENCIA[0], RANGO_ASISTENCIA[1]), 2)
        est["profesor"] = f"{fake.first_name()} {fake.last_name()}"
        
        # Si el estudiante cargado no tiene entrevista (o tiene la antigua), se la asignamos.
        if "Entrevistas" not in est or est["Entrevistas"] == "Sin entrevistas registradas.":
            est["Entrevistas"] = random.choice(ENTREVISTAS_EJEMPLOS) if random.random() < PROBABILIDAD_ENTREVISTA else "Sin entrevistas registradas."
            hubo_actualizacion_entrevista = True

    # Si actualizamos entrevistas, debemos re-guardar el roster
    if hubo_actualizacion_entrevista:
        print("Roster actualizado. Guardando cambios en 'roster_estudiantes.json'...")
        try:
            with open(ROSTER_FILE, 'w', encoding='utf-8') as f:
                json.dump(estudiantes, f, ensure_ascii=False, indent=4)
            print("Roster guardado con éxito.")
        except Exception as e:
            print(f"Error al re-guardar el roster: {e}")

else:

    print(f"No se encontró roster. Generando {NUM_ESTUDIANTES} estudiantes base nuevos...")
    id_estudiante_actual = 1
    for grado, letras in CURSOS_GRADOS.items():
        for letra in letras:
            curso_completo = f"{grado} {letra}"
            est_por_curso = round(NUM_ESTUDIANTES / len(CURSOS_GRADOS.keys()) / len(letras))
            
            for _ in range(est_por_curso):
                nombre = f"{fake.first_name()} {fake.last_name()} {fake.last_name()}"
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
                    "profesor": f"{fake.first_name()} {fake.last_name()}",
                    "Familia": f"Apoderado: {fake.name()}",
                    # --- LÍNEA MODIFICADA (FIX 2) ---
                    "Entrevistas": random.choice(ENTREVISTAS_EJEMPLOS) if random.random() < PROBABILIDAD_ENTREVISTA else "Sin entrevistas registradas."
                }
                estudiantes.append(estudiante_info)
                id_estudiante_actual += 1

    print(f"Se generaron {len(estudiantes)} estudiantes nuevos.")
    # Guardar el roster para futuras ejecuciones
    try:
        with open(ROSTER_FILE, 'w', encoding='utf-8') as f:
            json.dump(estudiantes, f, ensure_ascii=False, indent=4)
        print(f"Roster de estudiantes guardado en '{ROSTER_FILE}'.")
    except Exception as e:
        print(f"Error al guardar el roster: {e}")

# --- GENERACIÓN DE DATOS (NOTAS Y OBSERVACIONES) ---
print("Generando datos de asignaturas, notas y observaciones (nuevos)...")
for est in estudiantes:
    for asignatura in est["asignaturas"]:
        # Simular una leve mejoría o empeoramiento general
        factor_aleatorio = random.uniform(-0.5, 0.5)
        nota_base = 4.5 + factor_aleatorio # Centrado en 4.5
        
        # Ajustar nota si es materia débil
        if est["materias_debiles"] and est["materias_debiles"] in asignatura:
            nota_base -= 1.0
            
        # Asegurar que la nota esté en el rango
        nota = round(random.uniform(nota_base - 1.0, nota_base + 1.0), 1)
        nota = max(RANGO_NOTAS[0], min(RANGO_NOTAS[1], nota)) # Clamp
        
        # Generar observación
        observacion = ""
        if random.random() < PROBABILIDAD_OBSERVACION:
            if nota < 4.0:
                observacion = random.choice(OBSERVACIONES_NEGATIVAS)
            elif nota > 6.0:
                observacion = random.choice(OBSERVACIONES_POSITIVAS)
            elif random.random() < 0.5:
                 observacion = random.choice(OBSERVACIONES_NEGATIVAS)
            else:
                 observacion = random.choice(OBSERVACIONES_POSITIVAS)
        
        registro = {
            "ID Estudiante": est["id_estudiante"],
            "Nombre": est["nombre"],
            "curso": est["curso"],
            "edad": est["edad"],
            "Asignatura": asignatura,
            "Nota": nota,
            "Observacion de conducta": observacion,
            "materias_debiles": est["materias_debiles"],
            "Asistencia": est["asistencia"],
            "profesor": est["profesor"],
            "Familia": est["Familia"],
            "Entrevistas": est["Entrevistas"]
        }
        datos_completos.append(registro)

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