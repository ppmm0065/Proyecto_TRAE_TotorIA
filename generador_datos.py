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
    print(f"Cargando roster de estudiantes existente desde '{ROSTER_FILE}'...")
    with open(ROSTER_FILE, 'r', encoding='utf-8') as f:
        estudiantes = json.load(f)
    print(f"Se cargaron {len(estudiantes)} estudiantes.")
    
    # Actualizar datos que pueden variar ligeramente (ej. asistencia, profesor)
    for est in estudiantes:
        est["asistencia"] = round(random.uniform(RANGO_ASISTENCIA[0], RANGO_ASISTENCIA[1]), 2)
        est["profesor"] = f"{fake.first_name()} {fake.last_name()}"

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
                    "Entrevistas": "Sin entrevistas registradas."
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