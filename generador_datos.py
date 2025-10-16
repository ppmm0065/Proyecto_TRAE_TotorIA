# generador_datos.py
import pandas as pd
import random
import csv
from faker import Faker

# --- PARÁMETROS DE GENERACIÓN ---
NUM_ESTUDIANTES = 800
ASIGNATURAS = ['Matemáticas', 'Inglés', 'Lenguaje', 'Ciencias', 'Historia', 'Educación Física']
CURSOS = ['1° Básico A', '1° Básico B', '2° Básico A', '2° Básico B', '3° Básico A', '3° Básico B', '4° Básico A', '4° Básico B', '5° Básico A', '5° Básico B', '6° Básico A', '6° Básico B', '7° Básico A', '7° Básico B', '8° Básico A', '8° Básico B', '1° Medio A', '1° Medio B', '2° Medio A', '2° Medio B', '3° Medio A', '3° Medio B', '4° Medio A', '4° Medio B'
]
TIPOS_NOTA = ['Prueba', 'Tarea', 'Ensayo', 'Control', 'Laboratorio', 'Exposición']

OBSERVACIONES_POSITIVAS = [
    "Excelente participación en clases.", "Muestra liderazgo y colabora con sus compañeros.",
    "Ha mejorado notablemente su rendimiento.", "Muy responsable y puntual con sus entregas.",
    "Creativo y con gran potencial."
]
OBSERVACIONES_NEGATIVAS = [
    "Molesta a sus companeros constantemente.", "A menudo no trae sus materiales de trabajo.",
    "Se distrae con facilidad.", "Presenta dificultades para seguir instrucciones.", "Agrede a un compañero", "Falta de respeto con el profesor",
    "Copia en la prueba de un compañero."
]
OBSERVACIONES_COMPLEJAS = [
    "Muestra gran interés, pero a veces conversa en clases.", "Buen rendimiento académico, aunque es muy tímido.",
    "Ha mejorado su conducta, pero aún le cuesta concentrarse.", "Falta de estudio. Se cita al apoderado.",
    "Presenta un progreso inconsistente; alterna entre notas altas y bajas."
]

def generar_datos():
    """
    Genera un archivo CSV con datos ficticios de estudiantes en formato "largo".
    """
    print("Iniciando la generación de datos ficticios...")
    fake = Faker('es_ES')
    
    lista_estudiantes = []
    nombres_usados = set()
    while len(lista_estudiantes) < NUM_ESTUDIANTES:
        nombre = fake.name()
        if nombre not in nombres_usados:
            nombres_usados.add(nombre)
            edad = random.randint(14, 18)
            curso = random.choice(CURSOS)
            asistencia = f"{random.randint(80, 100)}%"
            lista_estudiantes.append({
                'Nombre': nombre,
                'curso': curso,
                'edad': edad,
                'Asistencia': asistencia
            })

    random.shuffle(lista_estudiantes)
    num_complejas = int(NUM_ESTUDIANTES * 0.30)
    
    for i, estudiante in enumerate(lista_estudiantes):
        if i < num_complejas:
            estudiante['Observacion de conducta'] = random.choice(OBSERVACIONES_COMPLEJAS)
        else:
            if random.random() > 0.4:
                 estudiante['Observacion de conducta'] = random.choice(OBSERVACIONES_POSITIVAS)
            else:
                 estudiante['Observacion de conducta'] = random.choice(OBSERVACIONES_NEGATIVAS)

    registros_notas = []
    for estudiante in lista_estudiantes:
        num_asignaturas_estudiante = random.randint(3, len(ASIGNATURAS))
        asignaturas_estudiante = random.sample(ASIGNATURAS, num_asignaturas_estudiante)
        
        for asignatura in asignaturas_estudiante:
            num_notas = random.randint(2, 4)
            for i in range(num_notas):
                nota = round(random.uniform(3.0, 7.0), 1)
                detalle_nota = f"{random.choice(TIPOS_NOTA)} {i + 1}"
                
                registros_notas.append({
                    'Nombre': estudiante['Nombre'],
                    'curso': estudiante['curso'],
                    'edad': estudiante['edad'],
                    'Asignatura': asignatura,
                    'Detalle_Nota': detalle_nota,
                    'Nota': str(nota).replace('.',','), # Usar coma como decimal para consistencia
                    'Asistencia': estudiante['Asistencia'],
                    'Observacion de conducta': estudiante['Observacion de conducta']
                })

    df = pd.DataFrame(registros_notas)
    columnas_ordenadas = [
        'Nombre', 'curso', 'edad', 'Asignatura', 'Detalle_Nota', 
        'Nota', 'Asistencia', 'Observacion de conducta'
    ]
    df = df[columnas_ordenadas]

    nombre_archivo = 'datos_completos_estudiantes.csv'
    
    # --- LÍNEA CORREGIDA ---
    # Forzamos el separador de coma (sep=',') y la codificación utf-8-sig para máxima compatibilidad.
    df.to_csv(nombre_archivo, index=False, sep=',', encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    
    print(f"\n¡Éxito! Se ha generado el archivo '{nombre_archivo}' con formato CSV estándar (separado por comas y codificación UTF-8).")

if __name__ == "__main__":
    generar_datos()