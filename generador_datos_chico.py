import pandas as pd
import random
import datetime
import json
import os
import argparse
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
NUM_ESTUDIANTES = 100
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
PROBABILIDAD_ENTREVISTA = 0.7

MATERIAS_DEBILES_EJEMPLOS = ["Matemáticas", "Lenguaje", "Física", "Inglés"]
RANGO_ASISTENCIA = (0.85, 1.0)
RANGO_EDAD_BASICA = (6, 14)
RANGO_EDAD_MEDIA = (14, 18)

ARCHIVO_SALIDA = None
ROSTER_FILE = "roster_estudiantes.json"

def _generar_lista_promedios_objetivo(count=None):
    print("Generando distribución de promedios objetivo...")
    lista_promedios = []
    total = int(count or NUM_ESTUDIANTES)
    num_bajos = int(total * 0.10)
    num_top = int(total * 0.01)
    num_altos = int(total * 0.25) - num_top
    num_medios = total - num_bajos - num_top - num_altos
    for _ in range(num_bajos):
        lista_promedios.append(random.uniform(3.0, 4.0))
    for _ in range(num_top):
        lista_promedios.append(random.uniform(6.9, 7.0))
    for _ in range(num_altos):
        lista_promedios.append(random.uniform(6.5, 6.89))
    mu_medios = 5.95
    sigma_medios = 0.4
    for _ in range(num_medios):
        prom = random.normalvariate(mu_medios, sigma_medios)
        prom = max(4.1, min(6.4, prom))
        lista_promedios.append(prom)
    print(f"Distribución generada. Media: {sum(lista_promedios) / len(lista_promedios):.2f}")
    random.shuffle(lista_promedios)
    return lista_promedios

def _age_for_course_name(course_name: str) -> int:
    import re
    m = re.match(r"\s*(\d+)\s*°?\s*(Básico|Medio)", str(course_name))
    if not m:
        return 0
    num = int(m.group(1))
    level = m.group(2)
    if level == 'Básico':
        return 6 + num
    return 14 + num

def generate_data(num_estudiantes=None, archivo_salida=None, roster_file=None, seed=None):
    fake = None
    if seed is None:
        random.seed(datetime.datetime.now().timestamp())
    else:
        random.seed(seed)
    datos_completos = []
    N = int(num_estudiantes or NUM_ESTUDIANTES)
    out_csv = str(archivo_salida or f"datos_completos_{N}_estudiantes.csv")
    roster_path = str(roster_file or ROSTER_FILE)
    lista_promedios_base = _generar_lista_promedios_objetivo(N)
    lista_promedios_base_copia = lista_promedios_base.copy()
    estudiantes = []
    if os.path.exists(roster_path):
        print(f"Cargando roster de estudiantes existente desde '{roster_path}'...")
        with open(roster_path, 'r', encoding='utf-8') as f:
            estudiantes = json.load(f)
        print(f"Se cargaron {len(estudiantes)} estudiantes.")
        # Si el tamaño del roster no coincide con N, regenerar roster para respetar N
        if len(estudiantes) != N:
            print(f"Tamaño de roster {len(estudiantes)} distinto de solicitado {N}. Regenerando roster...")
            estudiantes = []
            lista_promedios_para_generar = lista_promedios_base.copy()
            id_estudiante_actual = 1
            niveles = list(CURSOS_GRADOS.keys())
            total_niveles = len(niveles)
            base_por_nivel = N // total_niveles
            resto = N % total_niveles
            distrib_niveles = {niv: base_por_nivel + (i < resto) for i, niv in enumerate(niveles)}
            for grado in niveles:
                letras = CURSOS_GRADOS[grado]
                count_nivel = distrib_niveles[grado]
                a_count = (count_nivel + 1) // 2
                b_count = count_nivel - a_count
                paralelos_counts = {}
                if len(letras) >= 1:
                    paralelos_counts[letras[0]] = a_count
                if len(letras) >= 2:
                    paralelos_counts[letras[1]] = b_count
                for letra, est_por_paralelo in paralelos_counts.items():
                    curso_completo = f"{grado} {letra}"
                    for _ in range(est_por_paralelo):
                        if not lista_promedios_para_generar:
                            lista_promedios_para_generar.append(random.uniform(4.0, 5.0))
                        nombre = f"{fake_first()} {fake_last()} {fake_last()}"
                        asignaturas = ASIGNATURAS_POR_NIVEL["Básico"] if "Básico" in grado else ASIGNATURAS_POR_NIVEL["Medio"]
                        edad = _age_for_course_name(grado)
                        estudiante_info = {
                            "id_estudiante": id_estudiante_actual,
                            "nombre": nombre,
                            "curso": curso_completo,
                            "edad": edad,
                            "asignaturas": asignaturas,
                            "materias_debiles": random.choice(MATERIAS_DEBILES_EJEMPLOS) if random.random() < 0.3 else "",
                            "asistencia": round(random.uniform(RANGO_ASISTENCIA[0], RANGO_ASISTENCIA[1]), 2),
                            "profesor": f"{fake_first()} {fake_last()}",
                            "Familia": f"Apoderado: {fake_name()}",
                            "Entrevistas": random.choice(ENTREVISTAS_EJEMPLOS) if random.random() < PROBABILIDAD_ENTREVISTA else "Sin entrevistas registradas.",
                            "promedio_objetivo": round(lista_promedios_para_generar.pop(), 2)
                        }
                        estudiantes.append(estudiante_info)
                        id_estudiante_actual += 1
            try:
                with open(roster_path, 'w', encoding='utf-8') as f:
                    json.dump(estudiantes, f, ensure_ascii=False, indent=4)
            except Exception as e:
                print(f"Error al guardar el roster: {e}")
        else:
            hubo_actualizacion = False
            random.shuffle(lista_promedios_base_copia)
            for i, est in enumerate(estudiantes):
                if "asistencia" not in est or est["asistencia"] in (None, ""):
                    est["asistencia"] = round(random.uniform(RANGO_ASISTENCIA[0], RANGO_ASISTENCIA[1]), 2)
                if "profesor" not in est or not str(est["profesor"]).strip():
                    est["profesor"] = f"{fake_first()} {fake_last()}"
                if "nombre" not in est or not str(est["nombre"]).strip():
                    est["nombre"] = f"{fake_first()} {fake_last()} {fake_last()}"
                if "Familia" not in est or not str(est["Familia"]).strip():
                    est["Familia"] = f"Apoderado: {fake_name()}"
                try:
                    parts = (est.get("curso","") or '').split()
                    grado_key = ' '.join(parts[:2]) if len(parts) >= 2 else ''
                    est["edad"] = _age_for_course_name(grado_key)
                except Exception:
                    pass
                hubo_actualizacion = True
                if "Entrevistas" not in est or est["Entrevistas"] == "Sin entrevistas registradas.":
                    est["Entrevistas"] = random.choice(ENTREVISTAS_EJEMPLOS) if random.random() < PROBABILIDAD_ENTREVISTA else "Sin entrevistas registradas."
                    hubo_actualizacion = True
                if "promedio_objetivo" not in est:
                    est["promedio_objetivo"] = round(lista_promedios_base_copia[i], 2)
                    hubo_actualizacion = True
            if hubo_actualizacion:
                print("Roster actualizado. Guardando cambios...")
                try:
                    with open(roster_path, 'w', encoding='utf-8') as f:
                        json.dump(estudiantes, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f"Error al re-guardar el roster: {e}")
    else:
        print(f"No se encontró roster. Generando {N} estudiantes base nuevos...")
        lista_promedios_para_generar = lista_promedios_base.copy()
        id_estudiante_actual = 1
        niveles = list(CURSOS_GRADOS.keys())
        total_niveles = len(niveles)
        base_por_nivel = N // total_niveles
        resto = N % total_niveles
        distrib_niveles = {niv: base_por_nivel + (i < resto) for i, niv in enumerate(niveles)}
        for grado in niveles:
            letras = CURSOS_GRADOS[grado]
            count_nivel = distrib_niveles[grado]
            # distribuir entre paralelos lo más parejo posible
            a_count = (count_nivel + 1) // 2
            b_count = count_nivel - a_count
            paralelos_counts = {}
            if len(letras) >= 1:
                paralelos_counts[letras[0]] = a_count
            if len(letras) >= 2:
                paralelos_counts[letras[1]] = b_count
            for letra, est_por_paralelo in paralelos_counts.items():
                curso_completo = f"{grado} {letra}"
                for _ in range(est_por_paralelo):
                    if not lista_promedios_para_generar:
                        lista_promedios_para_generar.append(random.uniform(4.0, 5.0))
                    nombre = f"{fake_first()} {fake_last()} {fake_last()}"
                    asignaturas = ASIGNATURAS_POR_NIVEL["Básico"] if "Básico" in grado else ASIGNATURAS_POR_NIVEL["Medio"]
                    edad = _age_for_course_name(grado)
                    estudiante_info = {
                        "id_estudiante": id_estudiante_actual,
                        "nombre": nombre,
                        "curso": curso_completo,
                        "edad": edad,
                        "asignaturas": asignaturas,
                        "materias_debiles": random.choice(MATERIAS_DEBILES_EJEMPLOS) if random.random() < 0.3 else "",
                        "asistencia": round(random.uniform(RANGO_ASISTENCIA[0], RANGO_ASISTENCIA[1]), 2),
                        "profesor": f"{fake_first()} {fake_last()}",
                        "Familia": f"Apoderado: {fake_name()}",
                        "Entrevistas": random.choice(ENTREVISTAS_EJEMPLOS) if random.random() < PROBABILIDAD_ENTREVISTA else "Sin entrevistas registradas.",
                        "promedio_objetivo": round(lista_promedios_para_generar.pop(), 2)
                    }
                    estudiantes.append(estudiante_info)
                    id_estudiante_actual += 1
        try:
            with open(roster_path, 'w', encoding='utf-8') as f:
                json.dump(estudiantes, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error al guardar el roster: {e}")
    print("Generando datos de asignaturas, notas y observaciones...")
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
                "edad": est.get("edad"),
                "Asignatura": r["Asignatura"],
                "Nota": r["Nota"],
                "Observacion de conducta": r["Observacion de conducta"],
                "materias_debiles": est["materias_debiles"],
                "Asistencia": asistencia_run,
                "profesor": est["profesor"],
                "Familia": est["Familia"],
                "Entrevistas": est["Entrevistas"]
            })
    print(f"Creando DataFrame y guardando en '{out_csv}'...")
    df = pd.DataFrame(datos_completos)
    columnas_ordenadas = [
        "ID Estudiante", "Nombre", "curso", "edad", "Asignatura", "Nota",
        "Observacion de conducta", "materias_debiles", "Asistencia",
        "profesor", "Familia", "Entrevistas"
    ]
    df = df[columnas_ordenadas]
    df.to_csv(out_csv, index=False, sep=';', encoding='utf-8-sig')
    print("¡Proceso completado! Archivo CSV generado exitosamente.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--estudiantes", type=int, default=NUM_ESTUDIANTES)
    parser.add_argument("--salida", type=str, default=None)
    parser.add_argument("--roster", type=str, default=ROSTER_FILE)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    generate_data(args.estudiantes, args.salida, args.roster, args.seed)
