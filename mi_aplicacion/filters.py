from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import re

def ordenar_niveles_educativos(niveles_dict):
    """
    Ordena los niveles educativos: primero todos los Básicos (1°-8°) 
    y luego todos los de Media (1°-4°)
    
    Args:
        niveles_dict: Diccionario de niveles educativos
        
    Returns:
        Lista de tuplas (nivel, data) ordenada
    """
    if not niveles_dict:
        return []
    
    # Separar niveles Básicos y Medios
    basicos = []
    medios = []
    
    for nivel, data in niveles_dict.items():
        if 'Básico' in nivel or 'Basico' in nivel:
            basicos.append((nivel, data))
        elif 'Medio' in nivel:
            medios.append((nivel, data))
    
    # Función para extraer número del nivel
    def extraer_numero(nivel_tupla):
        nivel = nivel_tupla[0]
        match = re.search(r'(\d+)', nivel)
        return int(match.group(1)) if match else 0
    
    # Ordenar por número dentro de cada grupo
    basicos.sort(key=extraer_numero)
    medios.sort(key=extraer_numero)
    
    # Combinar: Básicos primero, luego Medios
    return basicos + medios

def nota_un_decimal(value):
    """
    Formatea una nota a 1 decimal con redondeo hacia arriba en .5 (ROUND_HALF_UP)
    y utiliza coma como separador decimal. Se usa solo para presentación.

    Ejemplos:
    - 5.85 -> "5,9"
    - 5.84 -> "5,8"
    - "N/A" -> "N/A"
    """
    if value is None:
        return "N/A"
    if isinstance(value, str):
        v = value.strip()
        if v.upper() == "N/A":
            return "N/A"
        # Permitir coma como entrada
        v = v.replace(',', '.')
    else:
        v = value

    try:
        d = Decimal(str(v))
    except (InvalidOperation, ValueError):
        # Si no es numérico, devolver tal cual
        return value

    rounded = d.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    s = f"{rounded}"
    # Asegurar exactamente 1 decimal
    if '.' in s:
        int_part, dec_part = s.split('.')
        dec_part = (dec_part + '0')[:1]
        s = f"{int_part}.{dec_part}"
    else:
        s = f"{s}.0"

    return s.replace('.', ',')

def fix_course_characters(course_name):
    if not isinstance(course_name, str):
        return course_name
    import unicodedata, re
    s = course_name
    s = s.replace('ø', '°')
    s = s.replace('\xa0', ' ')
    s = unicodedata.normalize('NFC', s)
    rep = {
        'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
        'Ã±': 'ñ', 'Ã‘': 'Ñ', 'Â°': '°', 'Âª': 'ª'
    }
    for k, v in rep.items():
        if k in s:
            s = s.replace(k, v)
    s = re.sub(r'([AaEeIiOoUu])\u00B4', lambda m: {
        'A': 'Á', 'a': 'á', 'E': 'É', 'e': 'é', 'I': 'Í', 'i': 'í', 'O': 'Ó', 'o': 'ó', 'U': 'Ú', 'u': 'ú'
    }[m.group(1)], s)
    s = re.sub(r'\bB\s*sico\b', 'Básico', s)
    return s
