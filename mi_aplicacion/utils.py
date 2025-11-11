import re
import unicodedata
import pytz
from pytz import timezone
try:
    # current_app solo disponible dentro de contexto Flask; protegemos import.
    from flask import current_app
except Exception:
    current_app = None


def normalize_text(s: str) -> str:
    """Normalize text for robust matching: to lowercase, strip accents, collapse whitespace.

    - Converts to string
    - Unicode NFKD decomposition and removes diacritics
    - Lowercases
    - Collapses multiple spaces to single space
    """
    if s is None:
        return ""
    s = str(s)
    # Decompose and remove diacritics
    nfkd = unicodedata.normalize("NFKD", s)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase
    lowered = no_accents.lower()
    # Collapse whitespace
    normalized = " ".join(lowered.strip().split())
    return normalized


def compile_any_keyword_pattern(keywords):
    """Compile a regex that matches if any of the provided keywords appear in the text.

    Keywords are normalized (using normalize_text) and escaped for regex safety.
    Returns a compiled regex pattern.
    """
    norm_escaped = [re.escape(normalize_text(k)) for k in keywords if k]
    if not norm_escaped:
        # Match nothing
        return re.compile(r"^$")
    joined = "|".join(norm_escaped)
    # Use non-capturing group; text will already be normalized to lowercase/ascii
    return re.compile(f"(?:{joined})")


def get_tz():
    """Devuelve la zona horaria configurada.

    - Lee TIMEZONE_NAME desde config de Flask si hay contexto activo.
    - Si no hay contexto, usa 'America/Santiago' por defecto.
    - Retorna objeto pytz.timezone.
    """
    try:
        if current_app is not None:
            tz_name = current_app.config.get('TIMEZONE_NAME', 'America/Santiago')
        else:
            tz_name = 'America/Santiago'
        return timezone(tz_name)
    except Exception:
        # Fallback robusto
        return timezone('America/Santiago')


def grade_to_qualitative(value):
    """Devuelve la interpretación cualitativa de una nota o promedio.

    Reglas pedidas:
    - 1.0 a 3.0   => "Muy Malo"
    - 3.1 a 4.0   => "Malo"
    - 4.1 a 5.0   => "Regular"
    - 5.1 a 6.0   => "Bueno"
    - 6.1 a 7.0   => "Muy bueno"

    Para evitar ambigüedades de punto flotante, usamos estos umbrales:
    - <= 3.0       => "Muy Malo"
    - > 3.0 y <= 4.0  => "Malo"
    - > 4.0 y <= 5.0  => "Regular"
    - > 5.0 y <= 6.0  => "Bueno"
    - > 6.0 y <= 7.0  => "Muy bueno"

    Si el valor es None o fuera del rango esperado, retorna None.
    """
    try:
        if value is None:
            return None
        n = float(value)
    except Exception:
        return None

    if n < 1.0 or n > 7.0:
        return None
    if n <= 3.0:
        return "Muy Malo"
    if n <= 4.0:
        return "Malo"
    if n <= 5.0:
        return "Regular"
    if n <= 6.0:
        return "Bueno"
    return "Muy bueno"
