from decimal import Decimal, ROUND_HALF_UP, InvalidOperation

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

