#!/usr/bin/env python3
"""Prueba de helpers de presupuesto de prompt.
Valida que el recorte por caracteres funcione correctamente.
"""
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        from mi_aplicacion.app_logic import _trim_to_char_budget

        text = "x" * 100
        trimmed = _trim_to_char_budget(text, 20)
        assert isinstance(trimmed, str)
        assert len(trimmed) >= 20
        assert trimmed.startswith("x" * 20)
        assert "truncado" in trimmed

        trimmed_full = _trim_to_char_budget("hola", 50)
        assert trimmed_full == "hola"

        print("✓ Helper de recorte por presupuesto funciona")
        return 0
    except Exception as e:
        print(f"\n✗ Error en prueba de presupuesto de prompt: {e}")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

