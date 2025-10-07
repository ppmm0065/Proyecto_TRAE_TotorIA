import os
import google.generativeai as genai
from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env (¡la línea clave!)
load_dotenv()

# Ahora os.getenv encontrará tu clave automáticamente
api_key = os.getenv('GEMINI_API_KEY')

if not api_key:
    print("Error: No se encontró la GEMINI_API_KEY en tu archivo .env")
    print("Asegúrate de que el archivo .env está en la carpeta principal y contiene la línea: GEMINI_API_KEY='tu_clave_aqui'")
else:
    try:
        genai.configure(api_key=api_key)
        print("API Key encontrada. Buscando modelos disponibles...\n")
        
        model_list = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                model_list.append(m.name)
        
        if model_list:
            print("--- ¡Éxito! Modelos Encontrados ---")
            for model_name in model_list:
                print(f"- {model_name}")
            print("\nInstrucciones: Copia uno de estos nombres y pégalo en tus archivos `app_logic.py` y `config.py`.")
        else:
            print("No se encontraron modelos compatibles con 'generateContent' para tu API Key.")

    except Exception as e:
        print(f"\nOcurrió un error al contactar la API de Gemini: {e}")
        print("Verifica que tu API Key sea correcta y que tengas conexión a internet.")