# run.py
import os
from mi_aplicacion import create_app # Importa la factory de la carpeta de tu aplicación

# Determinar la configuración a usar (ej. 'dev' o 'prod')
# Podrías usar una variable de entorno FLASK_ENV o FLASK_CONFIG
config_name = os.getenv('FLASK_CONFIG', 'dev')

app = create_app(config_name)

if __name__ == '__main__':
    # El host '0.0.0.0' hace que sea accesible desde otras máquinas en la red local.
    # Para desarrollo local solamente, '127.0.0.1' es suficiente.
    # El puerto 5000 es el por defecto de Flask.
    app.run(host='127.0.0.1', port=5000, debug=True) # debug=True se maneja por la configuración
