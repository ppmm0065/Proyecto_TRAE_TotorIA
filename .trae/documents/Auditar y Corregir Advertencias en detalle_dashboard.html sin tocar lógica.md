## Lo que haré (sin riesgo)
- Revisar el archivo `detalle_dashboard.html` en modo lectura para:
  1) Verificar cierre correcto de bloques Jinja (`{% ... %}`) y etiquetas HTML.
  2) Buscar reglas CSS o texto fuera del bloque `<style>` (evita que aparezcan como texto en la página).
  3) Comprobar que las clases utilitarias usadas existen (no introducir errores); mantener solo `bg-white`, `rounded-*`, `shadow-*`, `border`.
  4) Validar que no haya IDs duplicados o mal formados.

## Posibles causas de “188 problemas”
- Linter/validador leyendo Jinja como HTML y marcando sintaxis `{% %}`/`{{ }}` como errores.
- Reglas CSS o comentarios mal cerrados que el navegador muestra como texto.
- Etiquetas HTML no cerradas en bloques largos.
- Advertencias de Tailwind/HTML por clases desconocidas o atributos (no bloqueantes).

## Correcciones seguras (si se confirman)
- Mover cualquier regla CSS fuera de `<style>` adentro del bloque `<style>` y cerrar correctamente.
- Cerrar etiquetas y bloques Jinja donde falte.
- Sustituir clases no válidas por equivalentes estándar ya usados en el proyecto.
- Mantener exactamente el estilo de tarjetas acordado: `bg-white p-4 rounded-xl shadow-sm border border-slate-200 card-left-shadow` sin tocar cálculos ni rutas.

## Entregables
- Informe de hallazgos con línea exacta y recomendación de corrección.
- Aplicación de cambios presentacionales mínimos (si apruebas) y verificación con hard refresh.

¿Confirmas que avance con esta auditoría y, si encuentro causas claras del “188 problemas”, aplique correcciones de presentación (no lógicas) inmediatamente?