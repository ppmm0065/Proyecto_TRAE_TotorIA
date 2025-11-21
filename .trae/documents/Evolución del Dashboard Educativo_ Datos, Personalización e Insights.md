## Objetivo
Replicar el aspecto de las tarjetas del dashboard principal en “Análisis Detallado → Resumen de Métricas”, añadiendo una sombra lateral izquierda. Sin tocar lógica ni datos.

## Cambios (solo CSS/plantillas)
1) Añadir una utilidad CSS en `base.html` dentro del bloque `<style>`:
   - `.card-left-shadow { position: relative; box-shadow: -8px 0 16px rgba(226,232,240,0.7), 0 1px 2px rgba(0,0,0,0.04); }`
   - Permite una sombra suave hacia el borde izquierdo; es reversible y no depende de pseudo-elementos.

2) Aplicar la clase en `detalle_dashboard.html`:
   - En las 4 tarjetas del bloque “alumno” (Promedio, Curso, Edad, Peor Asignatura).
   - En las tarjetas del bloque “curso” (Nº Alumnos, Prom. Curso, Peor Asignatura) y en “Alumno Menor Prom.” (link y fallback).
   - Mantener el estilo base de tarjeta: `bg-white p-4 rounded-xl shadow-sm border border-slate-200`.

3) Verificación visual
   - Hard refresh (Ctrl+F5) en la página de detalle para ver la sombra.
   - Confirmar que ningún contenedor padre recorta la sombra.

4) Rollback inmediato
   - Quitar la clase `card-left-shadow` de las tarjetas o comentar la regla CSS si no convence.

## Alcance y seguridad
- No se modifica `app_logic.py`, ni rutas, ni cálculos; solo presentación.
- Cambios acotados y fácilmente reversibles.

¿Confirmas que aplique estos cambios de presentación ahora?