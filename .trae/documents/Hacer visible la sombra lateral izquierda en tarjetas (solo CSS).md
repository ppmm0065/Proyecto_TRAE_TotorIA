## Problema
La sombra lateral izquierda no se aprecia: el `box-shadow` negativo se mezcla con el fondo blanco y puede quedar visualmente imperceptible.

## Solución (solo presentación)
1) Implementar un pseudo‑elemento interno para la tarjeta:
   - `.card-left-shadow::before { content:''; position:absolute; left:0; top:8px; bottom:8px; width:12px; border-radius:12px; background: linear-gradient(to right, rgba(226,232,240,0.8), rgba(226,232,240,0)); pointer-events:none; }`
   - Mantener `position:relative` en `.card-left-shadow`.
   - Esto crea una franja sutil dentro del borde izquierdo, visible sobre fondo blanco.

2) Conservar el estilo base de tarjeta:
   - `bg-white p-4 rounded-xl shadow-sm border border-slate-200 card-left-shadow` (sin tocar lógica ni datos).

3) Aplicación:
   - Solo en “Análisis Detallado → Resumen de Métricas” (ya referencian `card-left-shadow`).
   - No cambiar otros componentes ni scripts.

4) Ajuste fino (si fuera necesario tras ver el resultado):
   - Opacidad y ancho de la franja (`0.6–0.8` y `10–14px`) para lograr el nivel de sutileza deseado.

5) Verificación
   - Hard refresh (Ctrl+F5). No requiere reinicio del servidor.

¿Confirmas que aplique esta mejora CSS para que la sombra lateral izquierda sea visible y sutil?