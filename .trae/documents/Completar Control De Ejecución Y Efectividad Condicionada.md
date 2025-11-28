## Objetivo
Implementar el control de ejecución por acción de cada Plan de Intervención y conectar esos datos con una vista y API de “Efectividad de Intervenciones” condicionada al conjunto de acciones efectivamente aplicadas. Mantener el aislamiento por usuario (demo) en biblioteca, contexto y métricas.

## Alcance
- Registro y gestión del estado de cada acción del plan (aplicada, no aplicada, en progreso, no aplicable). 
- Persistencia por plan y usuario; edición en la UI del plan.
- Cálculo de efectividad (Δ promedio, Δ asistencia) condicionada al subconjunto de acciones aplicadas dentro de una ventana temporal configurable.
- Correcciones de Biblioteca para listar Reportes 360, Planes y Observaciones por entidad y usuario.

## Cambios Técnicos
### Datos
- Tabla `intervention_actions` (plan_id, owner_username, action_title, status, applied_date, responsable, compliance_pct, notes; UNIQUE(plan_id, action_title, owner_username)).
- Campo `owner_username` ya presente en `follow_ups` para separar por usuario; se usará en consultas y FAISS.

### Backend
- Extracción de acciones desde el HTML del plan (strong/li) y inicialización automática al visualizar el plan.
- Endpoints:
  - `POST /api/intervention_action_status` para guardar/actualizar estado y metadatos de una acción.
  - Extender `GET /api/intervention_effectiveness` con parámetros: `days`, `entity_type`, `entity_name`, `only_applied=true` para calcular deltas considerando solo planes con acciones “applied”.
- Biblioteca:
  - Consulta por `related_filename` y `(owner_username = ? OR owner_username IS NULL)` para incluir históricos previos; filtros por `tipo_entidad` y `nombre_entidad`.

### Frontend
- `visualizar_plan_intervencion.html`:
  - Tabla editable de acciones del plan con estado, fecha, responsable, cumplimiento y notas; botón “Guardar” por fila.
  - Mensajes de confirmación al guardar.
- `efectividad_intervenciones.html`:
  - Añadir filtros por entidad y días; mostrar cumplimiento global del plan (X/Y acciones aplicadas, % promedio de cumplimiento) y deltas condicionados.
- Biblioteca:
  - Asegurar listado de Planes de Intervención y Observaciones junto con Reportes; mantener orden por timestamp.

## Seguridad y Aislamiento
- Aislar por usuario demo: lectura/escritura de `intervention_actions` y `follow_ups` con `owner_username`.
- Contexto y FAISS: rutas por usuario (`instance/users/<usuario>/faiss_index_*`); recarga tras guardar plan/observación.

## Verificación
- Casos:
  - Generar Reporte 360 → Generar Plan (curso y alumno) → Ver plan y extraer acciones → Marcar 2 acciones “applied” y 1 “pending” → Guardar.
  - Biblioteca: ver que el Plan aparece bajo la entidad y archivo del usuario.
  - Efectividad: ver deltas y resumen de cumplimiento cuando `only_applied=true` y comparar con la vista no condicionada.
- Pruebas de aislamiento: usuario `demo` vs `demo1` con archivos distintos; ver bibliotecas y contexto independientes.

## Compatibilidad
- Migración ligera añade `owner_username` y crea `intervention_actions` sin alterar registros previos; Biblioteca incluye históricos sin `owner_username`.

## Confirmación
Si apruebas, procedo a:
1) Implementar endpoint y cálculo de efectividad condicionada.
2) Completar la UI de control en la vista del plan (guardar estados) y filtros en efectividad.
3) Corregir Biblioteca para listar planes y observaciones consistentemente.
4) Validar con datos demo aislados por usuario y documentar uso rápido para docentes/directivos.