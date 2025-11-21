## Objetivo
Convertir la herramienta en un sistema de apoyo a la gestión pedagógica que aprende del historial del colegio, detecta riesgos tempranos, sugiere intervenciones efectivas y ofrece visualizaciones útiles para la toma de decisiones.

## Etapa 0: Base de Datos y Calidad de Datos
- Unificar el esquema de datos (estudiantes, cursos, evaluaciones, asistencia, intervenciones, resultados) en SQLite con ORM.
- ETL de CSV actuales hacia el esquema normalizado; controles de calidad (duplicados, codificación, tipos, outliers).
- Versionado de datasets y trazabilidad por fecha/cohorte.

Criterios de aceptación
- Tablas creadas y pobladas con controles de calidad; consultas consistentes para cohortes.

## Etapa 1: Detección Temprana (Reglas y Señales)
- Señales tempranas basadas en reglas: caída de promedio, inasistencia, rezago vs curso, materias débiles.
- Score de riesgo inicial (0–100) combinando señales ponderadas.
- Alertas y umbrales configurables por nivel/curso.

Criterios de aceptación
- Listados de estudiantes con nivel de riesgo y alertas por curso; falsos positivos controlados.

## Etapa 2: Modelo Predictivo de Riesgo
- Entrenamiento supervisado (p. ej., `LogisticRegression`/`XGBoost`) con features de historial (tendencias, varianza, asistencia, correlaciones por asignatura).
- Validación temporal (train/test por años/semestres); métricas (AUC, F1, precisión en top-k).
- Persistencia de artefactos del modelo y pipeline de features; monitor de drift.

Criterios de aceptación
- Modelo con métricas aceptables y scoring reproducible por cohorte; registro de versiones.

## Etapa 3: Recomendador de Intervenciones
- Motor de reglas + evidencia: cruza señales/modelo con catálogo de intervenciones (intensidad, costo, efectividad histórica).
- Integración con RAG (documentos institucionales) para generar recomendaciones contextualizadas y accionables.
- Captura de decisión y ejecución (quién, cuándo, qué) con objetivos y duración.

Criterios de aceptación
- Para cada estudiante en riesgo, sugerencias priorizadas con justificación y pasos concretos.

## Etapa 4: Visualizaciones para Decisión
- Dashboard por niveles: panorama de riesgo, cohortes, asignaturas críticas, tendencia.
- Vistas de detalle: estudiante, curso, asignatura, evolución temporal, impacto de intervenciones.
- “What-if” simple: simulación de mejoras en asistencia/notas y efecto en riesgo.

Criterios de aceptación
- Paneles navegables con filtros y tiempos de respuesta fluidos; exportes básicos.

## Etapa 5: Bucle de Aprendizaje del Historial
- Registro de resultados de intervenciones (cumplimiento, impacto en notas/asistencia, tiempo a mejora).
- Actualización periódica del modelo con nuevos datos; evaluación de uplift.
- Reporte de efectividad por tipo de intervención y contexto.

Criterios de aceptación
- El sistema aprende: mejoras medibles en precisión y recomendaciones con el tiempo.

## Etapa 6: Gobernanza, Privacidad y Seguridad
- Políticas de acceso y auditoría; anonimización para reportes agregados.
- Conformidad básica (consentimiento, minimización de datos, retención).
- Trazabilidad de cambios y responsables.

Criterios de aceptación
- Accesos controlados, auditorías registradas y datos sensibles protegidos.

## Etapa 7: Performance y Operación
- Optimización de consultas y cálculos (pandas vectorizado; índices en BD).
- Jobs programados para scoring y reentrenos; observabilidad.
- Tolerancia a fallos en ingesta y reindexación RAG.

Criterios de aceptación
- Procesos estables y medidos; SLIs básicos (latencia de scoring, tiempo de reentreno).

## Integración con el Código Actual
- `mi_aplicacion/app_logic.py` → dividir en servicios: `services/data.py`, `services/risk.py`, `services/interventions.py`, `services/visuals.py`, `services/rag.py`.
- `mi_aplicacion/__init__.py:create_app` → inyectar dependencias (modelos, vector stores) en `app.extensions`.
- `mi_aplicacion/routes.py` → nuevas rutas para dashboards y scoring; reutilizar prompts con RAG.
- `config.py` → parámetros de umbrales, ventanas temporales, versión de modelo.
- Referencia visible de validación de archivo actual: `mi_aplicacion/app_logic.py:42` (fortalecer controles y feedback).

## Entregables por Etapa
- Código y pruebas unitarias/funcionales.
- Métricas y reportes breves (no sensibles) para evaluar impacto.
- Guía de rollback y flags para activación gradual.

## Solicitud de Confirmación
Si apruebas este plan, comienzo por la Etapa 0 (BD y calidad de datos) y Etapa 1 (señales tempranas), con cambios mínimos, verificables y reversibles, integrando visualizaciones básicas para decisión rápida.