# Tulip 2.0 Anomaly Workbench

This project is a PostgreSQL-backed anomaly review system with a FastAPI backend and a Vite/React frontend.

The backend does four things:

1. Reads source data from the source PostgreSQL schema.
2. Builds SQL joins and rule-based features for a selected dataset.
3. Scores rows with the saved anomaly model and persists anomaly rows into the result table.
4. Serves review, anomaly-list, and reporting APIs to the frontend.

The storage model is intentionally split:

- `PostgreSQL` is the source of truth for app metadata and persisted anomaly rows.
- `Valkey` is the shared cache and temporary artifact store when available.
- `SQLAlchemy` is the Python database layer, not a database engine.
- Local process memory is used only for short-lived fallback caches and in-flight DataFrames.

Start with [ARCHITECTURE.md](/home/ruthikreddy/Desktop/may_29/ARCHITECTURE.md) for the system flow, module map, and request lifecycle.

## Repo Layout

- `backend/app/main.py`: FastAPI app setup, middleware, health check, router registration.
- `backend/app/api/routes_workbench.py`: HTTP entrypoints for tables, preview, run, review, anomalies, feedback, and reports.
- `backend/app/services/workbench/orchestrator.py`: main workbench execution flow.
- `backend/app/services/workbench/sql_runtime.py`: SQL join builder, rule pushdown, temp-table handling.
- `backend/app/services/workbench/result_store.py`: result-table writes and feedback updates.
- `backend/app/services/workbench/source_db.py`: source/result PostgreSQL access helpers.
- `backend/app/services/dashboard_service.py`: dashboard and review data assembly.
- `frontend/src/pages/WorkbenchPage.jsx`: UI for configuring and running the workbench.
- `frontend/src/api/anomalyApi.js`: frontend API client with a small in-memory cache.

## Core Workflow

1. Frontend posts a `WorkbenchRunRequest` to `/api/workbench/preview` or `/api/workbench/run`.
2. Routes apply trained-dataset defaults and call the orchestration layer.
3. The orchestration layer asks `sql_runtime.py` to build and execute the source SQL join.
4. Preview returns rows directly. Run continues into feature scoring and anomaly selection.
5. Anomaly rows are written into the result table in PostgreSQL.
6. Run metadata is written into the app database table `anomaly_workbench_runs`.
7. Review payload artifacts and shared query caches are stored in Valkey when available.
8. Dashboard and review endpoints read the persisted rows plus cached artifacts to serve the UI.

## Improvement Suggestions

- Add a generated API reference from the Pydantic request/response models so route contracts stay visible.
- Split `sql_runtime.py` into smaller modules by concern: join planning, rule SQL generation, temp-table reads.
- Add a short ADR for the dual-database design so future contributors understand why app metadata and anomaly rows live in different places.
