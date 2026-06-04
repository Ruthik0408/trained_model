# Architecture

## Purpose

Tulip 2.0 is an anomaly workbench. Users choose trained source tables, preview the joined dataset, run anomaly scoring, persist anomaly rows, review them, and inspect aggregate reports.

The codebase has three main execution surfaces:

- FastAPI backend for orchestration and APIs.
- PostgreSQL for source data, result rows, and app metadata.
- React frontend for workbench configuration and review flows.

## High-Level Components

### Frontend

- `frontend/src/pages/WorkbenchPage.jsx`
  Collects selected tables, join rules, feature rules, and run inputs.
- `frontend/src/api/anomalyApi.js`
  Calls backend APIs and caches GET responses in browser memory.
- `frontend/src/pages/ReviewPage.jsx`, `ReportsPage.jsx`, `AnomaliesPage.jsx`
  Render persisted anomaly rows and dashboard summaries.

### Backend API Layer

- `backend/app/main.py`
  Creates the FastAPI app, configures middleware, initializes the app database schema, and exposes `/health`.
- `backend/app/api/routes_workbench.py`
  Defines the workbench-facing HTTP routes and maps them to service-layer functions.

### Backend Service Layer

- `backend/app/services/workbench/orchestrator.py`
  Owns the end-to-end preview and run flow.
- `backend/app/services/workbench/sql_runtime.py`
  Builds safe SQL joins, pushes rule logic into SQL, materializes temp tables, and reads scoring frames.
- `backend/app/services/workbench/saved_model_inference.py`
  Loads the trained artifact and computes anomaly scores.
- `backend/app/services/workbench/result_store.py`
  Converts anomaly outputs into the result-table shape and writes them to PostgreSQL.
- `backend/app/services/dashboard_service.py`
  Reads persisted datasets and cached review payloads to support dashboards and review pages.
- `backend/app/services/reason_service.py`
  Builds human-readable explanation text from model/rule signals.

### Persistence and Infrastructure

- `backend/app/core/database.py`
  SQLAlchemy engine and session for the application database.
- `backend/app/core/models.py`
  ORM models for persisted app metadata, currently `anomaly_workbench_runs`.
- `backend/app/services/workbench/source_db.py`
  SQLAlchemy engine helpers for the source/result PostgreSQL database.
- `backend/app/core/cache.py`
  Shared TTL cache abstraction.
- `backend/app/core/valkey.py`
  Valkey access layer and fallback behavior.
- `backend/app/core/config.py`
  Environment loading and runtime settings.

## Data Boundaries

### 1. Source PostgreSQL database

Used for:

- reading source operational tables
- materializing temp workbench SQL tables
- writing anomaly rows into the result table

Primary module:

- `backend/app/services/workbench/source_db.py`

### 2. Application PostgreSQL database

Used for:

- persisting run metadata in `anomaly_workbench_runs`

Primary modules:

- `backend/app/core/database.py`
- `backend/app/core/models.py`

### 3. Valkey

Used for:

- preview artifacts
- cached workbench execution artifacts
- cached Isolation Forest artifacts
- cached review payload rows
- shared TTL-backed query caches
- rate limiting when enabled

If Valkey is unavailable, the system falls back to local in-process memory for some cache concerns, but shared cache behavior is reduced.

## End-to-End Request Flow

### Preview flow

1. Frontend sends `POST /api/workbench/preview`.
2. `routes_workbench.py` validates the request and applies trained dataset defaults.
3. `orchestrator.preview_workbench()` checks Valkey for a cached preview artifact.
4. On a cache miss, `sql_runtime.py` builds and executes the join SQL against the source database.
5. The preview result is serialized into Valkey and returned to the frontend.

### Run flow

1. Frontend sends `POST /api/workbench/run`.
2. `routes_workbench.py` validates input and opens an app DB session.
3. `orchestrator.run_workbench()` checks Valkey for cached execution artifacts.
4. On a cache miss:
   - `sql_runtime.py` builds the joined workbench SQL.
   - the SQL is executed in the source database
   - a temp staging table is materialized for scoring
5. `saved_model_inference.py` loads the trained model and scores the candidate rows.
6. `result_store._build_dataset_frame()` produces the persisted anomaly-row DataFrame.
7. `result_store._write_dataset_to_result()` appends anomaly rows into the result table in PostgreSQL.
8. `orchestrator._create_run_record()` persists run metadata into `anomaly_workbench_runs`.
9. Review payload rows and execution artifacts are cached in Valkey.
10. The API returns the run summary to the frontend.

### Review/report flow

1. Frontend calls review/anomaly/report endpoints.
2. `dashboard_service.py` reads persisted result rows from PostgreSQL.
3. It enriches rows with cached review payload artifacts from Valkey when present.
4. Response data is shaped for the UI and cached in short-lived TTL caches.

## Module Dependency Map

The main backend dependency chain is:

`main.py`
-> `routes_workbench.py`
-> `orchestrator.py`
-> `sql_runtime.py`
-> `saved_model_inference.py`
-> `result_store.py`
-> `source_db.py` / `database.py`

Supporting services:

- `dashboard_service.py` depends on persisted result rows plus cached review payloads.
- `reason_service.py` is used during anomaly explanation generation.
- `cache.py` and `valkey.py` are cross-cutting infrastructure.

## Operational Notes

- `TULIP_APP_DB_URL` points to the application PostgreSQL database.
- `TULIP_SOURCE_DB_*` variables point to the source/result PostgreSQL database.
- `TULIP_VALKEY_ENABLED` controls whether shared cache/rate-limit behavior uses Valkey.
- The app logs whether Valkey is connected or running in local fallback mode at startup.

## Known Design Pressure Points

- `sql_runtime.py` is carrying too many responsibilities and should eventually be split.
- The result table is assumed to already exist with the required columns.
- Dual-database behavior is correct here, but it needs explicit documentation because it is easy to confuse app DB, source DB, and cache responsibilities.

## Recommended Next Documentation Steps

- Add sequence diagrams for preview and run flows.
- Add a schema note for `anomaly_workbench_runs` and the result table.
- Add environment setup docs with a sample `.env` and an explanation of each DB variable.
