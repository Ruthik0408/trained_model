# Project Flow Report

This document explains the complete end-to-end flow of the project:

- how requests enter the system
- what runs in PostgreSQL
- what runs in backend RAM
- what is stored permanently
- how ML features are created, filtered, standardized, and scored
- how anomaly rows are saved
- how review and report screens fetch data

---

## 1. System Overview

The project runs in 3 main places:

1. Browser / user device
2. FastAPI backend server
3. PostgreSQL

High-level flow:

```text
User Browser
    ->
FastAPI Backend
    ->
PostgreSQL
```

### What each place does

- Browser:
  Sends requests, receives JSON, renders screens.
- Backend:
  Builds SQL, loads data into pandas, engineers features, runs Isolation Forest, saves results, prepares review/report responses.
- PostgreSQL:
  Stores source data, stores final anomaly rows, stores metadata, executes joins and summary queries.

---

## 2. Databases Used

There are effectively 2 database roles in the code:

### Source PostgreSQL database

This contains the actual business/source tables being analyzed, for example:

- `bill`
- `vendor`
- `cheque_slip`
- `ecs`
- `schedule3`

It also contains the saved anomaly result table used by the project:

- `public.ML_Features`

### Application PostgreSQL database

This stores app metadata in:

- `anomaly_workbench_runs`

Model file:

- [backend/app/core/models.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/core/models.py)

This table stores:

- run name
- selected tables
- feature rules
- outlier rules
- anomaly counts
- metrics JSON
- selected and dropped feature columns

---

## 3. App Startup Flow

Main file:

- [backend/app/main.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/main.py)

When the backend starts:

1. logging is configured
2. rate limiting middleware is enabled
3. request ID middleware is enabled
4. app DB connection is checked
5. `anomaly_workbench_runs` is created if missing

### Where this runs

- Backend CPU/RAM
- App DB for metadata table creation

### What is stored

- No dataset is loaded yet
- No data is stored on user devices except the webpage itself

---

## 4. Main API Endpoints

Router file:

- [backend/app/api/routes_workbench.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/api/routes_workbench.py)

Important endpoints:

- `GET /api/workbench/tables`
- `GET /api/workbench/connection`
- `POST /api/workbench/default-feature-rules`
- `POST /api/workbench/preview`
- `POST /api/workbench/run`
- `GET /api/workbench/datasets`
- `GET /api/workbench/review-table`
- `GET /api/workbench/review-rows`
- `POST /api/workbench/feedback`
- `POST /api/workbench/isolation-reason`
- `GET /api/workbench/report`

---

## 5. Source Table Discovery

File:

- [backend/app/services/workbench/source_db.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/workbench/source_db.py)

Function:

- `list_source_tables()`

### What happens

1. backend queries `information_schema.columns`
2. columns are grouped by table
3. result is cached in backend memory

### Where this runs

- Query executes in PostgreSQL
- Returned metadata is stored temporarily in backend RAM cache

### Example result

```json
[
  {
    "table_name": "bill",
    "columns": [
      {"column_name": "bill_no", "data_type": "text"},
      {"column_name": "amount", "data_type": "numeric"},
      {"column_name": "bill_date", "data_type": "date"}
    ]
  }
]
```

---

## 6. Built-in Feature Rule Suggestion

Files:

- [backend/app/services/workbench/default_rules.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/workbench/default_rules.py)
- [backend/app/services/workbench/sql_runtime.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/workbench/sql_runtime.py)

### What happens

The backend looks at selected tables and columns and creates likely useful feature rules such as:

- `daysbetween`
- `isweekend`
- `isbusinesshour`

### Example

If a table has:

- `bill_date`
- `approval_date`

The backend may propose:

```json
{
  "name": "bill_date_to_approval_date",
  "feature_type": "daysbetween",
  "first_column": "approval_date",
  "second_column": "bill_date"
}
```

### Where this runs

- PostgreSQL for presence-ratio queries
- Backend RAM for rule assembly

---

## 7. Preview Flow

File:

- [backend/app/services/workbench/orchestrator.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/workbench/orchestrator.py)

Function:

- `preview_workbench()`

### What happens

1. SQL join is built
2. SQL is executed
3. result is loaded into pandas
4. first 50 rows are returned to UI

### Where this runs

- SQL execution: PostgreSQL
- pandas DataFrame: backend RAM
- response display: browser

### Important note

Preview still loads the returned join result into backend memory.

---

## 8. SQL Join Construction

File:

- [backend/app/services/workbench/sql_runtime.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/workbench/sql_runtime.py)

Main function:

- `_build_join_sql(...)`

### What it does

1. validates selected tables
2. validates joins
3. quotes identifiers safely
4. builds `SELECT ... FROM ... JOIN ...`
5. optionally adds date filters
6. optionally adds built-in SQL anomaly rules

### Example input

- selected tables: `bill`, `vendor`
- join: `bill.vendor_id = vendor.vendor_id`
- date filter: `2025-01-01` to `2025-12-31`

### Example SQL shape

```sql
SELECT
    "bill"."bill_no" AS "bill.bill_no",
    "bill"."amount" AS "bill.amount",
    "bill"."bill_date" AS "bill.bill_date",
    "vendor"."vendor_name" AS "vendor.vendor_name"
FROM public."bill" AS "bill"
LEFT JOIN public."vendor" AS "vendor"
    ON "bill"."vendor_id" = "vendor"."vendor_id"
WHERE ("bill"."bill_date" >= DATE '2025-01-01')
  AND ("bill"."bill_date" < (DATE '2025-12-31' + INTERVAL '1 day'))
```

### Where this runs

- SQL string is built in backend RAM
- SQL executes in PostgreSQL

---

## 9. Run Flow Overview

Main function:

- `run_workbench()`

File:

- [backend/app/services/workbench/orchestrator.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/workbench/orchestrator.py)

### High-level run order

1. build joined SQL
2. execute SQL and create temp table in PostgreSQL
3. read scoring columns into pandas
4. generate ML features
5. remove weak feature columns
6. impute and standardize
7. run Isolation Forest
8. determine final anomaly rows
9. fetch anomaly payload rows
10. generate explanations
11. save anomalies to `ML_Features`
12. save metadata to `anomaly_workbench_runs`

---

## 10. Temporary PostgreSQL Table

Function:

- `_materialize_workbench_temp_table(...)`

File:

- [backend/app/services/workbench/sql_runtime.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/workbench/sql_runtime.py)

### What happens

The full joined result is first written into a PostgreSQL temporary table:

```sql
CREATE TEMP TABLE "tmp_ml_join_xxx"
ON COMMIT DROP
AS
SELECT row_number() OVER () AS "__ml_row_number", src.*
FROM (...joined_sql...) src
```

### Why this exists

- creates stable row IDs
- allows later re-reading only anomaly rows
- avoids rerunning the full join multiple times

### Where it is stored

- PostgreSQL temp storage

### When it is deleted

- automatically by PostgreSQL because of `ON COMMIT DROP`

It is not permanent.

---

## 11. Scoring Frame Read Into RAM

Function:

- `_read_temp_scoring_frame(...)`

### What is loaded

Only scoring-related columns are loaded first:

- `__ml_row_number`
- SQL anomaly flags
- SQL anomaly reasons
- feature-rule columns

### Not loaded yet

- full payload columns are not all loaded for scoring

### Where this is stored

- pandas DataFrame in backend RAM

---

## 12. Feature Engineering

File:

- [backend/app/services/workbench/orchestrator.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/workbench/orchestrator.py)

Function:

- `_build_feature_frame(...)`

### Step 12.1: Start from SQL feature-rule columns

The code looks for explicit feature-rule aliases and converts them to numeric:

```python
extra = joined[sql_feature_cols].apply(pd.to_numeric, errors="coerce")
```

This means non-numeric values become `NaN`.

### Step 12.2: Add statistical IQR flags

File:

- [backend/app/services/workbench/ml_pipeline.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/workbench/ml_pipeline.py)

Function:

- `_add_statistical_outlier_signals(...)`

For amount-like numeric columns:

1. compute Q1
2. compute Q3
3. compute IQR = `Q3 - Q1`
4. compute lower and upper fences
5. add a new binary feature:

`iqr_flag::<column>`

### Example

Original values:

| row | amount |
|---|---:|
| 1 | 100 |
| 2 | 110 |
| 3 | 120 |
| 4 | 9000 |

Added feature:

| row | iqr_flag::amount |
|---|---:|
| 1 | 0 |
| 2 | 0 |
| 3 | 0 |
| 4 | 1 |

### Where this happens

- backend RAM using pandas

---

## 13. Feature Cleanup and Selection

File:

- [backend/app/services/workbench/ml_pipeline.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/workbench/ml_pipeline.py)

Function:

- `_prepare_isolation_forest_feature_frame(...)`

### What happens now

The code no longer keeps every feature blindly.

It does:

1. replace `inf` and `-inf` with `NaN`
2. drop all-missing columns
3. drop zero-variance columns
4. score remaining columns
5. keep only strongest columns

### Scoring inputs

Each column is scored using:

- non-missing coverage
- variance rank
- uniqueness ratio

### Example

Input feature columns:

| column | values |
|---|---|
| `feature_a` | `NaN, NaN, NaN` |
| `feature_b` | `5, 5, 5` |
| `feature_c` | `0, 10, 20` |
| `feature_d` | `0, 0, 1` |

Result:

- `feature_a` dropped as all-missing
- `feature_b` dropped as constant
- `feature_c` likely kept
- `feature_d` may be kept or dropped depending on score/rank

### What is recorded in metrics

The run metadata stores:

- selected feature columns
- dropped all-missing columns
- dropped constant columns
- dropped low-score columns
- feature scores

### Where this happens

- backend RAM

---

## 14. Feature Frame Validation

Function:

- `_validate_isolation_forest_feature_frame(...)`

### Checks

The pipeline stops if:

- there are no feature columns
- all remaining columns are missing
- every row has no usable engineered values

### Why

This avoids training on useless data.

---

## 15. Imputation

File:

- [backend/app/services/workbench/orchestrator.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/workbench/orchestrator.py)

Pipeline step:

- `SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)`

### What this means

- missing numeric values are replaced with the median of that feature
- missing-indicator columns may be created

### Example

Before:

| row | amount_gap |
|---|---:|
| 1 | 10 |
| 2 | NaN |
| 3 | 30 |

Median is `20`, so after imputation:

| row | amount_gap |
|---|---:|
| 1 | 10 |
| 2 | 20 |
| 3 | 30 |

And an extra indicator may be created:

| row | amount_gap__missing |
|---|---:|
| 1 | 0 |
| 2 | 1 |
| 3 | 0 |

### Where this happens

- backend RAM with scikit-learn

---

## 16. Standardization

Pipeline step:

- `StandardScaler()`

### What happens

Each feature is standardized to roughly:

- mean = 0
- standard deviation = 1

### Example

Original:

| row | amount_gap |
|---|---:|
| 1 | 10 |
| 2 | 20 |
| 3 | 30 |

Standardized approximate values:

| row | amount_gap_scaled |
|---|---:|
| 1 | -1.22 |
| 2 | 0.00 |
| 3 | 1.22 |

### Why

This prevents large-scale numeric columns from dominating smaller ones.

### Where this happens

- backend RAM with NumPy/scikit-learn

---

## 17. Isolation Forest

Pipeline step:

- `IsolationForest(...)`

### What happens

The model learns patterns of normal rows and gives an anomaly score to each row.

Generated outputs:

- `isolation_scores`
- `ml_flag`
- `ml_threshold`

### Final anomaly decision

The code combines:

- human SQL outlier flag
- built-in SQL anomaly flag
- Isolation Forest flag

Final logic:

```text
final_flag = human_outlier_flag OR ml_flag
```

### Example

| row | SQL rule | ML flag | final |
|---|---|---|---|
| 1 | false | false | false |
| 2 | true | false | true |
| 3 | false | true | true |

### Where this happens

- backend RAM and CPU

No model training happens in PostgreSQL.

---

## 18. Explanation Signals

File:

- [backend/app/services/llm_reason_service.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/llm_reason_service.py)

Function:

- `build_feature_explanation_signals(...)`

### What happens

For each anomaly row:

1. transformed feature values are examined
2. strongest features by absolute magnitude are selected
3. a small explanation signal payload is built

### Example signal

```json
[
  {"feature": "amount_difference", "value": 89000, "strength": 2.7, "direction": "high"},
  {"feature": "approval_lag_days", "value": 28, "strength": 1.9, "direction": "high"}
]
```

### Where this happens

- backend RAM

---

## 19. LLM Reason Generation

File:

- [backend/app/services/llm_reason_service.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/llm_reason_service.py)

Function:

- `explain_isolation_anomaly(...)`

### What happens

1. prompt is built
2. cache is checked
3. circuit breaker is checked
4. local Ollama is called with retry/backoff
5. if it fails, fallback template is used

### Where this happens

- prompt/result in backend RAM
- optional model call to Ollama service
- cache stored temporarily in backend memory

### Important

This is not stored permanently unless embedded later into saved JSON fields.

---

## 20. Fetching Only Final Anomaly Rows

Function:

- `_read_temp_anomaly_payload_frame(...)`

### What happens

After ML scoring, only rows with `final_flag = true` are fetched from the temp table.

This means:

- the full joined dataset is used for scoring
- only anomaly rows are used for final review storage

### Where this happens

- anomaly row IDs decided in backend RAM
- anomaly rows fetched from PostgreSQL temp table
- fetched anomaly rows loaded into backend RAM

---

## 21. Final Dataset Construction

File:

- [backend/app/services/workbench/result_store.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/workbench/result_store.py)

Function:

- `_build_dataset_frame(...)`

### What columns are saved

The saved anomaly dataset contains columns like:

- `feature_name`
- `feature_values_json`
- `human_rule_name`
- `human_rule`
- `isolation_rule`
- `ml_if_score`
- `ml_threshold`
- `ml_run_id`
- `review_payload_json`

### What each means

- `feature_name`:
  identifies the run/table group
- `feature_values_json`:
  selected engineered features, explanation signals, and optional LLM reason
- `human_rule_name`:
  readable rule/anomaly reason
- `human_rule`:
  whether rule-based anomaly fired
- `isolation_rule`:
  whether Isolation Forest fired
- `review_payload_json`:
  original business fields shown on the review screen

### Example saved row

```json
{
  "feature_name": "bill.vendor",
  "feature_values_json": {
    "amount_difference": 89000,
    "iqr_flag::bill.amount": 1,
    "__ml_explanation_signals": [
      {"feature": "amount_difference", "value": 89000, "strength": 2.7, "direction": "high"}
    ],
    "__ml_llm_if_reason": "The amount is unusually high compared with similar records."
  },
  "human_rule_name": "Large amount threshold",
  "human_rule": true,
  "isolation_rule": true,
  "ml_if_score": 0.93,
  "ml_threshold": 0.61,
  "ml_run_id": 7,
  "review_payload_json": {
    "bill_no": "B003",
    "amount": 90000,
    "vendor_name": "Vendor B",
    "bill_date": "2025-01-04"
  }
}
```

### Where this happens

- pandas DataFrame in backend RAM

---

## 22. Writing Anomaly Rows to PostgreSQL

Function:

- `_write_dataset_to_result(...)`

### What happens

1. backend checks target result table exists
2. validates required columns
3. converts DataFrame rows to dictionaries
4. inserts them into `public.ML_Features`
5. gets inserted IDs back

### Where this is stored

- permanently in PostgreSQL

### What is temporary

- the pandas dataset DataFrame used before insert

---

## 23. Writing Run Metadata

Function:

- `_create_run_record(...)`

Table:

- `anomaly_workbench_runs`

### What is saved

- run ID
- run name
- selected tables
- total source rows
- anomaly counts
- selected model
- metrics JSON

### Metrics JSON includes

- `feature_count`
- selected feature columns
- dropped feature columns
- feature scores
- warnings
- source row counts
- SQL execution details

### Where this is stored

- permanently in app DB

---

## 24. What Gets Deleted After the Run

### Deleted automatically

- PostgreSQL temp join table

### Not stored permanently in RAM

- joined DataFrame
- engineered features DataFrame
- selected feature frame
- transformed NumPy arrays
- anomaly payload DataFrame

### Important note about RAM

The logical data is temporary, but Python may not immediately return all freed memory to the operating system. It can keep memory available for reuse by the same backend process.

### Permanent storage after run

- anomaly rows in `ML_Features`
- metadata row in `anomaly_workbench_runs`
- feedback updates later made by reviewers

---

## 25. Review Screen Flow

File:

- [backend/app/services/dashboard_service.py](/home/ruthikreddy/Desktop/new_2_qwen_shap/backend/app/services/dashboard_service.py)

Main functions:

- `review_table_data(...)`
- `review_rows_data(...)`

### What happens

1. backend finds latest dataset table/run
2. queries saved anomaly rows from `ML_Features`
3. applies filter:
   - `rule`
   - `ml`
   - `reviewed`
   - `not_reviewed`
   - `all`
4. converts saved JSON payload into UI-friendly fields

### Fields shown on review screen are derived from

- `review_payload_json`
- `feature_values_json`
- saved rule fields
- saved score fields

### Example review row returned to browser

```json
{
  "serial_no": 1,
  "prediction_id": 101,
  "anomaly": "Yes",
  "reason": "Large amount threshold, unusual amount pattern",
  "amount": 90000,
  "bill_no": "B003",
  "vendor_name": "Vendor B",
  "office_name": "Office X",
  "detected_on": "2025-01-04",
  "risk_score": 93.0
}
```

### Where this runs

- SQL summary/data fetch: PostgreSQL
- response assembly: backend RAM
- display: browser

---

## 26. Report Screen Flow

Function:

- `report_data(...)`

### What happens

1. backend selects latest or requested run
2. reads run metadata from `anomaly_workbench_runs`
3. reads anomaly totals from `ML_Features`
4. reads feedback status counts
5. returns report summary

### Example report response

```json
{
  "run_id": 42,
  "dataset_table": "ML_Features",
  "run_name": "Vendor amount anomaly scan",
  "selected_tables": ["bill", "vendor"],
  "total_rows": 150000,
  "anomaly_count": 420,
  "reviewed_count": 100,
  "pending_count": 320,
  "accepted_count": 55,
  "amount": 1240000.0,
  "from_date": "2025-01-01",
  "to_date": "2025-12-31",
  "selected_model": "IsolationForest"
}
```

### Where this runs

- PostgreSQL for querying
- backend RAM for response formatting

---

## 27. Feedback Update Flow

Function:

- `update_dataset_feedback(...)`

### What happens

When a reviewer clicks:

- accept
- reject
- maybe

the backend updates `feedback_score` in `ML_Features`.

### Mapping

- `accept -> 1.0`
- `maybe -> 0.5`
- `reject -> 0.0`

### Where this is stored

- permanently in PostgreSQL

---

## 28. Full Example End-to-End

Suppose the source joined result looks like this:

| row_id | bill_no | amount | approved_amount | vendor_name | bill_date |
|---|---|---:|---:|---|---|
| 1 | B001 | 1000 | 1000 | Vendor A | 2025-01-01 |
| 2 | B002 | 1200 | 1190 | Vendor A | 2025-01-03 |
| 3 | B003 | 90000 | 1000 | Vendor B | 2025-01-04 |

### Feature rules produce

| row_id | amount_difference | isweekend |
|---|---:|---:|
| 1 | 0 | 0 |
| 2 | 10 | 0 |
| 3 | 89000 | 1 |

### IQR feature adds

| row_id | iqr_flag::amount |
|---|---:|
| 1 | 0 |
| 2 | 0 |
| 3 | 1 |

### Final feature frame might be

| row_id | amount_difference | isweekend | iqr_flag::amount |
|---|---:|---:|---:|
| 1 | 0 | 0 | 0 |
| 2 | 10 | 0 | 0 |
| 3 | 89000 | 1 | 1 |

Then:

1. missing values are imputed if needed
2. values are standardized
3. Isolation Forest runs
4. row 3 gets high anomaly score
5. row 3 is fetched again from temp table as full payload
6. row 3 is saved into `ML_Features`
7. review screen later reads it back

---

## 29. Where Everything Lives

### Browser

- rendered UI
- paginated API responses
- not the full workbench dataset

### Backend RAM

- SQL strings
- pandas DataFrames
- NumPy arrays
- scikit-learn pipeline state during request
- caches
- explanation payloads

### PostgreSQL

- source business tables
- temporary join table during run
- permanent anomaly result rows in `ML_Features`
- permanent run metadata in `anomaly_workbench_runs`

---

## 30. Final Summary

### PostgreSQL does

- source data storage
- join execution
- temp table creation
- final anomaly row storage
- metadata storage
- review/report querying

### Backend RAM does

- feature engineering
- column scoring and selection
- imputation
- standardization
- Isolation Forest training/scoring
- explanation generation
- final response assembly

### Browser does

- sends API requests
- receives paginated JSON
- renders review/report screens

The full ML pipeline is server-side. Users only need the URL. They do not need the code or a local copy of the database.
