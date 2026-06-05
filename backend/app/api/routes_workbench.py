"""HTTP entrypoints for the workbench and review flows."""
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from app.services.workbench.business_rules import RULE_REGISTRY_INDEX
from app.core.database import check_app_db_connection, get_db
from app.core.errors import WorkbenchValidationError
from app.schemas.workbench_schema import (
    AnomalyListResponse,
    DatasetFeedbackRequest,
    ReportResponse,
    TableInfo,
    WorkbenchRunRequest,
    WorkbenchRunResponse,
)
from app.services.dashboard_service import anomaly_list_data, report_data, review_rows_data
from app.services.workbench.runner import run_workbench
from app.services.workbench.result_store import list_saved_datasets, update_dataset_feedback
from app.services.workbench.source_db import list_source_tables, source_connection_status
from app.services.workbench.trained_datasets import apply_trained_dataset_defaults, trained_selectable_tables

router = APIRouter(prefix="/api/workbench", tags=["workbench"])
logger = logging.getLogger(__name__)
SAFE_IDENTIFIER_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"


def _log_request(request: Request, message: str) -> None:
    """Write a request-scoped debug log line using the middleware request id."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.debug("[%s] %s", request_id, message)


@router.get(
    "/rules",
    summary="List deterministic business rules",
)
def get_rule_catalog(request: Request):
    """Return the shared deterministic rule catalog for debugging and review."""
    _log_request(request, f"Retrieved {len(RULE_REGISTRY_INDEX)} deterministic rules")
    return {
        "count": len(RULE_REGISTRY_INDEX),
        "rules": RULE_REGISTRY_INDEX,
    }


@router.get(
    "/tables",
    response_model=list[TableInfo],
    summary="List source tables",
    responses={
        503: {"description": "Source database is unavailable."},
        500: {"description": "Unexpected failure while inspecting source metadata."},
    },
)
def get_tables(request: Request):
    """Return trained source tables and columns for workbench configuration."""
    try:
        trained_table_set = set(trained_selectable_tables())
        tables = [
            table
            for table in list_source_tables()
            if table.get("table_name") in trained_table_set
        ]
        _log_request(request, f"Retrieved {len(tables)} tables")
        return tables
    except ConnectionError as exc:
        logger.warning("Connection error fetching tables: %s", exc)
        raise HTTPException(status_code=503, detail="Cannot connect to source database")
    except Exception as exc:
        status = source_connection_status()
        if not status.get("connected"):
            logger.warning("Source metadata unavailable while fetching tables: %s", status)
            raise HTTPException(status_code=503, detail=status)
        logger.exception("Error fetching tables: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve tables. Check the request ID in server logs.",
        )


@router.post(
    "/run",
    response_model=WorkbenchRunResponse,
    summary="Execute anomaly workbench",
    responses={
        400: {
            "description": (
                "Workbench validation failed, including edge cases such as empty engineered "
                "feature frames or feature rules that only produce missing values."
            )
        },
        503: {"description": "A required database connection is unavailable."},
    },
)
def run_workbench_route(
    payload: WorkbenchRunRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Execute the full workbench pipeline and persist anomaly results."""
    try:
        effective_payload = apply_trained_dataset_defaults(payload)
        result = run_workbench(db, effective_payload)
        _log_request(request, f"Workbench run completed — Run ID: {result.get('run_id')}")
        return result
    except ConnectionError as exc:
        logger.warning("Workbench source DB connection error: %s", exc)
        raise HTTPException(status_code=503, detail="Cannot connect to source database")
    except WorkbenchValidationError as exc:
        logger.warning("Workbench structured validation error: %s", exc.message)
        raise HTTPException(status_code=400, detail=exc.to_http_detail())
    except ValueError as exc:
        logger.warning("Workbench validation error: %s", exc)
        raise HTTPException(status_code=400, detail=f"Validation error: {str(exc)}")
    except HTTPException:
        raise
    except Exception:
        logger.exception("Workbench execution failed")
        raise HTTPException(
            status_code=500,
            detail="Workbench execution failed. Check the request ID in server logs.",
        )


@router.get(
    "/datasets",
    summary="List saved datasets",
    responses={503: {"description": "Application database is unavailable."}},
)
def dataset_list_route(request: Request, db: Session = Depends(get_db)):
    """Return datasets created by prior workbench runs."""
    try:
        datasets = list_saved_datasets(db)
        _log_request(request, f"Retrieved {len(datasets)} datasets")
        return datasets
    except OperationalError as exc:
        logger.warning("Application DB unavailable while fetching datasets: %s", exc)
        status = check_app_db_connection()
        raise HTTPException(
            status_code=503,
            detail=status
            if not status.get("connected")
            else {"connected": False, "error": "Application database is unavailable"},
        )
    except Exception:
        logger.exception("Error fetching datasets")
        raise HTTPException(status_code=500, detail="Failed to retrieve datasets")


@router.get("/review-rows", summary="Get review rows")
def review_rows_route(
    request: Request,
    dataset_table: Annotated[str | None, Query(pattern=SAFE_IDENTIFIER_PATTERN)] = None,
    anomaly_filter: str = "all",
    limit: int | None = None,
    offset: int = 0,
    run_id: int | None = None,
    db: Session = Depends(get_db),
):
    """Return review rows for a dataset or run, filtered by anomaly status."""
    try:
        result = review_rows_data(
            db,
            dataset_table=dataset_table,
            anomaly_filter=anomaly_filter,
            limit=limit,
            offset=offset,
            run_id=run_id,
        )
        _log_request(
            request,
            f"Retrieved {len(result) if isinstance(result, list) else 'rows'} from review",
        )
        return result
    except Exception:
        logger.exception("Error fetching review rows")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve review rows. Check the request ID in server logs.",
        )

@router.get(
    "/anomalies",
    response_model=AnomalyListResponse,
    summary="List anomalies from ML_Features",
)
def anomaly_list_route(
    request: Request,
    dataset_table: Annotated[str, Query(pattern=SAFE_IDENTIFIER_PATTERN)] = "ML_Features",
    table_filter: str | None = None,
    anomaly_type: str = "all",
    review_status: str = "all",
    limit: int | None = None,
    offset: int = 0,
):
    """Return a paginated anomaly list from ML_Features with table/type/review filters."""
    try:
        result = anomaly_list_data(
            dataset_table=dataset_table,
            table_filter=table_filter,
            anomaly_type=anomaly_type,
            review_status=review_status,
            limit=limit,
            offset=offset,
        )
        _log_request(request, f"Retrieved anomaly list: {len(result.get('rows', []))} rows")
        return result
    except ValueError as exc:
        logger.warning("Anomaly list validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Error fetching anomaly list")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve anomaly list. Check the request ID in server logs.",
        )


@router.post(
    "/feedback",
    summary="Save record feedback",
    responses={
        400: {"description": "The feedback payload is invalid or the record was not found."}
    },
)
def dataset_feedback_route(
    payload: DatasetFeedbackRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    """Persist reviewer feedback for an anomaly row."""
    try:
        result = update_dataset_feedback(db, payload)
        _log_request(request, f"Feedback saved for record {payload.record_id}")
        return result
    except ValueError as exc:
        logger.warning("Feedback error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Error saving feedback")
        raise HTTPException(status_code=500, detail="Failed to save feedback")


@router.get(
    "/report",
    response_model=ReportResponse,
    summary="Get run report",
)
def report_route(
    request: Request,
    dataset_table: Annotated[str | None, Query(pattern=SAFE_IDENTIFIER_PATTERN)] = None,
    run_id: int | None = None,
    db: Session = Depends(get_db),
):
    """Return report metrics for the latest dataset or a specific run identifier."""
    try:
        result = report_data(db, dataset_table=dataset_table, run_id=run_id)
        _log_request(
            request,
            f"Generated report for {dataset_table or 'current'}@run_id={run_id}",
        )
        return result
    except Exception:
        logger.exception("Error generating report")
        raise HTTPException(status_code=500, detail="Failed to generate report")
