
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from app.core.database import check_app_db_connection, get_db
from app.core.errors import WorkbenchValidationError
from app.schemas.workbench_schema import (
    BuiltinRuleRequest,
    DatasetFeedbackRequest,
    ReportResponse,
    ReviewTableResponse,
    TableInfo,
    WorkbenchRunRequest,
    WorkbenchRunResponse,
)
from app.services.dashboard_service import report_data, review_rows_data, review_table_data
from app.services.workbench.default_rules import builtin_feature_rules
from app.services.workbench.orchestrator import preview_workbench, run_workbench
from app.services.workbench.result_store import list_saved_datasets, update_dataset_feedback
from app.services.workbench.source_db import list_source_tables, source_connection_status

router = APIRouter(prefix="/api/workbench", tags=["workbench"])
logger = logging.getLogger(__name__)


def _log_request(request: Request, message: str) -> None:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.debug("[%s] %s", request_id, message)


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
    """Return available source tables and columns for workbench configuration."""
    try:
        tables = list_source_tables()
        _log_request(request, f"Retrieved {len(tables)} tables")
        return tables
    except ConnectionError as exc:
        logger.warning("Connection error fetching tables: %s", exc)
        raise HTTPException(status_code=503, detail="Cannot connect to source database")
    except Exception:
        logger.exception("Error fetching tables")
        raise HTTPException(status_code=500, detail="Failed to retrieve tables")


@router.get(
    "/connection",
    summary="Check source connectivity",
    responses={503: {"description": "Source database connection failed."}},
)
def get_connection_status(request: Request):
    """Check whether the source database is reachable before preview or run actions."""
    try:
        status = source_connection_status()
        if status.get("connected"):
            _log_request(request, "Connection OK")
            return status
        logger.warning("Connection check failed: %s", status)
        raise HTTPException(status_code=503, detail=status)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Connection status check failed")
        raise HTTPException(
            status_code=503,
            detail={"connected": False, "error": "Connection check failed"},
        )


@router.post(
    "/default-feature-rules",
    summary="Suggest built-in feature rules",
    responses={
        400: {"description": "Join/date configuration is invalid for rule generation."},
        503: {"description": "Source database is unavailable."},
    },
)
def get_default_feature_rules(payload: BuiltinRuleRequest, request: Request):
    """Suggest built-in feature rules inferred from the selected tables and join path."""
    try:
        rules = builtin_feature_rules(payload)
        _log_request(request, f"Generated {len(rules)} feature rules")
        return rules
    except ConnectionError as exc:
        logger.warning("Source DB connection error in feature rules: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        logger.warning("Validation error in feature rules: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Error generating feature rules")
        raise HTTPException(status_code=500, detail="Failed to generate feature rules")


@router.post(
    "/preview",
    summary="Preview workbench join output",
    responses={
        400: {"description": "The request is valid JSON but fails workbench validation."},
        503: {"description": "Source database is unavailable."},
    },
)
def preview_workbench_route(payload: WorkbenchRunRequest, request: Request):
    """Preview the joined dataset without training a model or writing result rows."""
    try:
        result = preview_workbench(payload)
        _log_request(request, "Preview completed successfully")
        return result
    except ConnectionError as exc:
        logger.warning("Workbench preview connection error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except WorkbenchValidationError as exc:
        logger.warning("Workbench preview structured validation error: %s", exc.message)
        raise HTTPException(status_code=400, detail=exc.to_http_detail())
    except ValueError as exc:
        logger.warning("Workbench preview validation error: %s", exc)
        raise HTTPException(status_code=400, detail=f"Validation error: {str(exc)}")
    except Exception:
        logger.exception("Workbench preview failed")
        raise HTTPException(status_code=500, detail="Preview execution failed")


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
        result = run_workbench(db, payload)
        _log_request(request, f"Workbench run completed — Run ID: {result.get('run_id')}")
        return result
    except ConnectionError as exc:
        logger.warning("Workbench source DB connection error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except WorkbenchValidationError as exc:
        logger.warning("Workbench structured validation error: %s", exc.message)
        raise HTTPException(status_code=400, detail=exc.to_http_detail())
    except ValueError as exc:
        logger.warning("Workbench validation error: %s", exc)
        raise HTTPException(status_code=400, detail=f"Validation error: {str(exc)}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Workbench execution failed")
        raise HTTPException(
            status_code=500,
            detail=str(exc) or "Workbench execution failed",
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


@router.get(
    "/review-table",
    response_model=ReviewTableResponse,
    summary="Get review-table summary",
)
def review_table_route(
    request: Request,
    dataset_table: str | None = None,
    anomaly_filter: str = "all",
    limit: int | None = None,
    offset: int = 0,
    run_id: int | None = None,
    db: Session = Depends(get_db),
):
    """Return paginated anomaly rows plus aggregate review-table totals."""
    try:
        result = review_table_data(
            db,
            dataset_table=dataset_table,
            anomaly_filter=anomaly_filter,
            limit=limit,
            offset=offset,
            run_id=run_id,
        )
        _log_request(
            request,
            f"Retrieved review table: {dataset_table or 'current'}@{anomaly_filter}",
        )
        return result
    except Exception:
        logger.exception("Error fetching review table")
        raise HTTPException(status_code=500, detail="Failed to retrieve review table")


@router.get("/review-rows", summary="Get review rows")
def review_rows_route(
    request: Request,
    dataset_table: str | None = None,
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
        raise HTTPException(status_code=500, detail="Failed to retrieve review rows")


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
    dataset_table: str | None = None,
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
