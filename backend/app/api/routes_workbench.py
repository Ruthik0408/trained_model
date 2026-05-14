import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError

from app.core.database import check_app_db_connection, get_db
from app.schemas.workbench_schema import (
    BuiltinRuleRequest,
    DatasetFeedbackRequest,
    IsolationReasonRequest,
    IsolationReasonResponse,
    ReportResponse,
    ReviewTableResponse,
    TableInfo,
    WorkbenchRunRequest,
    WorkbenchRunResponse,
)
from app.services.dashboard_service import report_data, review_rows_data, review_table_data
from app.services.llm_reason_service import explain_isolation_anomaly
from app.services.workbench.default_rules import builtin_feature_rules
from app.services.workbench.orchestrator import preview_workbench, run_workbench
from app.services.workbench.result_store import list_saved_datasets, update_dataset_feedback
from app.services.workbench.source_db import list_source_tables, source_connection_status

router = APIRouter(prefix="/api/workbench", tags=["workbench"])
logger = logging.getLogger(__name__)


def _log_request(request: Request, message: str):
    """Helper to log requests with request ID."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.debug(f"[{request_id}] {message}")


@router.get("/tables", response_model=list[TableInfo])
def get_tables(request: Request):
    """Get list of available source database tables."""
    try:
        tables = list_source_tables()
        _log_request(request, f"Retrieved {len(tables)} tables")
        return tables
    except ConnectionError as exc:
        logger.warning(f"Connection error fetching tables: {exc}")
        raise HTTPException(status_code=503, detail="Cannot connect to source database")
    except Exception as exc:
        logger.exception("Error fetching tables")
        raise HTTPException(status_code=500, detail="Failed to retrieve tables")


@router.get("/connection")
def get_connection_status(request: Request):
    """Check connection status to source database."""
    try:
        status = source_connection_status()
        if status.get("connected"):
            _log_request(request, "Connection OK")
            return status
        logger.warning(f"Connection check failed: {status}")
        raise HTTPException(status_code=503, detail=status)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Connection status check failed")
        raise HTTPException(status_code=503, detail={"connected": False, "error": "Connection check failed"})


@router.post("/default-feature-rules")
def get_default_feature_rules(payload: BuiltinRuleRequest, request: Request):
    """Get default feature rules for selected tables."""
    try:
        rules = builtin_feature_rules(payload)
        _log_request(request, f"Generated {len(rules)} feature rules")
        return rules
    except ConnectionError as exc:
        logger.warning(f"Source DB connection error in feature rules: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        logger.warning(f"Validation error in feature rules: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Error generating feature rules")
        raise HTTPException(status_code=500, detail="Failed to generate feature rules")


@router.post("/preview")
def preview_workbench_route(payload: WorkbenchRunRequest, request: Request):
    """Preview workbench execution without saving results."""
    try:
        result = preview_workbench(payload)
        _log_request(request, "Preview completed successfully")
        return result
    except ConnectionError as exc:
        logger.warning(f"Workbench preview connection error: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        logger.warning(f"Workbench preview validation error: {exc}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(exc)}")
    except Exception as exc:
        logger.exception("Workbench preview failed")
        raise HTTPException(status_code=500, detail="Preview execution failed")


@router.post("/run", response_model=WorkbenchRunResponse)
def run_workbench_route(payload: WorkbenchRunRequest, request: Request, db: Session = Depends(get_db)):
    """Execute workbench with specified configuration and save results."""
    try:
        result = run_workbench(db, payload)
        _log_request(request, f"Workbench run completed - Run ID: {result.get('run_id')}")
        return result
    except ConnectionError as exc:
        logger.warning(f"Workbench source DB connection error: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        logger.warning(f"Workbench validation error: {exc}")
        raise HTTPException(status_code=400, detail=f"Validation error: {str(exc)}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Workbench execution failed")
        raise HTTPException(status_code=500, detail=str(exc) or "Workbench execution failed")


@router.get("/datasets")
def dataset_list_route(request: Request, db: Session = Depends(get_db)):
    """Get list of saved datasets."""
    try:
        datasets = list_saved_datasets(db)
        _log_request(request, f"Retrieved {len(datasets)} datasets")
        return datasets
    except OperationalError as exc:
        logger.warning("Application DB is unavailable while fetching datasets: %s", exc)
        status = check_app_db_connection()
        raise HTTPException(
            status_code=503,
            detail=status if not status.get("connected") else {"connected": False, "error": "Application database is unavailable"},
        )
    except Exception as exc:
        logger.exception("Error fetching datasets")
        raise HTTPException(status_code=500, detail="Failed to retrieve datasets")


@router.get("/review-table", response_model=ReviewTableResponse)
def review_table_route(
    request: Request,
    dataset_table: str | None = None,
    anomaly_filter: str = "all",
    limit: int | None = None,
    offset: int = 0,
    run_id: int | None = None,
    db: Session = Depends(get_db),
):
    """Get review table with anomalies for dataset."""
    try:
        result = review_table_data(
            db,
            dataset_table=dataset_table,
            anomaly_filter=anomaly_filter,
            limit=limit,
            offset=offset,
            run_id=run_id,
        )
        _log_request(request, f"Retrieved review table: {dataset_table or 'current'}@{anomaly_filter}")
        return result
    except Exception as exc:
        logger.exception("Error fetching review table")
        raise HTTPException(status_code=500, detail="Failed to retrieve review table")


@router.get("/review-rows")
def review_rows_route(
    request: Request,
    dataset_table: str | None = None,
    anomaly_filter: str = "all",
    limit: int | None = None,
    offset: int = 0,
    run_id: int | None = None,
    db: Session = Depends(get_db),
):
    """Get review rows with anomalies for dataset."""
    try:
        result = review_rows_data(
            db,
            dataset_table=dataset_table,
            anomaly_filter=anomaly_filter,
            limit=limit,
            offset=offset,
            run_id=run_id,
        )
        _log_request(request, f"Retrieved {len(result) if isinstance(result, list) else 'rows'} from review")
        return result
    except Exception as exc:
        logger.exception("Error fetching review rows")
        raise HTTPException(status_code=500, detail="Failed to retrieve review rows")


@router.post("/feedback")
def dataset_feedback_route(payload: DatasetFeedbackRequest, request: Request, db: Session = Depends(get_db)):
    """Save review feedback for a dataset row."""
    try:
        result = update_dataset_feedback(db, payload)
        _log_request(request, f"Feedback saved for record {payload.record_id}")
        return result
    except ValueError as exc:
        logger.warning(f"Feedback error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Error saving feedback")
        raise HTTPException(status_code=500, detail="Failed to save feedback")


@router.post("/isolation-reason", response_model=IsolationReasonResponse)
def isolation_reason_route(payload: IsolationReasonRequest, request: Request):
    """Generate a short local-LLM explanation for an Isolation Forest anomaly."""
    try:
        result = explain_isolation_anomaly(payload)
        _log_request(request, f"Generated IF reason for prediction {payload.prediction_id}")
        return result
    except Exception:
        logger.exception("Error generating Isolation Forest reason")
        raise HTTPException(status_code=500, detail="Failed to generate Isolation Forest reason")

@router.get("/report", response_model=ReportResponse)
def report_route(
    request: Request,
    dataset_table: str | None = None,
    run_id: int | None = None,
    db: Session = Depends(get_db),
):
    """Get detailed report for dataset or specific run."""
    try:
        result = report_data(db, dataset_table=dataset_table, run_id=run_id)
        _log_request(request, f"Generated report for {dataset_table or 'current'}@run_id={run_id}")
        return result
    except Exception as exc:
        logger.exception("Error generating report")
        raise HTTPException(status_code=500, detail="Failed to generate report")
