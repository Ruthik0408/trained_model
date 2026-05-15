import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class WorkbenchValidationError(ValueError):
    """Validation error with optional user-facing suggestion and structured details."""

    def __init__(
        self,
        message: str,
        *,
        suggestion: str = "",
        details: Optional[Dict[str, Any]] = None,
        error_code: str = "VALIDATION_ERROR",
    ):
        self.message = message
        self.suggestion = suggestion
        self.details = details or {}
        self.error_code = error_code
        super().__init__(message)

    def to_http_detail(self) -> Dict[str, Any]:
        """Convert to HTTP response format."""
        detail: Dict[str, Any] = {
            "error": self.message,
            "error_code": self.error_code,
        }
        if self.suggestion:
            detail["suggestion"] = self.suggestion
        if self.details:
            detail["details"] = self.details
        return detail


class WorkbenchConnectionError(Exception):
    """Database or upstream service connection error."""

    def __init__(self, message: str, service: str = "unknown"):
        self.message = message
        self.service = service
        super().__init__(message)

    def to_http_detail(self) -> Dict[str, Any]:
        """Convert to HTTP response format (503 Service Unavailable)."""
        return {
            "error": f"Cannot connect to {self.service}",
            "error_code": "CONNECTION_ERROR",
            "details": {"service": self.service, "message": self.message},
        }


class WorkbenchExecutionError(Exception):
    """Runtime error during workbench execution."""

    def __init__(self, message: str, stage: str = "unknown"):
        self.message = message
        self.stage = stage
        super().__init__(message)

    def to_http_detail(self) -> Dict[str, Any]:
        """Convert to HTTP response format (500 Internal Server Error)."""
        return {
            "error": "Workbench execution failed",
            "error_code": "EXECUTION_ERROR",
            "details": {"stage": self.stage, "message": self.message},
        }


def format_error_response(
    error: Exception, status_code: int = 500
) -> Dict[str, Any]:
    """
    Format any exception into a consistent error response.
    
    Args:
        error: The exception to format
        status_code: HTTP status code
        
    Returns:
        Consistent error response dict with error, error_code, and details
    """
    if isinstance(error, WorkbenchValidationError):
        return error.to_http_detail()
    elif isinstance(error, WorkbenchConnectionError):
        return error.to_http_detail()
    elif isinstance(error, WorkbenchExecutionError):
        return error.to_http_detail()
    elif isinstance(error, ValueError):
        return {
            "error": str(error),
            "error_code": "INVALID_INPUT",
        }
    elif isinstance(error, ConnectionError):
        return {
            "error": "Connection failed",
            "error_code": "CONNECTION_ERROR",
            "details": {"message": str(error)},
        }
    else:
        # Generic error
        logger.exception("Unhandled exception: %s", error)
        return {
            "error": "An unexpected error occurred",
            "error_code": "INTERNAL_ERROR",
        }
