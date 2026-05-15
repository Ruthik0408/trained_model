from typing import Any


class WorkbenchValidationError(ValueError):
    """Validation error with optional user-facing suggestion and structured details."""

    def __init__(self, message: str, *, suggestion: str = "", details: dict[str, Any] | None = None):
        self.message = message
        self.suggestion = suggestion
        self.details = details or {}
        super().__init__(message)

    def to_http_detail(self) -> dict[str, Any]:
        detail: dict[str, Any] = {"error": self.message}
        if self.suggestion:
            detail["suggestion"] = self.suggestion
        if self.details:
            detail["details"] = self.details
        return detail
