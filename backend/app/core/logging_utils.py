import json
import logging
from datetime import datetime, timezone


class JsonLogFormatter(logging.Formatter):
    """Formats log records as one-line JSON for log aggregation systems."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging(use_json: bool) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonLogFormatter()
        if use_json
        else logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
