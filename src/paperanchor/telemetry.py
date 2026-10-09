"""Stage timing without recording request arguments or exception messages."""

from functools import wraps
import json
import logging
from time import perf_counter


logger = logging.getLogger("paperanchor.operations")


def timed(stage: str):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            started = perf_counter()
            fields = {"stage": stage, "outcome": "success"}
            try:
                result = function(*args, **kwargs)
                if isinstance(result, (tuple, list)):
                    fields["result_count"] = len(result)
                return result
            except Exception as error:
                fields.update(outcome="error", error_type=type(error).__name__)
                raise
            finally:
                fields["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
                logger.info(json.dumps(fields, sort_keys=True))
        return measured
    return decorate
