import json
import logging

import pytest

from paperanchor.telemetry import timed


def test_success_timing_counts_results_without_logging_arguments(caplog):
    @timed("test.retrieval")
    def retrieve(question, *, api_key):
        return [question, api_key]

    with caplog.at_level(logging.INFO, logger="paperanchor.operations"):
        assert retrieve("private-paper-text", api_key="secret-key") == ["private-paper-text", "secret-key"]
    record = json.loads(caplog.records[-1].message)
    assert record["result_count"] == 2
    assert record["outcome"] == "success"
    assert record["elapsed_ms"] >= 0
    assert "private-paper-text" not in caplog.text
    assert "secret-key" not in caplog.text


def test_failure_timing_preserves_exception_without_logging_its_message(caplog):
    error = ValueError("private-provider-response")

    @timed("test.model")
    def fail():
        raise error

    with caplog.at_level(logging.INFO, logger="paperanchor.operations"):
        with pytest.raises(ValueError) as caught:
            fail()
    assert caught.value is error
    record = json.loads(caplog.records[-1].message)
    assert record["outcome"] == "error"
    assert record["error_type"] == "ValueError"
    assert "private-provider-response" not in caplog.text
