import json

import httpx
import pytest

from paperanchor import ollama
from paperanchor.llm import ModelRequestError
from paperanchor.model_config import ModelConfig, ModelConfigError, load_model_config


CONFIG = ModelConfig("ollama", "http://127.0.0.1:11434", "", "qwen3:8b", "ollama")


def fake_server(monkeypatch, capabilities, *, status=200, reply=None):
    requests = []
    def handler(request):
        payload = json.loads(request.content)
        requests.append((request.url.path, payload))
        if request.url.path == "/api/show":
            return httpx.Response(status, json={"capabilities": capabilities})
        return httpx.Response(200, json=reply or {
            "message": {"thinking": "private intermediate text", "content": "0.8 [E1]"},
            "done": True, "done_reason": "stop"})
    original = httpx.Client
    monkeypatch.setattr(ollama.httpx, "Client", lambda **kwargs:
        original(transport=httpx.MockTransport(handler), **kwargs))
    monkeypatch.setattr(ollama, "get_setting", lambda _name: "auto")
    return requests


def test_local_model_requires_no_key_and_uses_configured_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERANCHOR_PROVIDER", "ollama")
    monkeypatch.setenv("PAPERANCHOR_OLLAMA_BASE_URL", "http://127.0.0.1:11435")
    monkeypatch.delenv("PAPERANCHOR_MODEL_ID", raising=False)
    config = load_model_config(tmp_path / "missing.env")
    assert config.api_key == ""
    assert config.model == "qwen3:8b"
    assert config.base_url.endswith(":11435")


def test_text_completion_returns_only_final_answer(monkeypatch):
    requests = fake_server(monkeypatch, ["completion", "thinking"])
    assert ollama.complete_local(CONFIG, "system", "question") == "0.8 [E1]"
    request = requests[-1][1]
    assert request["stream"] is False
    assert "images" not in request["messages"][-1]
    assert "think" not in request
    assert "temperature" not in request["options"]


def test_local_output_and_context_budgets_use_configuration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    requests = fake_server(monkeypatch, ["completion"])
    monkeypatch.setenv("PAPERANCHOR_TEXT_MAX_TOKENS", "2000")
    monkeypatch.setenv("PAPERANCHOR_OLLAMA_CONTEXT_TOKENS", "8000")
    ollama.complete_local(CONFIG, "system", "question")
    assert requests[-1][1]["options"] == {"num_ctx": 8000, "num_predict": 2000}
    monkeypatch.setenv("PAPERANCHOR_TEXT_MAX_TOKENS", "8000")
    with pytest.raises(ModelConfigError, match="smaller than"):
        ollama.complete_local(CONFIG, "system", "question")


def test_vision_keeps_image_order_and_separates_model_roles(monkeypatch):
    requests = fake_server(monkeypatch, ["completion", "thinking", "vision"])
    content = [{"type":"text", "text":"[E1] Full context"},
        {"type":"image_url", "image_url":{"url":"data:image/jpeg;base64,YQ=="}},
        {"type":"text", "text":"[E1] Target crop"},
        {"type":"image_url", "image_url":{"url":"data:image/jpeg;base64,Yg=="}}]
    assert ollama.complete_local(CONFIG, "system", content, model="qwen3-vl:4b-thinking") == "0.8 [E1]"
    assert requests[0][1]["model"] == "qwen3-vl:4b-thinking"
    request = requests[-1][1]
    assert request["model"] == "qwen3-vl:4b-thinking"
    assert request["messages"][-1]["images"] == ["YQ==", "Yg=="]
    assert "Image 1" in request["messages"][-1]["content"]
    assert "Image 2" in request["messages"][-1]["content"]


def test_thinking_text_model_is_rejected_before_image_generation(monkeypatch):
    requests = fake_server(monkeypatch, ["completion", "thinking"])
    with pytest.raises(ModelConfigError, match="no image input"):
        ollama.complete_local(CONFIG, "system", [
            {"type":"image_url", "image_url":{"url":"data:image/jpeg;base64,YQ=="}}])
    assert len(requests) == 1


def test_missing_model_is_actionable(monkeypatch):
    fake_server(monkeypatch, [], status=404)
    with pytest.raises(ModelConfigError, match="not installed"):
        ollama.complete_local(CONFIG, "system", "question")


@pytest.mark.parametrize("reply", [
    {"message":{"content":"", "thinking":"unfinished"}},
    {"message":{"content":"partial answer"}, "done_reason":"length"},
])
def test_incomplete_local_generation_is_not_a_success(monkeypatch, reply):
    fake_server(monkeypatch, ["completion", "thinking"], reply=reply)
    with pytest.raises(ModelRequestError) as caught:
        ollama.complete_local(CONFIG, "system", "question")
    assert caught.value.provider_code in ("empty_answer", "output_limit")


@pytest.mark.parametrize("reply", [{"message": None}, {"message": "bad response"}])
def test_malformed_local_response_is_a_safe_service_error(monkeypatch, reply):
    fake_server(monkeypatch, ["completion"], reply=reply)
    with pytest.raises(ModelRequestError) as caught:
        ollama.complete_local(CONFIG, "system", "question")
    assert caught.value.provider_code == "invalid_response"


def test_requested_thinking_is_sent_only_for_capable_model(monkeypatch):
    requests = fake_server(monkeypatch, ["completion", "thinking"])
    monkeypatch.setattr(ollama, "get_setting", lambda _name: "true")
    ollama.complete_local(CONFIG, "system", "question")
    assert requests[-1][1]["think"] is True


def test_requested_thinking_rejects_incapable_model(monkeypatch):
    requests = fake_server(monkeypatch, ["completion"])
    monkeypatch.setattr(ollama, "get_setting", lambda _name: "true")
    with pytest.raises(ModelConfigError, match="does not support thinking"):
        ollama.complete_local(CONFIG, "system", "question")
    assert len(requests) == 1


def test_unavailable_local_service_has_project_error(monkeypatch):
    def unavailable(_request):
        raise httpx.ConnectError("private network detail")
    original = httpx.Client
    monkeypatch.setattr(ollama.httpx, "Client", lambda **kwargs:
        original(transport=httpx.MockTransport(unavailable), **kwargs))
    with pytest.raises(ModelRequestError) as caught:
        ollama.complete_local(CONFIG, "system", "question")
    assert caught.value.provider == "ollama"
    assert "private network detail" not in str(caught.value)
