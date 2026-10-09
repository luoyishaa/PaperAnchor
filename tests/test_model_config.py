from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from openai import APIStatusError

from paperanchor import llm
from paperanchor.model_config import (ModelConfig, ModelConfigError, PROVIDERS,
                                       generation_config, load_model_config, model_extra_body,
                                       vision_model_id)


@pytest.fixture(autouse=True)
def clear_model_environment(monkeypatch) -> None:
    for name in ("PAPERANCHOR_PROVIDER", "PAPERANCHOR_MODEL_TIER", "PAPERANCHOR_MODEL_ID",
                 "PAPERANCHOR_REASONING_EFFORT", "PAPERANCHOR_THINKING",
                 "PAPERANCHOR_VISION_MODEL_ID", "PAPERANCHOR_OLLAMA_BASE_URL",
                 "PAPERANCHOR_OLLAMA_THINKING"):
        monkeypatch.delenv(name, raising=False)
    for preset in PROVIDERS.values():
        if preset.key_variable:
            monkeypatch.delenv(preset.key_variable, raising=False)


def test_generation_limits_preserve_defaults_and_validate_overrides(
        tmp_path: Path, monkeypatch) -> None:
    config = ModelConfig("glm", None, "test-key", "glm-4.7-flash", "openai")
    env_file = tmp_path / "missing.env"
    defaults = generation_config(config, env_file=env_file)
    assert (defaults.max_tokens, defaults.timeout, defaults.max_retries) == (4096, 150.0, 0)
    monkeypatch.setenv("PAPERANCHOR_OLLAMA_CONTEXT_TOKENS", "not-used-by-remote-provider")
    assert generation_config(config, env_file=env_file).context_tokens == 16384
    visual = generation_config(config, visual=True, model="glm-4.6v-flash",
                               env_file=env_file)
    assert (visual.max_tokens, visual.timeout, visual.max_retries) == (1600, 120.0, 0)

    monkeypatch.setenv("PAPERANCHOR_TEXT_MAX_TOKENS", "2048")
    monkeypatch.setenv("PAPERANCHOR_REQUEST_TIMEOUT", "45.5")
    monkeypatch.setenv("PAPERANCHOR_MAX_RETRIES", "2")
    assert generation_config(config, env_file=env_file).max_tokens == 2048
    assert generation_config(config, env_file=env_file).timeout == 45.5
    assert generation_config(config, env_file=env_file).max_retries == 2
    monkeypatch.setenv("PAPERANCHOR_REQUEST_TIMEOUT", "nan")
    with pytest.raises(ModelConfigError, match="PAPERANCHOR_REQUEST_TIMEOUT"):
        generation_config(config, env_file=env_file)


def test_remote_empty_answer_uses_safe_model_request_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = ModelConfig("glm", None, "test-key", "glm-4.7-flash", "openai")

    class EmptyClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=" \n "))])

    monkeypatch.setattr(llm, "load_model_config", lambda: config)
    monkeypatch.setattr(llm, "OpenAI", EmptyClient)
    with pytest.raises(llm.ModelRequestError) as caught:
        llm.generate_with_config("system", "question")
    assert caught.value.provider_code == "empty_answer"


@pytest.mark.parametrize("finish_reason, choices", [("length", True), (None, False)])
def test_remote_incomplete_output_is_not_accepted(tmp_path, monkeypatch, finish_reason, choices):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PAPERANCHOR_TEXT_MAX_TOKENS", "500")
    config = ModelConfig("glm", None, "test-key", "glm-4.7-flash", "openai")

    class Client:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            assert kwargs["max_tokens"] == 500
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason,
                message=SimpleNamespace(content="incomplete answer"))] if choices else [])

    monkeypatch.setattr(llm, "load_model_config", lambda: config)
    monkeypatch.setattr(llm, "OpenAI", Client)
    with pytest.raises(llm.ModelRequestError) as caught:
        llm.generate_with_config("system", "question")
    assert caught.value.provider_code == ("output_limit" if choices else "empty_answer")


def test_local_env_needs_only_the_default_provider_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=local-test-key\n", encoding="utf-8")

    config = load_model_config()

    assert config.provider == "deepseek"
    assert config.model == "deepseek-flash"
    assert config.api_key == "local-test-key"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_provider_presets_select_a_direct_endpoint_and_key(tmp_path: Path, provider: str) -> None:
    preset = PROVIDERS[provider]
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"PAPERANCHOR_PROVIDER={provider}\n"
        "PAPERANCHOR_MODEL_TIER=pro\n"
        + (f"{preset.key_variable}=local-test-key\n" if preset.key_variable else ""),
        encoding="utf-8",
    )

    config = load_model_config(env_file)

    assert config.provider == provider
    assert config.base_url == preset.base_url
    assert config.api_key == ("local-test-key" if preset.key_variable else "")
    assert config.model == preset.pro_model
    assert config.transport == preset.transport


def test_process_environment_overrides_local_file(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "PAPERANCHOR_PROVIDER=deepseek\nDEEPSEEK_API_KEY=file-key\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-key")
    monkeypatch.setenv("PAPERANCHOR_MODEL_ID", "deepseek-flash")

    config = load_model_config(env_file)

    assert config.api_key == "process-key"
    assert config.model == "deepseek-flash"


def test_glm_53_uses_a_supported_low_reasoning_tier(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("PAPERANCHOR_PROVIDER=glm\nPAPERANCHOR_MODEL_TIER=pro\n"
                        "ZHIPU_API_KEY=test-key\n",
                        encoding="utf-8")
    config = load_model_config(env_file)
    assert config.reasoning_effort == "low"
    env_file.write_text("PAPERANCHOR_PROVIDER=glm\nPAPERANCHOR_MODEL_TIER=pro\n"
                        "ZHIPU_API_KEY=test-key\n"
                        "PAPERANCHOR_REASONING_EFFORT=medium\n", encoding="utf-8")
    with pytest.raises(ModelConfigError, match="low, high, or max"):
        load_model_config(env_file)


def test_free_glm_text_and_vision_models_are_separate(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("PAPERANCHOR_PROVIDER=glm\nZHIPU_API_KEY=test-key\n",
                        encoding="utf-8")
    config = load_model_config(env_file)
    assert config.model == "glm-4.7-flash"
    assert config.reasoning_effort is None
    assert config.thinking_mode == "disabled"
    assert model_extra_body(config) == {"thinking": {"type": "disabled"}}
    assert vision_model_id(config, env_file) == "glm-4.6v-flash"
    assert model_extra_body(config, "glm-4.6v-flash") is None

    env_file.write_text("PAPERANCHOR_PROVIDER=glm\nZHIPU_API_KEY=test-key\n"
                        "PAPERANCHOR_THINKING=enabled\n"
                        "PAPERANCHOR_VISION_MODEL_ID=glm-5.3-flash\n",
                        encoding="utf-8")
    config = load_model_config(env_file)
    assert model_extra_body(config) == {"thinking": {"type": "enabled"}}
    assert vision_model_id(config, env_file) == "glm-5.3-flash"


def test_free_glm_rejects_53_only_reasoning_effort(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("PAPERANCHOR_PROVIDER=glm\nZHIPU_API_KEY=test-key\n"
                        "PAPERANCHOR_REASONING_EFFORT=low\n", encoding="utf-8")
    with pytest.raises(ModelConfigError, match="GLM 5.3"):
        load_model_config(env_file)


def test_free_glm_request_disables_thinking_by_default(monkeypatch) -> None:
    config = ModelConfig("glm", "https://open.bigmodel.cn/api/paas/v4", "test-key",
                         "glm-4.7-flash", "openai", thinking_mode="disabled")
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="Answer [E1]"))])

    monkeypatch.setattr(llm, "load_model_config", lambda: config)
    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)
    assert llm.generate_with_config("system", "question") == "Answer [E1]"
    assert captured["model"] == "glm-4.7-flash"
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["client"]["max_retries"] == 0


def test_glm_free_model_limit_codes_have_specific_safe_messages() -> None:
    assert "request-rate limit" in str(llm.ModelRequestError("glm", 429, "1302"))
    assert "temporarily overloaded" in str(llm.ModelRequestError("glm", 429, "1305"))


def test_missing_key_reports_the_required_variable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("PAPERANCHOR_PROVIDER=glm\n", encoding="utf-8")

    with pytest.raises(ModelConfigError, match="ZHIPU_API_KEY"):
        load_model_config(env_file)


def test_openai_compatible_provider_uses_selected_preset(monkeypatch) -> None:
    config = ModelConfig("deepseek", "https://api.deepseek.com", "test-key", "deepseek-flash", "openai")
    calls = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.chat = SimpleNamespace(completions=self)

        def create(self, **kwargs):
            calls["request"] = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Answer [E1]"))]
            )

    monkeypatch.setattr(llm, "load_model_config", lambda: config)
    monkeypatch.setattr(llm, "OpenAI", FakeOpenAI)

    assert llm.generate_with_config("system", "question") == "Answer [E1]"
    assert calls["client"]["base_url"] == "https://api.deepseek.com"
    assert calls["request"]["model"] == "deepseek-flash"
    assert calls["request"]["messages"][0]["content"] == "system"


def test_anthropic_uses_native_messages_api(monkeypatch) -> None:
    config = ModelConfig("anthropic", None, "test-key", "claude-opus-5-5", "anthropic")
    calls = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.messages = self

        def create(self, **kwargs):
            calls["request"] = kwargs
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="Answer [E1]")])

    monkeypatch.setattr(llm, "load_model_config", lambda: config)
    monkeypatch.setattr(llm, "Anthropic", FakeAnthropic)

    assert llm.generate_with_config("system", "question") == "Answer [E1]"
    assert calls["request"]["model"] == "claude-opus-5-5"
    assert calls["request"]["system"] == "system"
    assert calls["request"]["messages"] == [{"role": "user", "content": "question"}]


def test_provider_payment_failure_has_safe_actionable_message(monkeypatch) -> None:
    config = ModelConfig("deepseek", "https://api.deepseek.com", "test-key", "deepseek-flash", "openai")

    class FailingOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **_kwargs):
            response = httpx.Response(402, request=httpx.Request("POST", "https://api.deepseek.com"))
            raise APIStatusError("provider internal request ID 123", response=response, body=None)

    monkeypatch.setattr(llm, "load_model_config", lambda: config)
    monkeypatch.setattr(llm, "OpenAI", FailingOpenAI)

    with pytest.raises(llm.ModelRequestError, match="insufficient balance") as caught:
        llm.generate_with_config("system", "question")
    assert caught.value.status_code == 402
    assert "request ID" not in str(caught.value)


def test_glm_arrears_code_overrides_generic_rate_limit(monkeypatch) -> None:
    config = ModelConfig("glm", "https://open.bigmodel.cn/api/paas/v4", "test-key",
                         "glm-5.3-flash", "openai", "low")

    class FailingOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=self)

        def create(self, **_kwargs):
            response = httpx.Response(429, request=httpx.Request("POST", config.base_url))
            raise APIStatusError("private provider message", response=response,
                                 body={"error": {"code": "1113", "message": "private detail"}})

    monkeypatch.setattr(llm, "load_model_config", lambda: config)
    monkeypatch.setattr(llm, "OpenAI", FailingOpenAI)

    with pytest.raises(llm.ModelRequestError, match="account is in arrears") as caught:
        llm.generate_with_config("system", "question")
    assert caught.value.status_code == 429
    assert caught.value.provider_code == "1113"
    assert "private" not in str(caught.value)
