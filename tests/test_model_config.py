from pathlib import Path
from types import SimpleNamespace

import pytest

from paperanchor import llm
from paperanchor.model_config import ModelConfig, ModelConfigError, PROVIDERS, load_model_config


@pytest.fixture(autouse=True)
def clear_model_environment(monkeypatch) -> None:
    for name in ("PAPERANCHOR_PROVIDER", "PAPERANCHOR_MODEL_TIER", "PAPERANCHOR_MODEL_ID"):
        monkeypatch.delenv(name, raising=False)
    for preset in PROVIDERS.values():
        monkeypatch.delenv(preset.key_variable, raising=False)


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
        f"{preset.key_variable}=local-test-key\n",
        encoding="utf-8",
    )

    config = load_model_config(env_file)

    assert config.provider == provider
    assert config.base_url == preset.base_url
    assert config.api_key == "local-test-key"
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
