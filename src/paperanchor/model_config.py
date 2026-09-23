"""Resolve official model presets and local credentials for the answer adapter."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values


class ModelConfigError(ValueError):
    """The selected model provider is not configured for a request."""


@dataclass(frozen=True)
class ProviderPreset:
    base_url: str | None
    key_variable: str
    fast_model: str
    pro_model: str
    transport: Literal["openai", "anthropic"] = "openai"


# Model IDs and direct provider endpoints verified against official docs in Sep 2026.
PROVIDERS: dict[str, ProviderPreset] = {
    "deepseek": ProviderPreset(
        "https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-flash", "deepseek-v4-pro"
    ),
    "glm": ProviderPreset(
        "https://open.bigmodel.cn/api/paas/v4",
        "ZHIPU_API_KEY",
        "glm-5.3-flash",
        "glm-5.3",
    ),
    "qwen": ProviderPreset(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
        "qwen3.8-flash",
        "qwen3.8-max",
    ),
    "kimi": ProviderPreset(
        "https://api.moonshot.cn/v1", "MOONSHOT_API_KEY", "kimi-k2.6", "kimi-k3"
    ),
    "openai": ProviderPreset(
        "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-6-luna", "gpt-6-sol"
    ),
    "gemini": ProviderPreset(
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-3.8-flash",
        "gemini-3.1-pro-preview",
    ),
    "anthropic": ProviderPreset(
        None, "ANTHROPIC_API_KEY", "claude-haiku-4-5", "claude-opus-5-5", "anthropic"
    ),
}


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    base_url: str | None
    api_key: str
    model: str
    transport: Literal["openai", "anthropic"]


def load_model_config(env_file: Path | None = None) -> ModelConfig:
    """Read `.env` in the current directory, with process variables taking precedence."""
    values = dotenv_values(env_file or Path.cwd() / ".env")

    def setting(name: str) -> str:
        value = os.environ[name] if name in os.environ else values.get(name)
        return (value or "").strip()

    provider_name = setting("PAPERANCHOR_PROVIDER").lower() or "deepseek"
    preset = PROVIDERS.get(provider_name)
    if preset is None:
        raise ModelConfigError(
            f"Unknown provider {provider_name!r}. Choose: {', '.join(PROVIDERS)}."
        )

    tier = setting("PAPERANCHOR_MODEL_TIER").lower() or "fast"
    if tier not in ("fast", "pro"):
        raise ModelConfigError("PAPERANCHOR_MODEL_TIER must be 'fast' or 'pro'.")

    api_key = setting(preset.key_variable)
    if not api_key:
        raise ModelConfigError(
            f"Set {preset.key_variable} for {provider_name} in a local .env file "
            "or in the process environment. Never put a key in .env.example."
        )

    model = setting("PAPERANCHOR_MODEL_ID") or (
        preset.fast_model if tier == "fast" else preset.pro_model
    )
    return ModelConfig(provider_name, preset.base_url, api_key, model, preset.transport)
