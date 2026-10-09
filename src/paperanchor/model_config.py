"""Resolve official model presets and local credentials for the answer adapter."""

from dataclasses import dataclass
import math
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
    transport: Literal["openai", "anthropic", "ollama"] = "openai"


# Model IDs and direct provider endpoints verified against official docs in Sep 2026.
PROVIDERS: dict[str, ProviderPreset] = {
    "ollama": ProviderPreset("http://127.0.0.1:11434", "", "qwen3:8b", "qwen3:8b", "ollama"),
    "deepseek": ProviderPreset(
        "https://api.deepseek.com", "DEEPSEEK_API_KEY", "deepseek-flash", "deepseek-v4-pro"
    ),
    "glm": ProviderPreset(
        "https://open.bigmodel.cn/api/paas/v4",
        "ZHIPU_API_KEY",
        "glm-4.7-flash",
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
    transport: Literal["openai", "anthropic", "ollama"]
    reasoning_effort: str | None = None
    thinking_mode: str | None = None


@dataclass(frozen=True)
class GenerationConfig:
    max_tokens: int
    timeout: float
    max_retries: int
    context_tokens: int = 16384


def get_setting(name: str, env_file: Path | None = None) -> str:
    """Read a local setting with process environment taking precedence."""
    if name in os.environ:
        return os.environ[name].strip()
    values = dotenv_values(env_file or Path.cwd() / ".env")
    value = values.get(name)
    return (value or "").strip()


def generation_config(config: ModelConfig, *, visual: bool = False,
                      model: str | None = None,
                      env_file: Path | None = None) -> GenerationConfig:
    """Resolve request limits, preserving each provider's current defaults."""
    selected = model or config.model
    tokens = 4096 if config.transport == "ollama" else 1600 if visual else 4096
    timeout = (300.0 if config.transport == "ollama" else
               120.0 if visual else 60.0 if config.transport == "anthropic" else 150.0)
    retries = 0 if config.transport == "ollama" or selected in ("glm-4.7-flash", "glm-4.6v-flash") else 1
    settings = (
        ("PAPERANCHOR_VISION_MAX_TOKENS" if visual else "PAPERANCHOR_TEXT_MAX_TOKENS",
         tokens, int, 1),
        ("PAPERANCHOR_REQUEST_TIMEOUT", timeout, float, 0),
        ("PAPERANCHOR_MAX_RETRIES", retries, int, 0),
    )
    if config.transport == "ollama":
        settings += (("PAPERANCHOR_OLLAMA_CONTEXT_TOKENS", 16384, int, 1),)
    values = []
    for name, default, convert, minimum in settings:
        raw = ("" if config.transport == "ollama" and name == "PAPERANCHOR_MAX_RETRIES"
               else get_setting(name, env_file))
        try:
            value = convert(raw) if raw else default
        except ValueError as error:
            raise ModelConfigError(f"{name} must be a number.") from error
        if (not math.isfinite(value) or value < minimum
                or (name == "PAPERANCHOR_REQUEST_TIMEOUT" and value == 0)):
            raise ModelConfigError(f"{name} must be positive" if minimum or name == "PAPERANCHOR_REQUEST_TIMEOUT" else
                                   f"{name} must be nonnegative")
        values.append(value)
    limits = GenerationConfig(*values)
    if config.transport == "ollama" and limits.max_tokens >= limits.context_tokens:
        raise ModelConfigError("Output token budget must be smaller than Ollama context token budget.")
    return limits


def load_model_config(env_file: Path | None = None) -> ModelConfig:
    """Read `.env` in the current directory, with process variables taking precedence."""
    def setting(name: str) -> str:
        return get_setting(name, env_file)

    provider_name = setting("PAPERANCHOR_PROVIDER").lower() or "deepseek"
    preset = PROVIDERS.get(provider_name)
    if preset is None:
        raise ModelConfigError(
            f"Unknown provider {provider_name!r}. Choose: {', '.join(PROVIDERS)}."
        )

    tier = setting("PAPERANCHOR_MODEL_TIER").lower() or "fast"
    if tier not in ("fast", "pro"):
        raise ModelConfigError("PAPERANCHOR_MODEL_TIER must be 'fast' or 'pro'.")

    api_key = setting(preset.key_variable) if preset.key_variable else ""
    if preset.key_variable and not api_key:
        raise ModelConfigError(
            f"Set {preset.key_variable} for {provider_name} in a local .env file "
            "or in the process environment. Never put a key in .env.example."
        )

    model = setting("PAPERANCHOR_MODEL_ID") or (
        preset.fast_model if tier == "fast" else preset.pro_model
    )
    effort = setting("PAPERANCHOR_REASONING_EFFORT").lower()
    thinking = setting("PAPERANCHOR_THINKING").lower()
    if provider_name == "glm" and model.startswith("glm-5.3"):
        effort = effort or "low"
        if effort not in ("low", "high", "max"):
            raise ModelConfigError(
                "GLM 5.3 reasoning effort must be low, high, or max."
            )
    elif effort:
        raise ModelConfigError(
            "PAPERANCHOR_REASONING_EFFORT currently applies to GLM 5.3 models."
        )
    if provider_name == "glm" and model == "glm-4.7-flash":
        thinking = thinking or "disabled"
        if thinking not in ("enabled", "disabled"):
            raise ModelConfigError("PAPERANCHOR_THINKING must be enabled or disabled.")
    elif thinking:
        raise ModelConfigError(
            "PAPERANCHOR_THINKING currently applies to glm-4.7-flash."
        )
    base_url = (setting("PAPERANCHOR_OLLAMA_BASE_URL") or preset.base_url
                if provider_name == "ollama" else preset.base_url)
    return ModelConfig(provider_name, base_url, api_key, model,
                       preset.transport, effort or None, thinking or None)


def model_extra_body(config: ModelConfig, model: str | None = None) -> dict | None:
    """Keep provider-specific request controls beside the selected model."""
    selected = model or config.model
    if config.provider == "glm" and selected == "glm-4.7-flash":
        return {"thinking": {"type": config.thinking_mode or "disabled"}}
    if (config.provider == "glm" and selected.startswith("glm-5.3")
            and config.reasoning_effort):
        return {"reasoning_effort": config.reasoning_effort}
    return None


def vision_model_id(config: ModelConfig, env_file: Path | None = None) -> str:
    """Pick an image-capable model independently of the text model."""
    override = get_setting("PAPERANCHOR_VISION_MODEL_ID", env_file)
    if override:
        return override
    if config.provider == "ollama":
        # Capabilities are checked against the running server, never inferred
        # from a model name. A text-only default produces an actionable error.
        return config.model
    if config.provider == "glm":
        return "glm-4.6v-flash"
    if config.provider == "deepseek":
        return "deepseek-flash"
    raise ModelConfigError("Visual PDF reading requires a GLM, DeepSeek, or Ollama provider.")
