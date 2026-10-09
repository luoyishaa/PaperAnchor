"""Provider adapters for generating one answer from the supplied evidence."""

from anthropic import Anthropic, APIError as AnthropicAPIError
from openai import OpenAI, APIError as OpenAIAPIError

from .model_config import (ModelConfigError, generation_config, load_model_config,
                            model_extra_body)
from .telemetry import timed


class ModelRequestError(RuntimeError):
    """A provider failure with a safe message for local UI and evaluation logs."""

    def __init__(self, provider: str, status_code: int | None,
                 provider_code: str | None = None):
        self.provider = provider
        self.status_code = status_code
        self.provider_code = provider_code
        guidance = {
            400: "rejected the request. Check the selected model and input length.",
            401: "rejected the API key. Check this provider's key in your local .env file.",
            402: "reports insufficient balance. Add credit or select another provider.",
            403: "denied access. Check whether this account can use the selected model.",
            429: "is rate limited or out of quota. Wait or select another provider.",
        }.get(status_code)
        if provider == "glm" and provider_code == "1113":
            guidance = "reports that the account is in arrears. Add credit in the official GLM console."
        elif provider == "glm" and provider_code == "1302":
            guidance = "has reached the account's request-rate limit. Reduce request frequency and retry later."
        elif provider == "glm" and provider_code == "1305":
            guidance = "is temporarily overloaded. Retry later with a longer interval."
        elif provider_code == "empty_answer":
            guidance = "returned no final answer. Check the model output budget or thinking setting."
        elif provider_code == "output_limit":
            guidance = "exhausted its output budget. The incomplete answer was discarded."
        if guidance is None:
            guidance = ("is temporarily unavailable. Try again later."
                        if status_code and status_code >= 500
                        else "could not be reached. Check the provider connection.")
        super().__init__(f"{provider} {guidance}")


def provider_error_code(error: OpenAIAPIError | AnthropicAPIError) -> str | None:
    """Extract a provider's stable code without exposing its raw error body."""
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return None
    payload = body.get("error", body)
    if not isinstance(payload, dict):
        return None
    code = payload.get("code")
    return str(code) if isinstance(code, (str, int)) else None


@timed("model.text")
def generate_with_config(system_prompt: str, user_prompt: str) -> str:
    config = load_model_config()
    if config.transport == "ollama":
        from .ollama import complete_local
        return complete_local(config, system_prompt, user_prompt)
    limits = generation_config(config)
    try:
        if config.transport == "anthropic":
            client = Anthropic(api_key=config.api_key, timeout=limits.timeout,
                               max_retries=limits.max_retries)
            response = client.messages.create(
                model=config.model,
                max_tokens=limits.max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content = "".join(
                block.text for block in response.content if block.type == "text"
            )
            if getattr(response, "stop_reason", None) == "max_tokens":
                raise ModelRequestError(config.provider, 503, "output_limit")
        else:
            client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                timeout=limits.timeout,
                max_retries=limits.max_retries,
            )
            request = {"model": config.model, "max_tokens": limits.max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]}
            extra = model_extra_body(config)
            if extra is not None:
                request["extra_body"] = extra
            response = client.chat.completions.create(**request)
            if not response.choices:
                raise ModelRequestError(config.provider, 503, "empty_answer")
            if getattr(response.choices[0], "finish_reason", None) == "length":
                raise ModelRequestError(config.provider, 503, "output_limit")
            content = response.choices[0].message.content or ""
    except (OpenAIAPIError, AnthropicAPIError) as error:
        raise ModelRequestError(config.provider, getattr(error, "status_code", None),
                                provider_error_code(error)) from error

    if not isinstance(content, str) or not content.strip():
        raise ModelRequestError(config.provider, 503, "empty_answer")
    return content
