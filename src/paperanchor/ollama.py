"""Local Ollama transport with runtime capability checks and final-answer output."""

import httpx

from .model_config import ModelConfig, ModelConfigError, generation_config, get_setting


def complete_local(config: ModelConfig, system: str, content: str | list[dict], *,
                   model: str | None = None) -> str:
    # Import lazily to keep the project-level error shared with other adapters.
    from .llm import ModelRequestError

    selected = model or config.model
    limits = generation_config(config, visual=not isinstance(content, str), model=selected)
    image_data = []
    if isinstance(content, str):
        text = content
    else:
        parts = []
        for part in content:
            if part["type"] == "text":
                parts.append(part["text"])
            elif part["type"] == "image_url":
                url = part["image_url"]["url"]
                if not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError("Local visual reading requires inline image data.")
                image_data.append(url.split(";base64,", 1)[1])
                parts.append(f"[Image {len(image_data)}, in the supplied image order]")
            else:
                raise ValueError("Unsupported local model input.")
        text = "\n".join(parts)
    thinking = get_setting("PAPERANCHOR_OLLAMA_THINKING").lower() or "auto"
    if thinking not in ("auto", "true", "false"):
        raise ModelConfigError("PAPERANCHOR_OLLAMA_THINKING must be auto, true, or false.")
    try:
        with httpx.Client(base_url=config.base_url.rstrip("/") + "/", timeout=limits.timeout,
                          trust_env=False) as client:
            metadata = client.post("api/show", json={"model": selected})
            if metadata.status_code == 404:
                raise ModelConfigError(f"Ollama model {selected!r} is not installed. Pull it first.")
            metadata.raise_for_status()
            capabilities = metadata.json().get("capabilities", [])
            if "completion" not in capabilities:
                raise ModelConfigError(f"Ollama model {selected!r} cannot generate answers.")
            if image_data and "vision" not in capabilities:
                raise ModelConfigError(
                    f"Ollama model {selected!r} has no image input. Select an installed "
                    "vision model with PAPERANCHOR_VISION_MODEL_ID; thinking alone is not vision.")
            if thinking == "true" and "thinking" not in capabilities:
                raise ModelConfigError(f"Ollama model {selected!r} does not support thinking.")
            user = {"role": "user", "content": text}
            if image_data:
                user["images"] = image_data
            request = {"model": selected, "stream": False,
                       "messages": [{"role": "system", "content": system}, user],
                       # Retain sampling defaults stored with each model.
                       # Greedy decoding can trap reasoning models in repetition.
                        "options": {"num_ctx": limits.context_tokens,
                                    "num_predict": limits.max_tokens}}
            if thinking != "auto":
                request["think"] = thinking == "true"
            response = client.post("api/chat", json=request)
            response.raise_for_status()
            result = response.json()
    except httpx.HTTPStatusError as error:
        raise ModelRequestError("ollama", error.response.status_code) from error
    except httpx.RequestError as error:
        raise ModelRequestError("ollama", None) from error
    except ModelConfigError:
        raise
    except (ValueError, TypeError, AttributeError) as error:
        raise ModelRequestError("ollama", 503, "invalid_response") from error
    if not isinstance(result, dict) or not isinstance(result.get("message"), dict):
        raise ModelRequestError("ollama", 503, "invalid_response")
    answer = result["message"].get("content", "")
    if result.get("done_reason") == "length":
        raise ModelRequestError("ollama", 503, "output_limit")
    if not isinstance(answer, str) or not answer.strip():
        raise ModelRequestError("ollama", 503, "empty_answer")
    return answer
