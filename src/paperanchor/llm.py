"""Provider adapters for generating one answer from the supplied evidence."""

from anthropic import Anthropic
from openai import OpenAI

from .model_config import ModelConfigError, load_model_config


def generate_with_config(system_prompt: str, user_prompt: str) -> str:
    config = load_model_config()
    if config.transport == "anthropic":
        client = Anthropic(api_key=config.api_key, timeout=60.0, max_retries=1)
        response = client.messages.create(
            model=config.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = "".join(
            block.text for block in response.content if block.type == "text"
        )
    else:
        client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=60.0,
            max_retries=1,
        )
        response = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""

    if not content.strip():
        raise RuntimeError("The model returned an empty answer")
    return content
