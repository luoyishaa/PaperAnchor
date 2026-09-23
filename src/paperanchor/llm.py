"""One configurable Chat Completions adapter; no provider details enter RAG code."""

import os

from openai import OpenAI


class ModelConfigError(ValueError):
    pass


def generate_with_config(system_prompt: str, user_prompt: str) -> str:
    base_url = os.getenv("PAPERANCHOR_BASE_URL")
    api_key = os.getenv("PAPERANCHOR_API_KEY")
    model = os.getenv("PAPERANCHOR_MODEL")
    if not all((base_url, api_key, model)):
        raise ModelConfigError(
            "Set PAPERANCHOR_BASE_URL, PAPERANCHOR_API_KEY and PAPERANCHOR_MODEL "
            "for a Chat Completions-compatible provider. See README.md."
        )

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=1)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("The model returned an empty answer")
    return content
