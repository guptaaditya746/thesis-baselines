"""The model every baseline uses: the same LiteLLM gateway (llm-setup) and alias as the evqa run, so a comparison is
about the method, not the model.

Environment (the same names evqa uses):
  LITELLM_BASE_URL     default http://127.0.0.1:4000/v1
  LITELLM_MASTER_KEY   the gateway key (never written to any output file)
"""
from __future__ import annotations

import os
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:4000/v1"


def base_url() -> str:
    return os.environ.get("LITELLM_BASE_URL", DEFAULT_BASE).rstrip("/")


def litellm_name(model: str) -> str:
    """LiteLLM name of a gateway alias: the gateway speaks the OpenAI API, so 'orch-model' -> 'openai/orch-model'."""
    return model if "/" in model else f"openai/{model}"


def litellm_params(model: str, think: bool = False, temperature: float = 0.0, max_tokens: int = 4096
                   ) -> dict[str, Any]:
    """litellm_params for one gateway model. think=False sends enable_thinking=false, as evqa's lm.py does."""
    params: dict[str, Any] = {"model": litellm_name(model), "api_base": base_url(),
                              "api_key": os.environ.get("LITELLM_MASTER_KEY", "") or "none",
                              "temperature": temperature, "max_tokens": max_tokens}
    if not think:
        params["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return params


def router_config(model: str, think: bool = False, temperature: float = 0.0, max_tokens: int = 4096
                  ) -> dict[str, Any]:
    """A LiteLLM router config (PaperQA2's llm_config shape) with one model."""
    name = litellm_name(model)
    return {"name": name, "model_list": [{"model_name": name,
                                          "litellm_params": litellm_params(model, think, temperature, max_tokens)}]}


def public(params: dict[str, Any]) -> dict[str, Any]:
    """A copy without secrets, for run_info.json."""
    def scrub(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: ("<hidden>" if k == "api_key" else scrub(v)) for k, v in x.items()}
        if isinstance(x, list):
            return [scrub(v) for v in x]
        return x
    return scrub(params)
