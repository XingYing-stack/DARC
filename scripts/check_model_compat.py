#!/usr/bin/env python3
"""
Quick compatibility checks for a local HF model path used by this repo.

Checks:
- Tokenizer/config load with and without trust_remote_code
- Whether chat_template is present/usable
- Renders sample prompts matching this repo's vLLM server usage
- Optional: attempt to initialize vLLM engine and run 1 short generation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _render_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _load_tokenizer(model_path: str, trust_remote_code: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def _load_config(model_path: str, trust_remote_code: bool):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)


def _make_messages(question: str, *, categorical: bool) -> List[Dict[str, str]]:
    categorical_instruction = (
        "\nPlease reason step by step, and put your final answer option within \\boxed{}."
        " Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer."
    )
    if categorical:
        return [{"role": "user", "content": question + categorical_instruction}]
    return [
        {"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."},
        {"role": "user", "content": question},
    ]


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]]) -> Optional[str]:
    if not hasattr(tokenizer, "apply_chat_template"):
        return None
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=True,
        )
    except TypeError:
        # some tokenizers don't support add_special_tokens
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _try_vllm_init(
    model_path: str,
    tokenizer_path: str,
    *,
    trust_remote_code: bool,
    max_model_len: int,
    gpu_memory_utilization: float,
) -> Any:
    import vllm

    kwargs: Dict[str, Any] = {
        "model": model_path,
        "tokenizer": tokenizer_path,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
    }
    # vLLM versions differ; only pass trust_remote_code if supported
    try:
        from inspect import signature

        if "trust_remote_code" in signature(vllm.LLM).parameters:
            kwargs["trust_remote_code"] = trust_remote_code
    except Exception:
        pass

    return vllm.LLM(**kwargs)


def _try_vllm_generate(llm: Any, prompt: str, *, max_tokens: int) -> str:
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=max_tokens, temperature=0.0, n=1)
    out = llm.generate([prompt], sampling_params=params, use_tqdm=False)
    if not out or not out[0].outputs:
        return ""
    return out[0].outputs[0].text or ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Local HF model path (directory).")
    parser.add_argument("--tokenizer-path", default=None, help="Tokenizer path; default: model_path.")
    parser.add_argument("--vllm", action="store_true", help="Also try vLLM init + 1 generation (GPU required).")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--gen-max-tokens", type=int, default=32)
    args = parser.parse_args()

    model_path = args.model_path
    tok_path = args.tokenizer_path or model_path

    print("model_path:", model_path)
    print("tokenizer_path:", tok_path)
    print("python:", sys.version.replace("\n", " "))
    print("cwd:", os.getcwd())

    report: Dict[str, Any] = {"loads": {}, "chat": {}, "notes": []}

    for trust in (False, True):
        key = "trust_remote_code=true" if trust else "trust_remote_code=false"
        try:
            cfg = _load_config(model_path, trust_remote_code=trust)
            tok = _load_tokenizer(tok_path, trust_remote_code=trust)
            report["loads"][key] = {
                "ok": True,
                "config_class": cfg.__class__.__name__,
                "model_type": getattr(cfg, "model_type", None),
                "tokenizer_class": tok.__class__.__name__,
                "has_apply_chat_template": bool(hasattr(tok, "apply_chat_template")),
                "has_chat_template_attr": bool(hasattr(tok, "chat_template")),
                "chat_template_present": bool(getattr(tok, "chat_template", None)),
                "eos_token_id": getattr(tok, "eos_token_id", None),
                "pad_token_id": getattr(tok, "pad_token_id", None),
            }
        except Exception as exc:
            report["loads"][key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # Prefer tokenizer loaded with trust_remote_code=True if needed.
    tok = None
    tok_loaded_key = None
    if report["loads"].get("trust_remote_code=false", {}).get("ok"):
        tok_loaded_key = "trust_remote_code=false"
        tok = _load_tokenizer(tok_path, trust_remote_code=False)
    elif report["loads"].get("trust_remote_code=true", {}).get("ok"):
        tok_loaded_key = "trust_remote_code=true"
        tok = _load_tokenizer(tok_path, trust_remote_code=True)
        report["notes"].append("Tokenizer requires trust_remote_code=True; update server/trainer loading accordingly.")
    else:
        _eprint("Tokenizer failed to load with both trust_remote_code=false/true.")
        print(_render_json(report))
        return 2

    report["chat"]["tokenizer_load_mode"] = tok_loaded_key
    for categorical in (False, True):
        q = "What is 5.6 / 2.0?"
        if categorical:
            q = "What is 5.6 / 2.0?\nA. 1.4\nB. 2.8\nC. 3.6\nD. 7.6"
        msgs = _make_messages(q, categorical=categorical)
        rendered = _apply_chat_template(tok, msgs)
        report["chat"]["categorical" if categorical else "non_categorical"] = {
            "messages": msgs,
            "rendered_via_chat_template": rendered,
        }

    if args.vllm:
        try:
            llm = _try_vllm_init(
                model_path=model_path,
                tokenizer_path=tok_path,
                trust_remote_code=bool(report["loads"].get("trust_remote_code=true", {}).get("ok")),
                max_model_len=int(args.max_model_len),
                gpu_memory_utilization=float(args.gpu_memory_utilization),
            )
            report["vllm"] = {"init_ok": True}
            prompt = report["chat"]["non_categorical"]["rendered_via_chat_template"]
            if not isinstance(prompt, str) or not prompt.strip():
                # fallback to a plain prompt if no template
                prompt = "system: Please reason step by step, and put your final answer within \\boxed{}.\nuser: What is 5.6 / 2.0?"
            out = _try_vllm_generate(llm, prompt, max_tokens=int(args.gen_max_tokens))
            report["vllm"]["sample_prompt"] = prompt[:400]
            report["vllm"]["sample_output"] = out[:400]
        except Exception as exc:
            report["vllm"] = {"init_ok": False, "error": f"{type(exc).__name__}: {exc}"}

    print(_render_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

