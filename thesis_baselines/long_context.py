"""Long-context baseline: every paper in one prompt, one model call per question. No retrieval at all.

The medical corpus is about 290 KB (~75k tokens), so it fits a 128k-token window with room for the answer. If the
gateway refuses the prompt as too long, the question is recorded with that error (nothing is cut silently).
contexts = one entry per paper with text "" (= the whole paper was read).
"""
from __future__ import annotations

import hashlib
import re
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from thesis_baselines import contract, gateway

DEFAULTS: dict[str, Any] = {"model": "orch-model", "think": False, "max_tokens": 4096, "timeout": 900.0}
CANNOT = "I cannot answer"

# Prompt contract (docs/prompt-writing.md in thesis-research-agent): one task, explicit inputs, output, examples.
PROMPT = """Task: answer the question using only the papers below. Report every value that answers it.

Inputs:
- papers: each starts with a line "=== PAPER <id> ===" followed by the paper text.
- question: what the user asks, then what the user wants for each open point (time point, patient group, ...).

Output: plain text. For each value: the value with its unit, what it was measured on (drug, measure), the
settings it holds for (dose, time point, patient group), and the paper id in square brackets, e.g. [P-med-03].
Keep values from different settings or papers on separate lines; do not average or merge them.
If no paper reports what is asked, write exactly: {cannot}.

Examples:
- Question "How much does drugX 10 mg lower LDL?", a paper P-x-01 states -1.2 mmol/L at 12 weeks in adults:
  "drugX 10 mg vs placebo, change in LDL at 12 weeks, adults: -1.2 mmol/L [P-x-01]"
- Two papers report the same measure at different time points: one line per paper and time point.
- The papers report drugX only at 20 mg but the question asks about 10 mg: "{cannot}."

{papers}

Question:
{question}
"""


def merged(settings: dict[str, Any]) -> dict[str, Any]:
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown long-context settings {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    return {**DEFAULTS, **settings}


def corpus(papers: Path) -> tuple[str, list[str]]:
    ids = contract.paper_ids(papers)
    blocks = [f"=== PAPER {pid} ===\n{(papers / pid / 'paper.md').read_text(encoding='utf-8')}" for pid in ids]
    return "\n\n".join(blocks), ids


def cited(answer: str, ids: list[str]) -> list[str]:
    return sorted({pid for pid in ids if re.search(rf"\[[^\]]*\b{re.escape(pid)}\b[^\]]*\]", answer)})


def run(questions: list[dict[str, Any]], papers: Path, out: Path, settings: dict[str, Any], log=print,
        complete=None) -> dict[str, int]:
    """`complete(messages, **params) -> response` defaults to litellm.completion (tests pass a fake)."""
    s = merged(settings)
    if complete is None:
        import litellm
        complete = litellm.completion
    text, ids = corpus(papers)
    params = gateway.litellm_params(s["model"], s["think"], 0.0, s["max_tokens"])
    try:
        litellm_version = metadata.version("litellm")
    except metadata.PackageNotFoundError:
        litellm_version = None
    contract.write_json(out / "run_info.json", {
        "system": "long-context", "package_versions": {"litellm": litellm_version}, "settings": s,
        "settings_hash": hashlib.sha256(repr(sorted(s.items())).encode()).hexdigest()[:12],
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()[:12],
        "litellm_params": gateway.public(params), "papers": ids})
    todo = [q for q in questions if not contract.done(out, q["qid"])]
    done = failed = 0
    for q in todo:
        qtext, t0 = contract.question_text(q), time.monotonic()
        try:
            prompt = PROMPT.format(cannot=CANNOT, papers=text, question=qtext)
            resp = complete(messages=[{"role": "user", "content": prompt}], timeout=s["timeout"], **params)
            answer = str(resp.choices[0].message.content or "")
            usage = getattr(resp, "usage", None)
            rec = contract.record(
                qid=q["qid"], system="long-context", question_sent=qtext, answer=answer,
                cannot_answer=CANNOT.lower() in answer.lower(), citations=cited(answer, ids),
                contexts=[{"paper": pid, "text": "", "summary": "", "score": 0.0} for pid in ids],
                tokens={"prompt": int(getattr(usage, "prompt_tokens", 0) or 0),
                        "completion": int(getattr(usage, "completion_tokens", 0) or 0)},
                seconds=round(time.monotonic() - t0, 3))
            done += 1
        except Exception as e:  # noqa: BLE001 - one question's failure must not stop the others
            rec = contract.record(qid=q["qid"], system="long-context", question_sent=qtext,
                                  seconds=round(time.monotonic() - t0, 3), error=f"{type(e).__name__}: {e}"[:2000])
            failed += 1
        contract.write_json(out / "answers" / f"{q['qid']}.json", rec)
        log(f"{q['qid']}: {'problem ' + rec['error'] if rec['error'] else 'done'} ({rec['seconds']:.0f}s)")
    return {"questions": len(questions), "skipped": len(questions) - len(todo), "answered": done, "failed": failed}
