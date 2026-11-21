"""The files a baseline run writes. evqa (thesis-research-agent, `evqa/baselines/`) reads only these.

Input (written by evqa):
  questions.jsonl   one record per line: {"qid", "question", "intent_card": [{"point", "answer"}]}
                    (evqa corpus.pipeline_record: no type, no gold, nothing from the seed)
  papers/           <paper id>/paper.md, the papers the system may read (data/dataset/<domain>)

Output folder:
  run_info.json     {"system", "package_versions", "settings" (no secrets), "settings_hash"}
  answers/<qid>.json  one Answer record (below); written atomically; a rerun skips a question whose
                    file has error null and tries a failed one again

Answer record:
  qid               the question id
  system            "paperqa2-rag" | "paperqa2-agent" | "long-context"
  question_sent     the exact text the system was asked (question + the intent card answers)
  answer            the final answer text, as the system wrote it
  cannot_answer     true when the system said it cannot answer (PaperQA2: its fixed "I cannot answer" phrase)
  citations         paper ids the answer cites (sorted, no repeats)
  contexts          what the system read: [{"paper", "text", "summary", "score"}]; text = the chunk as stored by
                    the system, summary = the system's own note on it ("" when none)
  tokens            {"prompt": int, "completion": int} summed over every model call of the question
  cost              the system's own cost figure (0.0 for a local model)
  seconds           wall time of the question
  error             null, or "<ExceptionType>: <message>" when the question failed (answer is then "")
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

SYSTEMS = ("paperqa2-rag", "paperqa2-agent", "long-context")
FIELDS = ("qid", "system", "question_sent", "answer", "cannot_answer", "citations", "contexts", "tokens", "cost",
          "seconds", "error")


def question_text(rec: dict[str, Any]) -> str:
    """The question plus the user's intent card answers: the same information the evqa pipeline gets."""
    card = [c for c in rec.get("intent_card") or [] if str(c.get("answer", "")).strip()]
    if not card:
        return str(rec["question"])
    lines = [f"- {c.get('point', '').replace('_', ' ')}: {c['answer']}" for c in card]
    return f"{rec['question']}\n\nWhat the user wants:\n" + "\n".join(lines)


def read_questions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def paper_ids(papers: Path) -> list[str]:
    return sorted(p.parent.name for p in Path(papers).glob("*/paper.md"))


def record(**kw: Any) -> dict[str, Any]:
    """An Answer record with every field (missing ones get their empty value); unknown fields are refused."""
    unknown = set(kw) - set(FIELDS)
    if unknown:
        raise ValueError(f"unknown answer fields {sorted(unknown)}")
    empty: dict[str, Any] = {"answer": "", "cannot_answer": False, "citations": [], "contexts": [],
                             "tokens": {"prompt": 0, "completion": 0}, "cost": 0.0, "seconds": 0.0, "error": None}
    out = {**empty, **kw}
    missing = [f for f in FIELDS if f not in out]
    if missing:
        raise ValueError(f"missing answer fields {missing}")
    if out["system"] not in SYSTEMS:
        raise ValueError(f"unknown system {out['system']!r}")
    return {f: out[f] for f in FIELDS}


def done(out: Path, qid: str) -> bool:
    """A question is done when its answer file exists without an error (a failed one is tried again)."""
    path = Path(out) / "answers" / f"{qid}.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("error") is None
    except json.JSONDecodeError:
        return False


def write_json(path: Path, data: Any) -> None:
    """Atomic write: a crash never leaves half a file that a rerun would take as done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
