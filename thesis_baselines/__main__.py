"""Command line (evqa calls this; you can too):

    uv run python -m thesis_baselines run --system paperqa2-rag --questions q.jsonl --papers data/dataset/medical \
        --out runs/x --settings '{"model": "orch-model"}' [--only Q-med-p1,Q-med-p2]

Prints one line per question and, last, one JSON line with the counts. Exit code 1 when any question failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from thesis_baselines import contract


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="thesis_baselines")
    sub = p.add_subparsers(required=True, dest="cmd")
    r = sub.add_parser("run", help="answer questions with one baseline system")
    r.add_argument("--system", required=True, choices=contract.SYSTEMS)
    r.add_argument("--questions", type=Path, required=True)
    r.add_argument("--papers", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--settings", default="{}", help="JSON object of system settings (see each module's DEFAULTS)")
    r.add_argument("--only", default="", help="comma-separated qids")
    a = p.parse_args(argv)
    questions = contract.read_questions(a.questions)
    if a.only:
        want = set(a.only.split(","))
        questions = [q for q in questions if q["qid"] in want]
    settings = json.loads(a.settings)

    def log(line: str) -> None:
        print(line, flush=True)

    if a.system == "long-context":
        from thesis_baselines import long_context
        counts = long_context.run(questions, a.papers, a.out, settings, log=log)
    else:
        from thesis_baselines import paperqa2
        counts = paperqa2.run_sync(a.system, questions, a.papers, a.out, settings, log=log)
    print(json.dumps(counts), flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
