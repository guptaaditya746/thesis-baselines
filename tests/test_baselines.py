"""Offline tests: no real model. PaperQA2 runs against tests/fake_gateway.py."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_gateway import FakeGateway

from thesis_baselines import contract, gateway, long_context, paperqa2

PAPER_A = """# Drugzan in adults

## Results

At week 24, drugzan 20 mg lowered HbA1c by -0.80 percentage points versus placebo in adults.
At week 12 the difference was -0.55 percentage points.
"""
PAPER_B = """# Otherdrug trial

## Results

Otherdrug 5 mg changed body weight by 1.2 kg versus placebo at week 12.
"""
Q = {"qid": "Q-1", "question": "How much does drugzan 20 mg lower HbA1c?",
     "intent_card": [{"point": "time_point", "answer": "Show every time point."}]}


@pytest.fixture()
def papers(tmp_path: Path) -> Path:
    root = tmp_path / "papers"
    for pid, text in (("P-x-01", PAPER_A), ("P-x-02", PAPER_B)):
        (root / pid).mkdir(parents=True)
        (root / pid / "paper.md").write_text(text, encoding="utf-8")
    return root


def test_question_text_adds_intent_card() -> None:
    assert contract.question_text(Q) == ("How much does drugzan 20 mg lower HbA1c?\n\nWhat the user wants:\n"
                                         "- time point: Show every time point.")
    assert contract.question_text({"qid": "q", "question": "Q?", "intent_card": []}) == "Q?"


def test_record_refuses_unknown_fields_and_systems() -> None:
    with pytest.raises(ValueError):
        contract.record(qid="q", system="paperqa2-rag", question_sent="x", gold="leak")
    with pytest.raises(ValueError):
        contract.record(qid="q", system="other", question_sent="x")
    rec = contract.record(qid="q", system="long-context", question_sent="x")
    assert list(rec) == list(contract.FIELDS) and rec["error"] is None


def test_done_retries_failed(tmp_path: Path) -> None:
    contract.write_json(tmp_path / "answers" / "a.json", contract.record(qid="a", system="long-context",
                                                                          question_sent="x", error="Boom: no"))
    contract.write_json(tmp_path / "answers" / "b.json", contract.record(qid="b", system="long-context",
                                                                          question_sent="x"))
    assert not contract.done(tmp_path, "a") and contract.done(tmp_path, "b") and not contract.done(tmp_path, "c")


def test_gateway_params_hide_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_MASTER_KEY", "secret")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://gw:4000/v1/")
    p = gateway.litellm_params("orch-model")
    assert p["model"] == "openai/orch-model" and p["api_base"] == "http://gw:4000/v1"
    assert p["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert "secret" not in json.dumps(gateway.public(gateway.router_config("orch-model")))


def _resp(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def test_long_context_answers_cites_and_resumes(papers: Path, tmp_path: Path) -> None:
    calls: list[dict] = []

    def complete(messages, **kw):
        calls.append({"messages": messages, **kw})
        return _resp("drugzan 20 mg vs placebo, HbA1c, week 24: -0.80 percentage points [P-x-01]")

    out = tmp_path / "out"
    counts = long_context.run([Q], papers, out, {}, log=lambda _: None, complete=complete)
    assert counts == {"questions": 1, "skipped": 0, "answered": 1, "failed": 0}
    prompt = calls[0]["messages"][0]["content"]
    assert "=== PAPER P-x-01 ===" in prompt and "=== PAPER P-x-02 ===" in prompt and "Show every time point" in prompt
    rec = json.loads((out / "answers" / "Q-1.json").read_text())
    assert rec["citations"] == ["P-x-01"] and rec["tokens"] == {"prompt": 100, "completion": 20}
    assert not rec["cannot_answer"] and len(rec["contexts"]) == 2
    assert long_context.run([Q], papers, out, {}, log=lambda _: None, complete=complete)["skipped"] == 1
    assert len(calls) == 1


def test_long_context_failure_is_recorded(papers: Path, tmp_path: Path) -> None:
    def complete(messages, **kw):
        raise RuntimeError("context window exceeded")

    counts = long_context.run([Q], papers, tmp_path / "o", {}, log=lambda _: None, complete=complete)
    assert counts["failed"] == 1
    rec = json.loads((tmp_path / "o" / "answers" / "Q-1.json").read_text())
    assert rec["error"].startswith("RuntimeError") and rec["answer"] == ""


def test_unknown_settings_refused() -> None:
    with pytest.raises(ValueError):
        paperqa2.merged({"evidence": 3})
    with pytest.raises(ValueError):
        long_context.merged({"temperature": 1})


def _pqa_reply(messages, body) -> str:
    text = "\n".join(str(m.get("content")) for m in messages)
    if body.get("tools"):                         # ToolSelector agent: search, gather evidence, answer, complete
        names = [t["function"]["name"] for t in body["tools"]]
        done = sum(1 for m in messages if m.get("role") == "tool")
        plan = [("paper_search", {"query": "drugzan HbA1c", "min_year": None, "max_year": None}),
                ("gather_evidence", {"question": "How much does drugzan 20 mg lower HbA1c?"}),
                ("gen_answer", {}), ("complete", {"has_successful_answer": True})]
        name, args = plan[min(done, len(plan) - 1)]
        if name not in names:
            raise AssertionError(f"tool {name} not offered: {names}")
        return f"TOOL {name} {json.dumps(args)}"
    if "unique keyword searches" in text:          # the agent's search step
        return "drugzan HbA1c, 2000-2026\ndrugzan, 2000-2026\nHbA1c placebo, 2000-2026"
    if "relevance_score" in text:
        if "drugzan" in text.lower() and "-0.80" in text:
            return json.dumps({"summary": "Drugzan 20 mg: HbA1c -0.80 points at week 24.", "relevance_score": 9})
        return json.dumps({"summary": "", "relevance_score": 0})
    line = next((ln for ln in text.splitlines() if ln.startswith("Valid Keys:")), "")
    valid = re.findall(r"pqac-[0-9a-f]{8}", line)
    return f"Drugzan 20 mg lowered HbA1c by -0.80 percentage points at week 24 ({valid[0] if valid else 'none'})."


@pytest.mark.parametrize("system,extra", [("paperqa2-rag", {}), ("paperqa2-agent", {"agent_type": "fake"}),
                                          ("paperqa2-agent", {})])
def test_paperqa2_end_to_end_with_fake_gateway(papers: Path, tmp_path: Path, system: str, extra: dict) -> None:
    out = tmp_path / "out"
    with FakeGateway(_pqa_reply) as gw:
        os.environ["LITELLM_BASE_URL"] = gw.url
        os.environ["LITELLM_MASTER_KEY"] = "k"
        counts = paperqa2.run_sync(system, [Q], papers, out, {"chunk_chars": 3000, **extra}, log=lambda _: None)
        bodies = [r["body"] for r in gw.requests if r["path"].endswith("/chat/completions")]
    assert counts["answered"] == 1, json.loads((out / "answers" / "Q-1.json").read_text())["error"]
    rec = json.loads((out / "answers" / "Q-1.json").read_text())
    assert "-0.80" in rec["answer"] and rec["citations"] == ["P-x-01"] and not rec["cannot_answer"]
    assert {c["paper"] for c in rec["contexts"]} <= {"P-x-01", "P-x-02"}
    for c in rec["contexts"]:     # every chunk is a piece of its paper, byte for byte
        assert c["text"] in (papers / c["paper"] / "paper.md").read_text()
    assert all(b["model"] == "orch-model" for b in bodies)
    assert all(b.get("chat_template_kwargs") == {"enable_thinking": False} for b in bodies)
    info = json.loads((out / "run_info.json").read_text())
    assert info["settings"]["chunk_chars"] == 3000 and "k" not in json.dumps(info["litellm_params"]).split('"')


def test_tool_choice_patch_sends_auto_instead_of_required():
    """vLLM 0.21 answers tool_choice "required" with an empty tool list; the agent must send "auto"."""
    import asyncio

    from aviary.core import Message, Tool, ToolSelector

    def search(query: str) -> str:
        """Search the papers.

        Args:
            query: words to search for.
        """
        return query

    sent = []

    async def acompletion(*_a, **kw):
        sent.append(kw.get("tool_choice"))
        call = {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"query": "x"}'}}
        msg = SimpleNamespace(model_dump=lambda: {"role": "assistant", "content": None, "tool_calls": [call]})
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="tool_calls", message=msg)], usage=None)

    original = ToolSelector.__call__
    try:
        assert paperqa2.patch_tool_choice("required") is False
        assert paperqa2.patch_tool_choice("auto") is True
        tool = Tool.from_function(search)
        selector = ToolSelector(model_name="m", acompletion=acompletion)
        asyncio.run(selector([Message(content="q")], [tool]))
        asyncio.run(selector([Message(content="q")], [tool], tool_choice=tool))
    finally:
        ToolSelector.__call__ = original
    assert sent[0] == "auto"
    assert sent[1] == {"type": "function", "function": {"name": "search"}}  # a named tool is left alone


def test_retrieval_query_patch_searches_with_bare_question(monkeypatch):
    """The sparse search gets the bare question; the evidence and answer steps keep the intent card."""
    import asyncio

    from paperqa import Docs

    seen = []

    async def fake_retrieve(self, query, k, *a, **kw):
        seen.append(query)
        return []

    monkeypatch.setattr(Docs, "retrieve_texts", fake_retrieve)
    monkeypatch.setattr(paperqa2, "RETRIEVAL_QUERY", {})
    assert paperqa2.patch_retrieval_query() is True
    assert paperqa2.patch_retrieval_query() is False  # applied once
    q = {"question": "How much does drugzan lower HbA1c?", "intent_card": [{"point": "time_point", "answer": "all"}]}
    full = contract.question_text(q)
    paperqa2.RETRIEVAL_QUERY[full] = q["question"]
    asyncio.run(Docs().retrieve_texts(full, 10))
    asyncio.run(Docs().retrieve_texts("a query the agent wrote", 10))
    assert seen == [q["question"], "a query the agent wrote"]
