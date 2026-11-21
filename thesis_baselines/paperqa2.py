"""PaperQA2 (FutureHouse `paper-qa`) as a baseline, run as its authors ship it: their prompts, their chunking,
their evidence step. Only the model, the papers and a few size settings are ours.

Two systems:
  paperqa2-rag     Docs.aquery: the fixed PaperQA2 pipeline (retrieve chunks -> one summary per chunk
                   (the "evidence" step, which also scores relevance) -> answer from the best summaries)
  paperqa2-agent   agent_query: the PaperQA2 agent (search the paper index, gather evidence, answer), the
                   setting the PaperQA2 paper reports as its best

What we change, and why:
  - papers are copied to <out>/stage/<paper id>.txt so PaperQA2 reads them as plain text (it would read .md as
    code and chunk by lines); docname = citation = paper id, so citations name the paper.
  - use_doc_details off, multimodal off: no Crossref/Semantic Scholar lookups and no image model; the papers
    are made up, outside metadata would be wrong or missing.
  - every model call (answer, summary, agent) goes to the same gateway alias as the evqa run (gateway.py).
  - embedding: "sparse" by default (PaperQA2's built-in keyword embedding; no embedding model needed, like evqa's
    exact-word search). Set `embedding` to e.g. "hybrid-openai/qwen-embed" with `embedding_config` for a dense one.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from thesis_baselines import contract, gateway

DEFAULTS: dict[str, Any] = {
    "model": "orch-model",          # gateway alias (the evqa run's llm.model)
    "think": False,
    "max_tokens": 4096,
    "embedding": "sparse",
    "embedding_config": {},
    "evidence_k": 10,               # PaperQA2 default
    "answer_max_sources": 5,        # PaperQA2 default
    "chunk_chars": 5000,            # PaperQA2 default
    "overlap": 250,                 # PaperQA2 default
    "max_concurrent_requests": 4,   # PaperQA2 default
    "agent_timeout": 900.0,         # seconds per question (agent only)
    "agent_max_timesteps": None,    # agent steps cap (None = PaperQA2 default)
    "agent_type": "ToolSelector",   # PaperQA2 default: the model picks tools (needs tool calling on the gateway);
                                    # "fake": PaperQA2's fixed order (search -> gather evidence -> answer), no tool calls
}


def merged(settings: dict[str, Any]) -> dict[str, Any]:
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown paperqa2 settings {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    return {**DEFAULTS, **settings}


def settings_hash(s: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest()[:12]


def stage(papers: Path, out: Path) -> Path:
    """Copy <papers>/<id>/paper.md to <out>/stage/<id>.txt (once; the same text byte for byte)."""
    folder = out / "stage"
    folder.mkdir(parents=True, exist_ok=True)
    for pid in contract.paper_ids(papers):
        dst = folder / f"{pid}.txt"
        src = papers / pid / "paper.md"
        if not dst.is_file() or dst.read_bytes() != src.read_bytes():
            shutil.copyfile(src, dst)
    # the agent's index reads this manifest: name each paper by its id (and skip the citation lookup)
    rows = ["file_location,docname,citation,title,dockey"]
    rows += [f"{pid}.txt,{pid},{pid},{pid},{pid}" for pid in contract.paper_ids(papers)]
    (out / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return folder


def patch_router() -> bool:
    """paper-qa 2026.8.12 builds its ToolSelector agent with LiteLLMModel.get_router(), which fhlmi (every release
    from 0.48 to 1.11) no longer has, so the agent fails at its first step. Add it back: a litellm Router over the
    model's own model_list, which is what the method returned before. True when the patch was applied."""
    from lmi import LiteLLMModel
    if hasattr(LiteLLMModel, "get_router"):
        return False
    import litellm

    def get_router(self):
        return litellm.Router(model_list=self.config["model_list"])
    LiteLLMModel.get_router = get_router
    return True


def make_settings(s: dict[str, Any], staged: Path, out: Path):  # -> paperqa.Settings
    from paperqa import Settings
    from paperqa.settings import AgentSettings, AnswerSettings, IndexSettings, MultimodalOptions, ParsingSettings
    cfg = gateway.router_config(s["model"], think=s["think"], max_tokens=s["max_tokens"])
    agent_kw: dict[str, Any] = {"agent_llm": cfg["name"], "agent_llm_config": cfg, "timeout": s["agent_timeout"],
                                "agent_type": s["agent_type"],
                                "index": IndexSettings(name="evqa-baseline", paper_directory=staged,
                                                       index_directory=out / "index",
                                                       manifest_file=out / "manifest.csv",
                                                       sync_with_paper_directory=True)}
    if s["agent_max_timesteps"] is not None:
        agent_kw["max_timesteps"] = s["agent_max_timesteps"]
    return Settings(
        llm=cfg["name"], llm_config=cfg, summary_llm=cfg["name"], summary_llm_config=cfg,
        embedding=s["embedding"], embedding_config=s["embedding_config"] or None, temperature=0.0,
        answer=AnswerSettings(evidence_k=s["evidence_k"], answer_max_sources=s["answer_max_sources"],
                              max_concurrent_requests=s["max_concurrent_requests"]),
        parsing=ParsingSettings(use_doc_details=False, multimodal=MultimodalOptions.OFF,
                                reader_config={"chunk_chars": s["chunk_chars"], "overlap": s["overlap"]}),
        agent=AgentSettings(**agent_kw),
    )


async def chunk_texts(staged: Path, s: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """{chunk name: (paper id, chunk text)}, cut exactly as PaperQA2 cuts the staged files. The agent clears the
    chunk text from the contexts it returns (PQASession.filter_content_for_user), so the text is found here by the
    chunk name ("P-med-01 chunk 3")."""
    from paperqa import Doc
    from paperqa.readers import read_doc
    out: dict[str, tuple[str, str]] = {}
    for path in sorted(staged.glob("*.txt")):
        pid = path.stem
        doc = Doc(docname=pid, citation=pid, dockey=pid)
        for t in await read_doc(path, doc, chunk_chars=s["chunk_chars"], overlap=s["overlap"]):
            out[t.name] = (pid, t.text)
    return out


def _paper_of(ctx: Any, chunks: dict[str, tuple[str, str]]) -> str:
    """Paper id of a context. rag: docname = paper id. agent: PaperQA2 makes its own docname (authors + year +
    title, e.g. "unknownauthorsUnknownyearpmed01") and drops the title before returning, but its citation still
    holds the title, which the manifest sets to the paper id ("Unknown authors. P-med-01. Unknown journal, ...")."""
    doc = getattr(ctx.text, "doc", None)
    ids = {pid for pid, _ in chunks.values()}
    docname = str(getattr(doc, "docname", "") or "")
    if docname in ids:
        return docname
    citation = str(getattr(doc, "citation", "") or "")
    found = [pid for pid in ids if re.search(rf"(?<![\w-]){re.escape(pid)}(?![\w-])", citation)]
    if len(found) == 1:
        return found[0]
    return docname or ctx.text.name.split(" ")[0]


def _text_of(ctx: Any, chunks: dict[str, tuple[str, str]]) -> str:
    """The chunk text; when the agent cleared it, the same chunk re-cut by chunk_texts ("<id> chunk <n>")."""
    if str(ctx.text.text):
        return str(ctx.text.text)
    n = ctx.text.name.rsplit(" chunk ", 1)
    key = f"{_paper_of(ctx, chunks)} chunk {n[1]}" if len(n) == 2 else ""
    return chunks.get(key, ("", ""))[1]


def from_session(session: Any, qid: str, system: str, question: str, seconds: float, chunks: dict[str, tuple[str, str]]
                 ) -> dict[str, Any]:
    """An Answer record (contract.py) from a PaperQA2 PQASession."""
    from paperqa.prompts import CANNOT_ANSWER_PHRASE
    answer = str(session.answer or "")
    contexts = [{"paper": _paper_of(c, chunks), "text": _text_of(c, chunks), "summary": str(c.context or ""),
                 "score": float(c.score or 0)} for c in session.contexts]
    # cited: contexts whose chunk name ("P-med-01 chunk 3") is in the answer; if none, whose key ("pqac-1a2b3c4d")
    # is in the answer or the raw answer (PaperQA2 swaps keys for names, but keeps them in raw_answer)
    said = f"{answer}\n{session.raw_answer or ''}"
    cited = {_paper_of(c, chunks) for c in session.contexts if c.text.name and c.text.name in answer}
    if not cited:
        cited = {_paper_of(c, chunks) for c in session.contexts if c.id and str(c.id) in said}
    tokens = {"prompt": 0, "completion": 0}
    for counts in (session.token_counts or {}).values():
        tokens["prompt"] += int(counts[0])
        tokens["completion"] += int(counts[1])
    return contract.record(qid=qid, system=system, question_sent=question, answer=answer,
                           cannot_answer=CANNOT_ANSWER_PHRASE.lower() in answer.lower(), citations=sorted(cited),
                           contexts=contexts, tokens=tokens, cost=float(session.cost or 0.0),
                           seconds=round(seconds, 3))


async def _rag_docs(settings: Any, staged: Path):  # -> paperqa.Docs
    from paperqa import Docs
    docs = Docs()
    for path in sorted(staged.glob("*.txt")):
        pid = path.stem
        await docs.aadd(path, citation=pid, docname=pid, dockey=pid, title=pid, settings=settings)
    return docs


async def run(system: str, questions: list[dict[str, Any]], papers: Path, out: Path, settings: dict[str, Any],
              log=print) -> dict[str, int]:
    """Answer every question not done yet (contract.done); a failed question is recorded, the run goes on, and the
    next run tries it again."""
    if system not in ("paperqa2-rag", "paperqa2-agent"):
        raise ValueError(system)
    s = merged(settings)
    staged = stage(papers, out)
    chunks = await chunk_texts(staged, s)
    pqa = make_settings(s, staged, out)
    patched = patch_router() if system == "paperqa2-agent" else False
    contract.write_json(out / "run_info.json", {
        "system": system, "patches": ["LiteLLMModel.get_router"] if patched else [],
        "package_versions": {p: metadata.version(p) for p in ("paper-qa", "fhlmi", "fhaviary", "litellm")},
        "settings": s, "settings_hash": settings_hash(s),
        "litellm_params": gateway.public(gateway.litellm_params(s["model"], s["think"], 0.0, s["max_tokens"]))})
    todo = [q for q in questions if not contract.done(out, q["qid"])]
    done = failed = 0
    docs = await _rag_docs(pqa, staged) if system == "paperqa2-rag" and todo else None
    for q in todo:
        text, t0 = contract.question_text(q), time.monotonic()
        try:
            if docs is not None:
                session = await docs.aquery(text, settings=pqa)
            else:
                from paperqa import agent_query
                session = (await agent_query(text, pqa, agent_type=s["agent_type"])).session
            rec = from_session(session, q["qid"], system, text, time.monotonic() - t0, chunks)
            done += 1
        except Exception as e:  # noqa: BLE001 - one question's failure must not stop the others
            rec = contract.record(qid=q["qid"], system=system, question_sent=text,
                                  seconds=round(time.monotonic() - t0, 3), error=f"{type(e).__name__}: {e}"[:2000])
            failed += 1
        contract.write_json(out / "answers" / f"{q['qid']}.json", rec)
        log(f"{q['qid']}: {'problem ' + rec['error'] if rec['error'] else 'done'} ({rec['seconds']:.0f}s)")
    return {"questions": len(questions), "skipped": len(questions) - len(todo), "answered": done, "failed": failed}


def run_sync(*args: Any, **kw: Any) -> dict[str, int]:
    return asyncio.run(run(*args, **kw))
