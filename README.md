# thesis-baselines

Answering baselines for [thesis-research-agent](https://github.com/guptaaditya746/thesis-research-agent) (evqa).
Each one answers the same questions from the same papers with the same model as evqa, and writes JSON that
evqa scores. Used as the git submodule `baselines/` of thesis-research-agent (like `llm-setup/`).

| System | What it is | Module |
|---|---|---|
| `paperqa2-rag` | PaperQA2 fixed pipeline: retrieve chunks, summarise each (evidence), answer | `paperqa2.py` |
| `paperqa2-agent` | PaperQA2 agent: search, gather evidence, answer (`agent_type` ToolSelector or fixed order "fake") | `paperqa2.py` |
| `long-context` | every paper in one prompt, one call per question | `long_context.py` |

Output files: [`thesis_baselines/contract.py`](thesis_baselines/contract.py). Plan and reasons: [docs/plan.md](docs/plan.md).

## Run (normally evqa does this)

From thesis-research-agent, on the GPU job like any evqa run:

```bash
evqa run --config configs/evqa/practice-pqa-rag.yaml --text
```

```bash
evqa compare runs/evqa/practice-qwen-<hash> runs/evqa/practice-pqa-rag-<hash>
```

By hand (in this folder, its own uv environment):

```bash
uv sync --locked
uv run python -m thesis_baselines run --system paperqa2-rag --questions q.jsonl \
    --papers ../data/dataset/medical --out /tmp/pqa --settings '{"model": "orch-model"}'
```

Model: every call goes to the LiteLLM gateway of `llm-setup` (`LITELLM_BASE_URL`, default
`http://127.0.0.1:4000/v1`; key `LITELLM_MASTER_KEY`, never written to output). `think: false` sends
`enable_thinking=false`, as evqa does.

## What is fixed, and why

- `paper-qa==2026.8.12` with `fhlmi==1.0.5` and `fhaviary==0.35.0` (the versions it was released with; `uv.lock`
  pins the rest).
- paper-qa 2026.8.12 calls `LiteLLMModel.get_router()`, which no fhlmi release has, so its ToolSelector agent fails
  at its first step. `paperqa2.patch_router()` adds the method back (a litellm Router over the model's own
  model list); `run_info.json` lists the patch when it was used.
- Papers are staged as `<paper id>.txt` (PaperQA2 reads `.md` as code); an index manifest names each paper by its
  id. No Crossref / Semantic Scholar lookups, no image model.
- Embedding `sparse` (PaperQA2's keyword embedding): no embedding model.
- The agent clears chunk text from its results; `chunk_texts()` cuts the papers again the same way to get it back.
- Known, harmless: in the fixed-order agent ("fake") the last "complete" step fails inside fhlmi/fhaviary
  (`_acompletion() takes 0 positional arguments`); the answer was already made, so it is kept.
- The ToolSelector agent needs tool calling on the gateway (vLLM `--enable-auto-tool-choice` and a tool parser).

## Test

```bash
uv run pytest -q     # offline: a fake OpenAI-style gateway (tests/fake_gateway.py), no model
```
