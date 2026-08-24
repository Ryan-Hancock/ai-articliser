# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Everything goes through the Makefile. `make` with no target lists them.

```sh
make install                       # uv sync
make test                          # full suite: no GPU, no weights, no network
make serve                         # reading site on :8000 (PORT=n to change)
make worker                        # long-running: periodic discovery + queue drain
make doctor                        # verify Ollama, GPU and configured models are reachable
make generate SOURCE=path/to.pdf   # one PDF end to end, errors in your terminal
make ingest                        # queue every unpublished PDF in data/pdfs/ (RUN=1 drains now)
make book SOURCE=x.pdf             # split a book by its outline into a series (DRY=1, LIMIT=n)
make discover                      # find papers from what the corpus cites (STRATEGY=, DRY=1, RUN=1)
make seed                          # sample articles, so the site is reviewable with no GPU
make reset-db                      # delete the SQLite database
```

**Always run Python through `env -u PYTHONPATH uv run`** (what `PY` in the Makefile
expands to). The shell on this machine carries a ROS `PYTHONPATH` that crashes pytest's
plugin autoloading. A bare `uv run pytest` will fail for reasons that have nothing to do
with the code.

Single test:

```sh
env -u PYTHONPATH uv run pytest tests/test_pdf.py::test_name -q
```

`make ingest DRY=1` does **not** dry-run — the Makefile target only wires `RUN`, though
`scripts/ingest.py` supports `--dry-run`. Call the script directly for a real dry run.

## Architecture

Two processes over one SQLite file:

```
web (uvicorn)  ──write Job rows──►  data/articliser.db  ◄──read/write──  worker
   reads Articles/Spans                                        sequential GPU stages
   accepts PDF/URL submissions                                 Ollama ► tagger ► SDXL
```

The split is a VRAM constraint, not tidiness. **The web process imports no torch and
loads no weights** — it renders what the worker already wrote and appends `Job` rows for
submissions. Keep it that way; anything that pulls a model import into
[src/articliser/web/app.py](src/articliser/web/app.py) defeats the design.

**The pipeline stage order is a VRAM schedule.** [generate/pipeline.py](src/articliser/generate/pipeline.py)
loads, runs and explicitly unloads each model before touching the next, because none of
the three fit alongside each other on a 12GB card. Generation goes first because its
failure means no article at all; highlighting and illustration follow and are
**best-effort** — they log and return rather than raise, so a failure downgrades the
article instead of losing it. Preserve that asymmetry when editing.

**Every GPU stage goes through `gpu_stage()`** in [worker/gpu.py](src/articliser/worker/gpu.py),
which preflights on `torch.cuda.mem_get_info()` (device aggregate, not per-process) and
asks the host's Ollama to unload first. Generation is the one stage passing
`release_host=False`, since the host's loaded model is what does the work.

**The worker is strictly serial** ([worker/runner.py](src/articliser/worker/runner.py)).
The GPU stages must take the card in turn anyway, and serial execution means a crash
leaves exactly one job `RUNNING`, which `scheduler.py` requeues on startup.

Module map: `ingest/` PDF text + section detection + arXiv + book outline splitting ·
`generate/` prompts, Ollama backend, embeddings, the pipeline · `highlighting/`
rhetorical-role tagging · `images/` SDXL-Turbo · `discovery/` corpus-driven paper finding ·
`db/` SQLModel schema and session · `web/` FastAPI + Jinja · `worker/` queue and GPU
arbitration.

### Data model

Four concepts in [db/models.py](src/articliser/db/models.py): a `Source` (PDF, arXiv
entry, submitted URL) becomes an `Article`, which carries `Span` rows (character ranges
over `body_md`, tagged with a rhetorical role) and an optional `Series` membership.

`Series` is the one departure from one-source-one-article: a book is a single `Source` and
many `Article`s. `total_parts` records sections the outline *yielded*, not parts published
— generation is resumable across runs.

Spans are stored rather than computed at render time specifically so the web process
holds no model. Embeddings are a plain JSON float list — at this corpus size "related
articles" is a numpy dot product over every row, so no vector store.

Migrations are additive-only: `_ADDED_COLUMNS` in [db/session.py](src/articliser/db/session.py)
`ALTER TABLE`s missing columns on startup. There is no version table. A change needing
data rewritten or a column dropped needs a real migration tool.

## Constraints that will bite you

These are measured, not assumed. [docs/findings.md](docs/findings.md) carries the evidence
and the numbers, including the unflattering ones — read the relevant finding before
reversing any of these decisions.

- **Python 3.12–3.13, never 3.14.** spacy ships no `cp314` wheels and
  `highlighting/chunking.py` needs it. `uv pip install --dry-run` appears to succeed on
  3.14; only strict `uv lock` surfaces the failure.
- **The GPU is shared with the Windows host** under WSL2, and host allocations appear in
  *no* Linux-side `nvidia-smi` process list. This is why the preflight reads aggregate
  free memory and why Ollama is asked to unload rather than waited out.
- **15GB of system RAM is the binding constraint, not the 12GB of VRAM.** It is what
  rules out FLUX and what makes SDXL's `enable_model_cpu_offload()` both lighter *and*
  faster than keeping components resident.
- **Bigger models are a false economy here.** Anything that doesn't fit resident spills to
  system memory and lands ~12x slower while disrupting the whole machine.
  `qwen3.5:latest` (7.6GB) beats a task-specific 14B summariser by an order of magnitude.
- **AirLLM and FLUX were removed, not left selectable.** Neither can work on this
  hardware, so a fallback flag's only reachable behaviour was re-downloading ~100GB and
  then failing slowly. Findings 7 and 9 are the record; don't reintroduce them.

## Conventions

Module docstrings here explain *why the design is what it is*, not what the code does —
usually naming the constraint or the measurement behind a choice. Match that when adding
modules; a docstring restating the function signature is worse than none.

Comments mark load-bearing subtleties (why `db/models.py` omits
`from __future__ import annotations`, why `next_run_time=None` is wrong in the scheduler).
They are there because the obvious edit breaks something.

`highlighting/{schema,spans,labels,crf_model,predict}.py` are **vendored** from
semantic-highlighting-slm @ `5d4235c` with only import paths flattened. Each carries a
provenance header. Fix upstream and re-copy rather than editing in place.

Configuration is environment variables only, resolved once in
[config.py](src/articliser/config.py). Add knobs there rather than threading arguments
through call sites.

The tagger is swappable via `ARTICLISER_TAGGER` (`zeroshot` default, `crf` alternative).
The CRF is kept because it is ~30x faster, not because it is better — its spans land on
the wrong boundaries on article prose (Finding 5). It becomes the right choice again only
if the schema is retrained on articles rather than abstracts.
