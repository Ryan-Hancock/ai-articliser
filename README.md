# Articliser

Turns research PDFs into short, illustrated, rhetorically-highlighted articles and
serves them as a reading site.

A batch pipeline reads a PDF, picks out the sections that carry the paper's claims, asks a
local LLM to write an article about it, tags the result with a rhetorical-role classifier,
generates a hero image, and publishes. A separate web process serves what the pipeline
produced. End to end, about **a minute per article**. See [design.md](design.md) for the
original brief.

## What runs where

```
web (uvicorn)  ──write jobs──►  articliser.db  ◄──read/write──  worker (APScheduler)
   reads articles                                                     │
   accepts PDF/URL submissions                          sequential GPU stages:
                                                        Ollama ► tagger ► SDXL
```

The split is not architectural neatness: image generation and highlighting both need the
GPU for tens of seconds, and no request should wait on that. The web process loads no
models at all — it imports no torch and holds no weights.

Every GPU stage loads, runs and explicitly unloads before the next begins, so the card is
occupied for about a minute per article and free the rest of the time.

## Setup

```sh
make install      # uv sync
make seed         # sample articles, so the site is reviewable immediately
make serve        # http://localhost:8000
```

That much needs no GPU, no model weights and no network.

To generate real articles you need [Ollama](https://ollama.com) running with the default
model pulled — on WSL2 that normally means Ollama on the Windows host, which the code
finds via the default gateway:

```sh
ollama pull qwen3.5:latest
make doctor       # checks Ollama, the GPU and the configured models
make generate SOURCE=path/to/paper.pdf
make worker       # or: run it on a schedule, with arXiv discovery
```

## Commands

Run `make` with no arguments for the full list. The main ones:

| | |
|---|---|
| `make serve` | the reading site (loads no models) |
| `make worker` | drains the queue on a schedule, discovers new arXiv papers |
| `make generate SOURCE=x.pdf` | one PDF, start to finish, errors in your terminal |
| `make ingest` | queue every new PDF in `data/pdfs/` (`RUN=1` to generate now) |
| `make discover` | find papers from what the corpus cites, tags and summarises |
| `make test` | 116 tests, no GPU or network required |

## Configuration

Everything tunable is an environment variable, resolved in
[`src/articliser/config.py`](src/articliser/config.py). The ones worth knowing:

| Variable | Default | |
|---|---|---|
| `ARTICLISER_OLLAMA_MODEL` | `qwen3.5:latest` | must fit in VRAM, or it spills and slows ~12x |
| `ARTICLISER_IMAGE_MODEL` | `stabilityai/sdxl-turbo` | must fit in VRAM alongside the tagger |
| `ARTICLISER_EVIDENCE_CHARS` | `12000` | how much of the paper reaches the prompt |
| `ARTICLISER_MAX_NEW_TOKENS` | `2000` | generation budget |
| `ARTICLISER_MIN_FREE_VRAM_MB` | `9000` | preflight threshold before any GPU stage |
| `ARTICLISER_TAGGER` | `zeroshot` | `crf` switches back to the fine-tuned tagger |
| `ARTICLISER_ARXIV_QUERY` | `cat:cs.LG OR cat:cs.CL OR cat:cs.RO` | discovery scope |

## Environment constraints

These are load-bearing enough that changing them will break something. They are recorded
with their evidence in [docs/findings.md](docs/findings.md).

- **Python 3.12–3.13, not 3.14.** spacy ships no `cp314` wheels; `uv lock` fails on 3.14.
- **The GPU is shared with the Windows host** under WSL2, and host allocations are
  invisible to Linux-side `nvidia-smi` process lists. `worker/gpu.py` preflights on
  aggregate free memory and asks Ollama to unload before allocating.
- **15GB of system RAM is the binding constraint, not the 12GB of VRAM.** It is what rules
  out FLUX, and what makes SDXL's component offload safe. Check it before swapping models in.

## Books and series

A book is too long for the paper pipeline: the evidence budget holds 12,000 characters and
a textbook chapter runs to 46 pages. `make book` splits it on the PDF's own embedded
outline instead — exact titles and page boundaries authored by the publisher, rather than
headings inferred from font sizes.

```sh
make book SOURCE=MR-v2.pdf DRY=1        # show the split, generate nothing
make book SOURCE=MR-v2.pdf LIMIT=3      # first 3 parts
make book SOURCE=MR-v2.pdf              # the whole book
```

Granularity is chosen per chapter, not globally: subchapters where a chapter has usable
ones, the chapter itself where it does not. Front and back matter — forewords, exercises,
indexes — is dropped. Runs are resumable, so a book interrupted at part 40 of 57 is
finished by running the same command again.

Each part becomes an article in a `Series`, with chapter grouping and previous/next
navigation at `/series/{slug}`. Book sections use a different generation prompt from
papers: a textbook section is teaching an idea, not reporting a result, and the paper
prompt's "what the researchers did" framing makes the model invent findings.

Measured on *Modern Robotics* (644 pages, 231 outline entries): 57 parts, roughly 70
seconds each.

## Discovery

`make discover` decides what to read next from the corpus you already have, rather than
from a fixed subject query. Three strategies run together, and results are interleaved so
no one of them fills the batch:

| strategy | reads | blind spot |
|---|---|---|
| `references` | what your PDFs cite, ranked by how many cite it | can only surface prior work |
| `keywords` | the tags your articles were given | never escapes the existing vocabulary |
| `prompted` | summaries, via the model | unverifiable, so ranked lowest |

Every queued paper records which strategy found it and why, visible on the `/jobs` page.
Run one at a time with `make discover STRATEGY=references`, preview with `DRY=1`, generate
immediately with `RUN=1`. With no corpus yet it falls back to a fixed arXiv query.

Reference mining is the highest-precision strategy and the most corpus-dependent: measured
on an eight-paper corpus, only 23% of the 528 reference entries carried a DOI and 1% an
arXiv id, and most cited works were journal papers not on arXiv at all. It gets stronger
as the corpus grows.

## Highlighting

Articles are tagged with rhetorical roles — Contribution, Method, Result, Evidence,
Limitation, FutureWork, Safety — from `highlighting/schema.py`, which is the single source
of truth for the label set, the legend and the highlight colours.

Two taggers implement it. The default is zero-shot
(`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`) at sentence level; `ARTICLISER_TAGGER=crf`
switches to the fine-tuned ModernBERT+CRF. The zero-shot model is ~30x slower and
substantially better on this input, because the CRF was trained on paper abstracts and
these articles are plain prose — it finds the right region and cuts it in the wrong place.
Finding 5 and 6 in [docs/findings.md](docs/findings.md) have the head-to-head.

## What was tried and dropped

Two approaches from the original design were built, measured, and removed. Both are
written up in [docs/findings.md](docs/findings.md) rather than deleted from the record,
because the measurements are the reason the current design looks the way it does.

- **AirLLM** streaming a 32B model's layers from disk: works, and costs 20.4s/token —
  6.8 hours for one article, with the GPU 97% idle by design. Every layer is loaded,
  decompressed, uploaded and evicted *per token*.
- **FLUX.1-dev** for imagery: ~33GB of weights held in system RAM under offload, against
  13GB usable. One image did not finish in ten minutes.

Neither is selectable. Leaving them as config flags would have meant options whose only
effect was to re-download 100GB and then fail slowly.

## Reused work

The arXiv fetcher, the document chunking and the CRF tagger come from
semantic-highlighting-slm. The CRF modules under
`src/articliser/highlighting/` are vendored copies (see the header on each file for why
a path dependency was not possible); model weights still load from the Hub repo
`Rychanfox/semantic-highlighting-modernbert-crf`.
