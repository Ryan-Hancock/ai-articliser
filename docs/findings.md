# Findings

Measurements and constraints discovered while building this, kept in the same spirit as
semantic-highlighting-slm's own findings file: the numbers that were actually observed,
including the unflattering ones.

## Finding 1: the Python version is decided by spacy, not by anything else

The system default here is Python 3.14.4, and the obvious move is to build on it.

spacy publishes no `cp314` wheels — 3.13 is its ceiling — and `highlighting/chunking.py`
needs spacy for sentence segmentation. `uv lock` against `requires-python = ">=3.14"`
fails outright:

```
Because only the following versions of spacy are available: spacy<=3.8.14, spacy==3.8.15
and spacy>=3.8.14 has no wheels with a matching Python version tag (e.g., `cp314`)
```

Worth noting that `uv pip install --dry-run spacy` on a 3.14 venv *appears* to succeed,
which is how this nearly got missed. Only the strict `uv lock` resolution surfaces it.
The project pins `>=3.12,<3.14`.

## Finding 2: airllm and the highlighting project cannot share an environment

airllm 3.1.0 requires `transformers<5.13`. semantic-highlighting-slm requires
`transformers>=5.14.1`. These are mutually exclusive, and the failure mode is quiet
rather than loud: adding the sibling project as a uv path dependency resolves happily by
backtracking airllm to **2.11.0**, an older release without 3.1.0's generic streaming
path. Nothing errors; you just get a different, older library than the one you asked
for.

Three options were considered — accept 2.11.0, split into two virtualenvs, or vendor the
CRF modules. Vendoring won: seven files, ~850 lines, no logic changes, only flattened
import paths. Verified to lock cleanly at 105 packages with airllm 3.1.0 and transformers
5.12.1. ModernBERT has been supported in transformers since 4.48, so the older pin costs
the classifier nothing.

The trade is real, though: upstream fixes to the CRF code now have to be re-copied rather
than picked up by a version bump. Each vendored file carries a provenance header naming
the source commit (`5d4235c`) so that stays possible.

## Finding 3: the GPU this guest can see is not the GPU it has

Carried over from semantic-highlighting-slm's `docs/findings.md:123` and designed around
here rather than rediscovered. Under WSL2 the physical card is shared with the Windows
host. Ollama running on the host holds weights that appear in **no** Linux-side
`nvidia-smi` process list — only in the aggregate memory-used figure. The previous
project lost a training run to a CUDA OOM from exactly this.

Two consequences for `worker/gpu.py`:

- Free VRAM is read via `torch.cuda.mem_get_info()`, which reports the device total.
  Summing per-process usage would report memory that isn't actually available.
- Before any GPU stage, the worker asks Ollama on the host gateway to unload
  (`POST /api/generate {"keep_alive": 0}`). Waiting for Ollama's idle timeout does not
  work, because the timeout is longer than the gap between pipeline stages.

Confirmed working: the gateway is detected as `172.29.160.1:11434`, and with Ollama idle
the preflight reads 11036MB free of 12282MB total.

## Finding 4: two section-detection bugs that produced plausible-looking wrong output

Both found by the test suite, and both would have been easy to miss in review because
neither crashes.

**Plural headings never matched.** The canonical-section matcher used `^(?:result)\b`.
`\b` requires a non-word character, and the character after "result" in "Results" is
"s" — so `Results` and `References` never matched. They weren't dropped; they were folded
into whichever section came before. In the test fixture that put Results inside Method
and References inside Conclusion, so the evidence bundle silently shipped the
bibliography to the model and mislabelled the results. Fixed by matching `[a-z]*` after
the stem instead of a word boundary.

**The evidence bundle overran its budget.** Sections were truncated against a running
character budget, then joined with `"\n\n"` — two characters per section that the budget
never accounted for. A 120-character budget produced 122 characters. Trivial in size,
but the budget exists to bound prompt cost on a path where tokens are expensive, so an
unbounded overrun is the wrong direction to be wrong in.

## Finding 5: the CRF's problem is span boundaries, not labels

The fine-tuned ModernBERT+CRF tagger runs fine — 0.59s over a 256-word article, ~15s cold
load — but its output on generated article prose is largely unusable. On a real article
body it produced four spans, three of which were fragments:

```
Result      'almost exactly, which points'
Result      'storage rather than'
Result      'bottleneck.  The team reports a sustained 0.8 tokens per second...'
Limitation  'For anything interactive the latency is disqualifying, ...'
```

Only the last is a span a reader would want highlighted. Earlier samples were the same
shape: `'383 held'` tagged Contribution, `'should extend'` tagged Result.

This is domain shift, and specifically it shows up in *boundaries* rather than in labels.
The model was trained on paper abstracts — dense, formulaic, claim-heavy, full of "we
propose" and "we evaluate". Generated articles are deliberately none of those things:
they explain in plain language and avoid exactly the phrasing the tagger keys on. So the
CRF finds the right general region and then cuts it in the wrong place.

Worth being clear that this is not a criticism of the model. It does well on the task it
was trained for; this pipeline is feeding it a different one.

## Finding 6: a zero-shot NLI classifier replaces it, at ~30x the cost

**The search first.** Purpose-built rhetorical-role models on the Hub are effectively
abandoned: the most-downloaded hit is 69 downloads, and nearly all of them are legal-domain
(rhetorical role labelling is mainly an Indian-court-judgment task). PubMed-RCT sentence
classifiers (`gubartz/*`) are the right task family, but they are trained on medical
abstracts — the same domain shift, arguably worse, and with a fixed five-label schema that
doesn't match `schema.py`. Nothing off-the-shelf is both in-domain and better.

**What was chosen instead.** `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` (435M params,
MIT, 122k downloads) run at sentence level. Two properties matter:

- Its labels are natural-language hypotheses, so there is no training domain to shift away
  from. The label set stays defined by `schema.py`.
- It classifies whole sentences, so a boundary can only ever land at a sentence edge. The
  entire failure mode in Finding 5 becomes structurally impossible.

`large` is clearly better than `base` and worth the extra 0.5GB. On the sample where base
returned `(none, best Safety 0.44)` for a limitation sentence, large returned
`Limitation 0.53`; on a numeric-evidence sentence base said `Result 0.98` where large said
`Evidence 0.99`.

**Two failure modes found and fixed.**

*Everything gets a label.* Given only the seven real labels, the model tags 100% of
sentences, because it will always find something — it labelled "Researchers have studied
bearing faults for decades" as Limitation at **0.99** confidence. A high-confidence false
positive is worse than a low-confidence one, so thresholding alone cannot fix it. The fix
is a competing eighth hypothesis describing background/filler, which gives such sentences
somewhere to go. That drops coverage from 100% to ~83%.

*One label eats the budget.* Ranking the remainder by raw confidence and filling a coverage
budget produces articles highlighted entirely as Evidence — numeric sentences score ~0.99
and crowd everything else out. The budget is now spent round-robin across labels, best-first
within each. On the test article that moved the result from 2 spans/1 label to 4 spans/3
labels at the same 20% coverage.

**Where it lands.**

| | CRF | zero-shot |
|---|---|---|
| Tag time, 256-word article | 0.59s | 4.9s |
| Cold load | ~15s | ~21s |
| Spans returned | 4 | 4 |
| Of which usable | 1 | 4 |
| Distinct labels | 2 | 3 |
| Coverage | 20% | 20% |

Roughly 30x slower per article, and irrelevant next to a layer-streamed 32B generation
measured in minutes. VRAM is ~0.9GB in fp16.

**It is not perfect.** On the same article it labelled "The catch is that this is strictly
a batch technique" as Method, where Limitation is plainly right. Sentence-level tagging also
cannot highlight a clause inside a long sentence, which the CRF could in principle do well.
Neither has been measured against gold labels on article prose — there is no such gold set,
and the comparison above is qualitative on a handful of articles, not an evaluation.

The CRF is kept and selectable with `ARTICLISER_TAGGER=crf`. It becomes the better choice
the moment someone fine-tunes the schema on article prose rather than abstracts, at which
point it is 30x faster for free.

## Finding 7: AirLLM works, and costs 20s/token — the GPU is 97% idle by design

The setup is correct. `device` resolves to `cuda:0`, dtype is bfloat16, the 4-bit shards
(18.4GB, split once from a 62GB checkpoint) are found and reused, and generation produces
coherent output. Compute does happen on the GPU. Measured: **20.4s/token, ~0.05 tok/s.**

**Where the time goes** (`profiling_mode=True`, 3 tokens):

| | per token | share |
|---|---|---|
| 4-bit decompression | 7.4s | 36% |
| GPU transfer + compute | ~8.9s | 44% |
| disk read | 4.0s | 20% |

An earlier reading of this called it disk-bound, on the grounds that 18.4GB of shards
exceed 14GB of RAM. That was wrong. Disk is the *smallest* of the three costs, and the
12.1s of `load_safe_tensor` across 3 tokens works out to ~4.6GB/s — well above the drive's
1.5GB/s, which means the page cache is largely doing its job. The dominant costs are
bitsandbytes decompression and the per-layer GPU round trip.

**The GPU sits idle.** Peak VRAM use is roughly one layer, ~300MB out of ~11GB free.
`airllm_base.py`'s `_post_hook` calls `module.to('meta')` immediately after each layer
runs, so all 64 layers are loaded, decompressed, uploaded, executed and evicted *for every
token*. Only the embedding is kept resident (`_load_resident_modules`). There is no option
to spend spare VRAM on keeping layers around — AirLLM exists to run 70B models on 4GB
cards, and a 12GB card is simply more headroom than its design has any use for.

**The tuning levers are all worse.** `compression=None` re-enables prefetching (any
compression disables it, `airllm_base.py:172`), but takes the shards from 18.4GB to ~62GB;
at 1.5GB/s that is ~41s/token of unavoidable disk, and prefetching cannot hide 41s behind
9s of compute. `8bit` also disables prefetching and reads more than 4-bit. So `4bit` is
already the best available configuration, and `max_seq_len` is stored but never used for
truncation, so its 512 default is harmless.

**The shards are the correct size, and AirLLM is meeting its own claim.** Checked because
"18.4GB" sounds like something went wrong with the split. It didn't:

| | |
|---|---|
| 64 layers x 274.3 MB | 17.6 GB |
| embed + lm_head, also 4-bit, 438 MB each | 0.9 GB |
| **total** | **18.4 GB** vs 19.8 GB predicted from `config.json` |

Everything is quantised, embeddings included; 274.3 MB/layer against a predicted 261 MB is
quantisation block metadata. Nothing is stored uncompressed and there is nothing to reclaim.

AirLLM's headline claim is "70B on a single 4GB **GPU**" — that is a statement about VRAM,
and it is being honoured here: a 32B model whose weights need 18.4GB is running on a card
with ~11GB free, using ~300MB. Its speed claims in the changelog are all *relative* --
compression is advertised as "3x run time speed up" and prefetching as a "10% speed
improvement" -- never absolute. So `compression="4bit"` with prefetching disabled is not a
compromise, it is the faster of the two configurations by the library's own numbers, and
20.4s/token is AirLLM working correctly rather than misconfigured.

**The floor is physics, not implementation.** Every layer must cross PCIe once per token.
Measured pinned host-to-device bandwidth on this machine is 8.4 GB/s:

| model | bytes/token | PCIe floor | floor per 1200-token article |
|---|---|---|---|
| Qwen2.5-32B 4-bit | 18.4 GB | 2.19 s/token | 44 min |
| 14B 4-bit | 8.2 GB | 0.97 s/token | 20 min |
| 7B 4-bit | 4.1 GB | 0.49 s/token | 10 min |

The measured 20.4s/token is 9x that floor, so AirLLM leaves real headroom on the table
(decompression is not overlapped with transfer). But even a perfect layer-streaming
implementation could not get a 32B article below ~44 minutes on this hardware, and the
models whose floors look attractive -- 14B at 8.2GB, 7B at 4.1GB -- both fit in this card's
11GB of free VRAM outright, where they would run at ~0.02s/token without streaming at all.

There is therefore no model size at which AirLLM is the right choice on a 12GB card for
long-form generation: below ~14B the model fits and streaming is pure overhead, and at 32B
streaming works but costs hours per article.

**The structural problem.** AirLLM's premise is that the model does not fit in VRAM. At
4-bit, 32B is 18.4GB against ~11GB free, so it genuinely qualifies. But the technique costs
a full load-decompress-upload sweep of every layer per token, and long-form generation wants
1000+ tokens. AirLLM is a good fit for short outputs — classification, extraction, a
few-sentence answer — where that sweep is paid a handful of times. Writing a 900-word
article pays it 1200 times.

Below roughly 14B at 4-bit (~8-9GB) a model fits in this card's VRAM outright, at which
point AirLLM is strictly worse than loading it normally.

**Measured comparison**, same card, the Q4 14B already installed on the Windows host:

| | tok/s | s/token | ~900-word article |
|---|---|---|---|
| AirLLM, Qwen2.5-32B, 4-bit | 0.049 | 20.4 | **6.8 hours** |
| Ollama, Paper-Summarizer-Qwen3-14B, Q4_K_M | 5.9 | 0.17 | **3.4 minutes** |

About 120x. Worth noting the Ollama figure is itself well below what this card should manage
for a 9GB model — with ~2.3GB already used by the desktop it is likely spilling partly to
CPU — so that column has headroom the AirLLM column does not.

## Finding 8: generation moved to Ollama on the host, ~470x faster and out of process

The requirement that settled this was "generate articles without jamming up the machine",
which is a question about *what the generation does to everything else*, not only about
throughput. Ollama wins on both.

**Model choice mattered more than model size.** Benchmarked on the host, same card:

| model | tok/s | VRAM / total footprint | fully resident? |
|---|---|---|---|
| `qwen3.5:2b` | 123.5 | 3.5 / 3.5 GB | yes |
| **`qwen3.5:latest`** (9.7B Q4) | **64.4** | **7.6 / 7.6 GB** | **yes** |
| `ministral-3` (8.9B Q4) | 11.2 | 9.6 / 14.8 GB | no, spills |
| `Paper-Summarizer-Qwen3-14B` (Q4_K_M) | 5.4 | 10.5 / 20.3 GB | no, spills badly |

The task-specific 14B summariser is the *worst* option here, and for the reason that
matters: at 20.3GB against a 12GB card it spills to system memory, which makes it both 12x
slower and disruptive to everything else using the GPU. `qwen3.5:latest` keeps all 7.6GB
resident and leaves ~4GB of headroom.

**Measured end to end**, real PDF through the full pipeline:

| stage | time |
|---|---|
| generation (939 tokens @ 44.6 tok/s) | 21.0s |
| embedding + persistence | ~3s |
| zero-shot highlighting (674 words) | 13.2s |
| illustration | 10.5s |
| **whole article** | **~52s** |

Against AirLLM's 6.8 hours for the same article. Roughly 470x.

Three properties earn it the default beyond speed. The weights live in Ollama's process, so
the worker never holds them and a crashed worker cannot strand VRAM. `unload()` posts
`keep_alive: 0`, returning the card immediately rather than at the idle timeout, which is
what lets the illustration stage have the GPU seconds later. And constrained decoding
against the `ArticleDraft` JSON schema means the response is valid JSON by construction --
`parse_draft`'s repair path still runs, shared with the AirLLM backend, but has nothing to
fix.

Two defects surfaced and were fixed while wiring it up. Pydantic marks a field optional the
moment it has a default, so the decoder was free to skip `tags` and did -- `_draft_schema()`
now marks all six required with `minItems`/`maxItems` on tags. And the model emitted `#`
headings inside the body despite the prompt asking for `##`, duplicating the page's only
`<h1>`; a validator now demotes them, leaving fenced code alone.

## Finding 9: FLUX cannot run on this machine; SDXL-Turbo can

FLUX.1-dev was the plan's choice and it does not work here. A single 1024x768 image did not
finish in **ten minutes**, and swap was in active use throughout.

The arithmetic is unambiguous. FLUX.1-dev is an 11.9B transformer plus a 4.7B T5 text
encoder: ~33GB of weights in bf16, and the on-disk cache is 38GB rather than the 6.8GB an
early partial download suggested. `enable_sequential_cpu_offload` keeps those weights in
*system* RAM and pages submodules to the GPU. This machine has 13GB usable. The offload
therefore pages to disk, and that is the whole story. FLUX.1-schnell would not help: same
transformer, same footprint, fewer steps against an unusable per-step cost.

SDXL-Turbo replaces it -- ~6.6GB of weights in fp16, comfortably inside both budgets:

| | peak VRAM | reserved | RAM | per image |
|---|---|---|---|---|
| all components resident on GPU | 9.55 GB | 12.28 GB | 2.5 GB | 5.2s |
| **`enable_model_cpu_offload()`** | **5.03 GB** | **5.59 GB** | **9.2 GB** | **3.6s** |

Component offload is both lighter and *faster*, which is not the obvious result. Keeping
everything resident reserved 12.28GB on a 12GB card, and the allocator spent the difference
on fragmentation pressure. Offloading leaves ~7GB of the card free for whatever else the
machine is doing, which is the actual requirement.

Worth stating plainly that this is the same API call that made FLUX unusable. The difference
is arithmetic, not technique: SDXL's ~7GB fits in system RAM with room to spare; FLUX's
~33GB does not. Peak VRAM is also resolution-independent here (9.55GB at 1024x768, 768x512
and 640x448 alike) -- the spike is component weights, not image size, so an earlier attempt
to buy headroom by shrinking the output was measuring the caching allocator rather than
real usage.

Quality is a real step down from FLUX. Output at 4 steps reads as texture more than as a
composition, and tends to fill the frame despite the style prompt asking for negative space.
It is good enough for a hero image and it does not disturb the machine, which was the trade
being made.

## Finding 10: discovery is only as good as what the corpus cites

Corpus-driven discovery replaced the fixed arXiv query. Three findings from building it,
all measured on the eight-paper corpus.

**References are precise but thin.** 528 reference entries across the corpus; 23% carry a
DOI, 1% an arXiv id. Identifier matching therefore cannot carry the strategy -- titles
have to, and titles have to be extracted from three mutually incompatible citation styles.
Co-citation exists but is sparse (five cross-paper matches in 493 entries) because the
papers span offshore wind, DRAM acceleration and 6G teleoperation alike, so citation count
*ranks* candidates rather than gating them. Most cited works turn out not to be on arXiv at
all -- the energy and wind papers cite journals almost exclusively -- so this strategy
yields two or three candidates where keywords yields a dozen. That is not a defect: the two
it yields are the two the corpus's own authors judged most relevant.

**PDF extraction corrupts the titles you then have to match.** `normalise()` rejoins
hyphen-broken words, which is right for "band-\nwidth" and wrong for "multiple-\nantenna":
the compound becomes "multipleantenna", which shares no *word* with the real title. Word
overlap then ranked "Quasi-Static SIMO Fading Channels" (0.75) above the actually-cited
"Quasi-Static Multiple-Antenna Fading Channels" (0.67) and would have queued the wrong
paper -- silently, since both look plausible. Comparing letters alone with punctuation
removed is immune to a merged hyphen; `title_similarity` takes the better of the two
measures, which puts the correct paper at 1.00.

**Don't ask a model for query syntax.** The prompted strategy first asked for arXiv field
queries directly and got back `(abs:`, `(all:"robot control" OR all:`, `abs:matrix math OR
abs:` -- truncated, unbalanced, and parenthesised in ways arXiv answers with a 400. Three
of five queries were malformed on the first run. Sanitising was a losing game: each fix
revealed another malformation, and a query that survives sanitising but means the wrong
thing fails *silently* by returning nothing. Asking instead for two or three plain search
terms and composing the syntax in code removed the entire class of bug.

**One operational bug worth recording.** Discovery issues tens of separate arXiv lookups in
a loop -- a title search per citation, plus a query per keyword and per suggestion. The
polite delay was applied only between pages *within* one paginated fetch, so those arrived
back to back and earned an HTTP 429. The delay is now a global gate in `_fetch_page`, which
no caller can forget.

## Finding 11: a book's own outline beats every heuristic

Splitting a 644-page textbook by chapter looked like it would need the heading detection
from Finding 4, generalised. It did not: the PDF carried a 231-entry, three-level embedded
outline with exact titles and page boundaries. Publishers author these; nothing inferred
from font sizes or line spacing will match one.

**Granularity has to be per chapter.** Measured on *Modern Robotics*:

| level | sections | median pages | fits the 12k evidence budget? |
|---|---|---|---|
| L1 chapters | 15 | 46 | no, overflows several times over |
| L2 subchapters | 70 | 4 | yes |
| L3 | 96 | 2, some 0 | too granular, and some are empty |

So L2 is right — but only where it exists. Choosing globally would either lose the
chapters that have no subchapters or shred the ones that do, so each chapter uses its own
children when at least half of them span two pages or more, and stands alone otherwise. On
this book that gave 57 parts: 56 subchapters and one whole chapter.

**Running heads are the most frequent text in a section.** Textbooks repeat the chapter and
section title on every page, and extraction interleaves them with the prose — "3.2.
Rotations and Angular Velocities" appeared four times in an eight-page span, more often
than any real phrase. Left in, the model reads them as emphasis. Any line short enough to
be furniture and present on at least half a section's pages is furniture.

**The paper prompt is actively wrong for a book.** It asks for "what the researchers did,
why it matters, what the limits are". A textbook section is teaching an idea, not reporting
a result, and prompting for the latter makes the model invent findings the source never
claimed. `build_book_prompt` asks for the intuition first, forbids reproducing equations
and figure numbers, and says explicitly that this is one part of a series so the model
does not write an introduction to the whole book each time.

**Two smaller things.** PDF metadata titles are very often empty -- this book's was, and the
filename gave a series called "MR v2" -- so the title page's capitalised opening lines are
the better source. And a series has to record how many parts were *planned*, not how many
are published: generation is resumable across runs, and "part 3 of 3" with 54 still to come
is worse than no count at all.

## Both dropped approaches were removed, not left selectable

AirLLM and FLUX are gone from the codebase and their weights (18GB of layer shards, a 62GB
checkpoint, a 38GB diffusion cache — 116GB total) deleted. They were briefly kept as
config-selectable fallbacks, which was the wrong call: neither can work on this hardware,
so the flags' only reachable behaviour was to re-download 100GB and then fail slowly.
Findings 7 and 9 are the record; the code is not.

Removing airllm also lifted the `transformers<5.13` ceiling that forced Finding 2's
vendoring decision. The CRF modules stay vendored anyway — a path dependency on
semantic-highlighting-slm would now resolve, but it pulls gradio, datasets and
scikit-learn in to supply seven files. Their provenance headers say so.

## Still unmeasured

The two expensive stages have not been run end-to-end, because both require large
one-time downloads:

- **A full article generated end to end by AirLLM.** The per-token cost is now measured
  (Finding 7) but no complete article has been generated, because at 20.4s/token the
  configured 2000-token budget is a 12.8-hour run.
- **FLUX illustration.** Weights are cached (6.8GB) but the pipeline has not been
  executed. Expect 1–2 min/image with sequential CPU offload. The open risk is system
  RAM, not VRAM: 15GB total is tight for offload, and if it thrashes the answer is
  FLUX.1-schnell rather than more aggressive offloading.

Everything else in the pipeline — PDF extraction, section detection, evidence bundling,
JSON parsing and repair, embeddings, CRF tagging, persistence, GPU staging, rendering —
has been run against a real PDF (`vibration-ml/Terra/supplimentary_files.pdf`) and
verified end to end in 14.2s.
