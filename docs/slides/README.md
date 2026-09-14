# Slide decks

## Serving a 1.64B model online: a self-study guide

`inference-lessons-2026-09-13.html` (68 slides) and `inference-lessons-2026-09-13.pdf` (the same slides, one per page).

A self-study guide to model inference for ML engineers who know transformers and PyTorch but have never served a
model. Each concept is introduced from first principles in a **primer** slide, then shown with a real measurement
from [`../WRITEUP.md`](../WRITEUP.md) and [`../local_phase.md`](../local_phase.md). Part 3 of a series, after
[glukicov/ltm](https://github.com/glukicov/ltm) (TabFM vs CatBoost vs an LLM) and
[glukicov/ltm_ft](https://github.com/glukicov/ltm_ft) (fine-tuning TabFM).

1. **The model:** training vs serving, what TabFM is, KV caches in LLMs vs this project's exact context cache, the masking property.
2. **Measure first:** latency vs throughput, percentiles and tails, GPU asynchrony and honest timing, bf16 vs fp32, noise floors, per-stage timing.
3. **Model-level optimisation:** the context cache and its memory cost, table width and request size, ensemble size, member batching, compute/memory/overhead-bound regimes, `torch.compile` and CUDA graphs, shape buckets and warm-up, precision per device.
4. **Serving and load testing:** anatomy of a request, queueing and the knee, dynamic batching, a context-first API and batcher, open vs closed-loop load generation and coordinated omission, the client bottleneck, what an inference server (Triton) does.
5. **Kubernetes and GPUs:** Kubernetes objects for GPU inference, KServe and KEDA, cold-start anatomy, eager vs compiled serving on an L4, a synthesis of how the bottleneck moved, autoscaling vs quota.
6. **Lessons and reference:** eight principles, a capacity model, everything that went wrong, a two-slide glossary and further reading.

**How to study it.** Go in order. Primer slides (kicker "Primer · …") come before the slides that use the concept.
Tinted **Key idea** boxes hold the sentence to remember and red-tinted **In production** boxes say what changes in a
real system. Each section ends with **Check your understanding**: answer the three questions before reading the grey
answers, then run the **Try it** command in the repo. Unfamiliar terms are in the glossary (slides 65–66).

All 15 figures in [`../figures/`](../figures/) appear with an explanation. Two diagrams are Mermaid, pre-rendered and
inlined as SVG; the rest are built-in flow boxes.

**Viewing.** Open the HTML file in a browser from this folder: images load from `../figures/`, so the file does not
work standalone once moved. Use ← → (or click) to move between slides, Esc for the overview grid, and `#N` in the URL to
jump to slide N. The PDF is self-contained (page images at 1920×1080, so text in it isn't selectable).

**Editing.** All slide markup sits between the `SLIDES START` and `SLIDES END` markers. Each `<section class="slide">`
has a 0-indexed comment above it (`<!-- 13: HONEST TIMING -->` is display slide 14, `#14`). Renumber these comments
whenever you add or remove a slide. Snippets are verbatim source trimmed with `…`, and each carries `data-src` and
`data-sha256`. Re-cite any snippet you change with the SlideOps `cite.py`, rather than editing a hash by hand.
Content slides reserve `padding-bottom:88px` so content stays clear of the nav pill; check with a screenshot after any
edit. The PDF does not update itself: re-export it after editing.

**Staying accurate.** This is a snapshot of one project's measurements, so no automated check is wired up. To see
whether the quoted code has drifted:

```bash
python3 ~/.claude/skills/slideops/scripts/check.py docs/slides/ --repo . --suggest
```
