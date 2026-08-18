# modded-nanogpt: Complete Working Knowledge

> A single-source reference for this repository: what it is, every technique in it, and the
> background needed to understand each one. Written for an engineer who wants no black boxes.
>
> Every claim about *this code* carries a `file:line` reference and was verified against source.
> Claims about *external* work are cited to a paper. Where sources disagree or something is
> unverifiable, it says so explicitly; see **Part 12: Known Discrepancies**.

---

## Table of Contents

| Part | Topic |
|---|---|
| **0** | [Executive orientation: read this first](#part-0--executive-orientation) |
| **1** | [What this project is: the speedrun, the rules, the three tracks](#part-1--what-this-project-is) |
| **2** | [Foundations: language modelling, cross-entropy, tokenization, data](#part-2--foundations) |
| **3** | [The hardware: GPUs, CUDA, and H100 internals](#part-3--the-hardware) |
| **4** | [The optimizer story: Muon → Polar Express → NorMuon](#part-4--the-optimizer-story) |
| **5** | [The model architecture: every trick, decoded](#part-5--the-model-architecture) |
| **6** | [GPU kernels: Triton, CUDA, FP8](#part-6--gpu-kernels) |
| **7** | [Distributed training and the PyTorch runtime](#part-7--distributed-training-and-the-pytorch-runtime) |
| **8** | [How a run actually executes, second by second](#part-8--how-a-run-actually-executes) |
| **9** | [Track 3: the optimizer research environment (your working area)](#part-9--track-3-the-optimizer-research-environment) |
| **10** | [Timeline: how the current design was derived](#part-10--timeline) |
| **11** | [Glossary](#part-11--glossary) |
| **12** | [Known discrepancies, traps, and what I could not verify](#part-12--known-discrepancies-and-traps) |

---

# Part 0: Executive Orientation

## What you are looking at

**modded-nanogpt** is a *competitive optimization benchmark* disguised as a training script. It is
Keller Jordan's "NanoGPT speedrun": a public, adversarial contest to train a GPT-2-class language
model to a **fixed quality bar** as fast as possible on **8× NVIDIA H100** GPUs.

- The quality bar: **≤ 3.28 mean cross-entropy loss** on a fixed 10,485,760-token slice of the
  FineWeb validation set.
- The baseline: Andrej Karpathy's `llm.c` reproduction of GPT-2 (124M), which hit that loss in
  **45 minutes** using **10 billion tokens**.
- The current record: **1.23 minutes** (~74 seconds), using **under 400 million tokens**.

That is roughly a **36× wall-clock speedup** and a **~25× sample-efficiency gain**, achieved over
~89 accepted records by dozens of contributors between June 2024 and July 2026.

## Why this matters as a research artifact

Most optimizer/architecture papers compare against a baseline the authors themselves tuned. That
is a structurally corrupt comparison; see Part 9.6. This repo inverts the incentive: the baseline
is a *standing public record* that someone else set, anyone can attack it, and every accepted
record ships its complete source code and multi-seed logs. It is one of the few places in ML where
a claimed improvement has been adversarially validated.

## The directory map

```
modded-nanogpt/
├── train_gpt.py                 # 110 KB: THE track-1 record trainer (model+optimizer+loop, one file)
├── train_gpt_medium.py          #  78 KB: the track-2 (GPT-2 Medium) trainer
├── triton_kernels.py            #  45 KB: 8 Triton kernels + 1 raw CUDA-C kernel
├── dc_triton_kernels.py         #  42 KB: 11 Triton kernels for the DCFormer attention correction
├── run.sh                       # torchrun --standalone --nproc_per_node=8 train_gpt.py
├── Dockerfile                   # CUDA 12.6.2 + Python 3.12.7 + torch nightly, for reproducible timing
├── requirements.txt             # numpy, tqdm, torch==2.10, huggingface-hub, kernels, datasets, tiktoken
├── README.md                    # 52 KB: rules, the full 89-record leaderboard, Muon writeup, FAQ
├── data/
│   ├── fineweb.py               # tokenize FineWeb → .bin shards (the format spec lives here)
│   ├── cached_fineweb10B.py     # download PRE-tokenized shards (what you actually run)
│   ├── cached_fineweb100B.py
│   └── cached_finewebedu10B.py
├── evals/hellaswag.py           # OPTIONAL sanity eval. Not part of scoring.
└── records/
    ├── track_1_short/           # ~98 dated record dirs: the archaeology of every trick
    ├── track_2_medium/          # ~18 dated record dirs
    └── track_3_optimization/    # ← THE OPTIMIZER RESEARCH BENCHMARK (your working area)
        ├── README.md            # 43 KB: rules, statistical protocol, 46-entry leaderboard
        ├── train_gpt_simple.py  # 14 KB: the FROZEN baseline trainer
        ├── make_figures.py      # parses README + logs → the leaderboard plots
        └── results/             # ~48 experiment dirs, each with README + multi-seed logs + script
```

## The single most important orientation fact

Your parent directory is named `open-ended-research-optimization-env`. That is a strong signal that
your working area is **`records/track_3_optimization/`**, not the top-level speedrun.

**These are different games with different objectives:**

| | Track 1 (`train_gpt.py`) | Track 3 (`records/track_3_optimization/`) |
|---|---|---|
| Minimize | **wall-clock seconds** | **optimizer steps** |
| Allowed to change | *anything*: architecture, kernels, precision | **only the optimizer**, its hyperparameters, and weight init |
| Frozen | data pipeline only | data, batch size, **architecture** |
| Hardware | 8×H100, standardized | any {1,2,4,8}×{A100,H100,A40,…} (irrelevant to the metric) |
| Model | 11 layers, 6 heads, `head_dim=128`, `d=768`, exotic | 12 layers, `d=768`, **plain causal attention, 1024 ctx** |
| Slow methods | disqualifying | **explicitly fine** |
| Current best | 1.23 min | 2690 steps (baseline: 3250) |

If you optimize a Triton kernel, you win track 1 and change nothing in track 3. If you invent an
optimizer that halves step count but doubles per-step cost, you win track 3 and lose track 1.
**Know which game you are playing before you touch anything.**

---

# Part 1: What This Project Is

## 1.1 The task, precisely

From `README.md`:

> This repository hosts the *NanoGPT speedrun*, in which we (collaboratively|competitively) search
> for the fastest algorithm to use 8 NVIDIA H100 GPUs to train a language model that attains 3.28
> cross-entropy loss on the FineWeb validation set.

The 3.28 number is not arbitrary. It is where Karpathy's `llm.c` GPT-2 (124M) reproduction landed
(~3.29, rounded down for a clean target). Choosing an *existing, externally-set* quality bar is what
makes the benchmark honest: nobody gets to pick a target that flatters their method.

## 1.2 The rules (track 1), verbatim

> New records must:
> 1. **Not modify the train or validation data pipelines.** (You can change the batch size, sequence
>    length, attention structure etc.; just don't change the underlying streams of tokens.)
> 2. **Attain ≤3.28 mean val loss.** (Due to inter-run variance, submissions must provide enough run
>    logs to attain a statistical significance level of p<0.01 that their mean val loss is ≤3.28. …
>    For submissions which improve speed by optimizing the systems performance, without touching the
>    ML, this requirement is waived.)
> 3. **Not use any extra `torch._inductor.config` or `torch.compile` flags.** (These can save a few
>    seconds, but they can also make compilation take >30min. This rule was introduced after the 21st
>    record.)
> 4. **Run faster than the prior record when baselined on the same hardware.**

Two discretionary rejection grounds are also stated:

> Disproportionately degrades the readability of the codebase. A 200 line kernel to drop 300ms is
> considered worthwhile. 500 lines that convolute the optimizer layout for a 50ms gain will likely
> be rejected.

> The current record is intentionally kept roughly 0.001-0.002 loss below 3.28 to make validation
> simpler. If a PR substantially consumes this buffer, it should do so in a way that outperforms a
> simple step count decrease, when measured at equivalent loss.

`README.md:220` confirms the buffer rule verbatim.

**The rule change (`records/track_1_short/2025-02-01_RuleTweak/`).** After record #21, two things
changed: (a) the old "10 untimed warmup steps grace period" was replaced by an **explicit untimed
kernel-warmup section running on dummy data**, which added ~850 ms of real runtime; and
(b) `torch._inductor.config.coordinate_descent_tuning` was banned; it saves ~3 s of runtime but
costs ~25 min of untimed pre-run compilation. Record #21 was re-timed 2.933 → 2.997 min under the
new rules. This is why you will see `# BANNED` next to that flag at `train_gpt.py:38`.

## 1.3 How the target metric is *actually* defined

The README states the goal not as "mean loss ≤ 3.28" but as:

> obtain a probability model of language which assigns a probability of at least
> `math.exp(-3.28 * 10485760)` to the first 10,485,760 tokens of the FineWeb valset.

These are algebraically identical (Part 2.3), but the second framing is strictly more rigorous, and
the reason is worth internalizing:

**Mean cross-entropy is only well-defined once you fix how the token stream is factorized.** Chop
the val stream into 1024-token chunks with context reset at each boundary, and the first tokens of
each chunk score badly. Chop it into 65536-token chunks and you get a *lower* number for the same
weights. Both are legitimate: each is a different valid probability model over the same stream.
Stating the target as "the probability your model assigns to this specific prefix" makes the metric
invariant to chunking. That is why the repo can allow evaluation at any sequence length: longer
eval context is not cheating, it *is* a different (better) model, scored on the same fixed quantity.

The binding constraints hidden in "valid probability model" are (a) strict causality (`p(x_i)` may
condition only on `x_<i`) and (b) normalization over the vocabulary. This is why the repo ruled
that @samacqua's test-time-training run was "technically a valid probability model" (it *is* causal)
and had to disallow it on a separate procedural ground: untimed backward passes.

In code: `val_tokens: int = 10485760` at `train_gpt.py:1886`, with the comment *"it's important to
keep this fixed for consistent comparisons."*

## 1.4 The three tracks

**`track_1_short`: GPT-2 Small.** Target ≤3.28. Minimize wall-clock on 8×H100. 89 records,
45 min → 1.23 min. This is what `train_gpt.py` implements.

**`track_2_medium`: GPT-2 Medium.** Target ≤2.92, matching `llm.c`'s 350M-parameter baseline
(5.8 hours). Current: 17.35 min (record 18, "bulk transfer short-track features"). Same rules
except `coordinate_descent_tuning` is *allowed* after record 6; the runs are long enough that
25 min of compile amortizes.

**`track_3_optimization`: the optimizer benchmark.** Minimize **step count**, not seconds. See
Part 9. This is a scientifically different instrument and is almost certainly your working area.

## 1.5 Measurement methodology (the part most papers skip)

- **Standardized hardware.** Rule 4 requires beating the prior record *on the same hardware*.
  Records are validated on 8×H100 from PrimeIntellect, who sponsor validation runs.
- **Statistical significance.** Because run-to-run variance (σ ≈ 0.0013 val loss) is comparable to
  the improvements being claimed, ML-affecting submissions must supply enough logs for **p<0.01**
  that mean val loss ≤3.28. Reference p-value code lives in
  `records/track_1_short/2025-01-04_SoftCap/`.
- **The waiver.** Pure systems optimizations that provably don't change the math skip the
  significance requirement. That is correct, because if the computation is bit-equivalent there
  is no distribution to test.
- **Timing definition.** The clock excludes `torch.compile` time, excludes validation, and starts
  only after an explicit warm-up-then-state-reset (Part 8.3). Every record dir ships the full
  source of the trainer *inside the log file*, so any record is reproducible by copy-paste.

This methodology is stricter than most published ML papers'. Say that plainly to anyone who
dismisses this as a leaderboard-chasing exercise.

## 1.6 The stated caveat, in the repo's own words

The README does not oversell. On "won't this just overfit val loss / not scale?":

> hard to refute, since 'at scale' is an infinite category

and it concedes that structure-imposing tricks (logit softcapping is named explicitly) likely will
not scale. The counter-evidence offered is downstream and honest: scaling the 10/18/24 version to
1.5B parameters reproduces GPT-2(1.5B)-level HellaSwag at **$233 vs $576**, 2.5× cheaper than
Karpathy's baseline.

## 1.7 Out-of-rules runs worth knowing about

- **@alexjc's 2.77-min TokenMonster run (01/20/25).** Violates rule 1 (different tokenizer). The
  speedup came from roughly halving the vocab, shrinking the head and embedding matrices, *not*
  from better data efficiency. Cross-tokenizer loss comparisons are meaningless (Part 2.5).
- **@samacqua's test-time-training run (01/23/26).** Per-document Adam "parameter nudging" on each
  document's early ~500 tokens. A valid probability model, but untimed backward passes are not
  allowed.

## 1.8 Running it

```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install -r requirements.txt
python data/cached_fineweb10B.py 9    # first 900M training tokens (~1.8 GB)
./run.sh                              # torchrun --standalone --nproc_per_node=8 train_gpt.py
```

> `torch.compile` will add around 7 minutes of latency the first time.

Fewer GPUs: change `--nproc_per_node` in `run.sh`. `grad_accum_steps = 8 // world_size`
(`train_gpt.py:57`) preserves the global batch, so the math is unchanged; only wall-clock differs.
For precise timing, use the Docker image (pins CUDA/NCCL/cuDNN/Python).

---

# Part 2: Foundations

*What is actually being trained, and what does 3.28 mean.*

## 2.1 Autoregressive language modelling

A language model is a parameterized conditional distribution `p_θ(x_i | x_1..x_{i-1})` over a fixed
vocabulary. Applied repeatedly it defines a distribution over whole sequences via the **chain rule
of probability** (an identity, not an assumption):

```
P(x_1..x_n) = Π_{i=1..n} P(x_i | x_<i)
```

The model never represents the joint directly; it only emits a length-V categorical distribution per
position.

**Teacher forcing** means that during training the conditioning prefix is always the ground-truth
prefix, never the model's own samples. This is what makes training embarrassingly parallel across
positions: a causal-masked transformer computes all `n` conditionals in one forward pass, because
position `i` may only attend to positions `≤ i`. The cost is a train/inference mismatch (*exposure
bias*), irrelevant here since evaluation is also teacher-forced.

## 2.2 Cross-entropy loss: the metric, precisely

```
L = -(1/N) Σ_{i=1..N} log p_θ(x_i | x_<i)
```

This is the **negative average log-likelihood** of the data under the model. Minimizing it is
maximum-likelihood estimation. It is also the cross-entropy `H(q, p_θ)` between the empirical
one-hot distribution and the model; hence the name.

**Units.** `log` is natural log, so `L` is in **nats/token**. Divide by `ln 2 ≈ 0.693147` for bits:

```
3.28 nats ÷ ln 2 = 4.7320 bits/token
```

That is a compression statement: this model encodes the corpus in ~4.73 bits per GPT-2 token.

**Perplexity** `= exp(L)`:

```
exp(3.28) = 26.5758
```

Read it as: at each position the model is as uncertain as if choosing **uniformly among ~26.6
equally likely tokens**.

## 2.3 The equivalence, in one line

With `N = 10,485,760`:

```
P_θ(x_1..x_N) ≥ exp(-3.28·N)
⟺ log P_θ(x_1..x_N) ≥ -3.28·N
⟺ -(1/N) log P_θ(x_1..x_N) ≤ 3.28
⟺ mean cross-entropy ≤ 3.28
```

For scale: `log10` of that threshold is ≈ `-1.4937 × 10^7`. Nobody computes it; it exists to pin the
definition.

## 2.4 Calibration table

| Setting | CE (nats) | Perplexity | bits/token |
|---|---|---|---|
| Uniform over 50257 tokens | 10.8249 | 50257 | 15.62 |
| Unigram/bigram-class baselines | ~6–7 | ~400–1100 | ~8.7–10.1 |
| GPT-2 small (124M), `llm.c` reproduction | ~3.29 | ~26.84 | 4.746 |
| **Track 1 & 3 target** | **3.28** | **26.58** | **4.732** |
| **Track 2 target** (GPT-2 Medium class) | **2.92** | **18.54** | 4.213 |
| Modern well-trained multi-B model on web text | ~2.2–2.6 | ~9–13 | ~3.2–3.8 |

Only the 3.28/3.29/2.92 rows come from this repo; the others are order-of-magnitude calibration.

Note the shape of the curve. Going from random to 3.28 removes 10.87 bits/token. Going from 3.28 to
2.92 removes another 0.52. **Loss differences of 0.01 nats are large at this end.** That is exactly
why the rules demand statistical significance rather than a single lucky run.

## 2.5 Why loss numbers are only comparable within a fixed tokenizer + corpus

Cross-entropy is per *token*. Change the tokenizer and you change what a token is: a vocabulary
with longer average tokens has fewer, harder predictions, so its per-token loss is higher for the
same underlying model quality. This is why rule 1 freezes the data pipeline and why the TokenMonster
run is listed as out-of-rules rather than as a record.

## 2.6 The GPT lineage

- **GPT-1** (2018): 117M decoder-only transformer; generative pretraining + supervised fine-tuning.
  The contribution was the recipe.
- **GPT-2** (2019): same architecture scaled to 1.5B, WebText corpus, pre-LayerNorm blocks,
  byte-level BPE. The claim was zero-shot multitask capability.
- **GPT-3** (2020): 175B params, ~300B tokens, alternating dense/sparse attention. The contribution
  was in-context/few-shot learning as an emergent property of scale.

**GPT-2 small, exactly.** 12 layers, 12 heads, `d_model = 768` (head dim 64), context 1024, vocab
50257, tied embeddings. Parameter arithmetic: embeddings `50257 × 768 = 38,597,376`; learned
positions `1024 × 768 = 786,432`; 12 blocks × 7,087,872; plus a final LayerNorm →
**124,439,808**, the canonical "124M".

**"GPT-2 small class" means it clears the same quality bar at the same parameter budget, not that
it is architecturally identical.** The speedrun model has rotary embeddings instead of learned
positions, ReLU² instead of GELU, QK-norm, sliding-window attention, value embeddings, U-Net skips,
and eleven layers, not twelve.

Why the community still uses it: it is the largest model that trains to a meaningful loss in
minutes-to-hours on one node, has a decade of published reference numbers, and sits above the regime
where results are pure noise.

## 2.7 Tokenization

**BPE has two distinct procedures.**

*Merge training*: start with a base alphabet; count adjacent symbol-pair frequencies over a corpus;
merge the most frequent pair into a new symbol; append that merge to an **ordered** rule list;
repeat `k` times.

*Encoding*: split text into pretoken chunks by a regex, then repeatedly apply learned merges in
**rank order** (lowest rank first) until no rule applies. Deterministic; depends only on the ordered
merge list.

**GPT-2 uses byte-level BPE**: the base alphabet is the 256 possible *bytes*, not Unicode
characters. Any byte string is representable, so **out-of-vocabulary is structurally impossible**;
worst case a string degrades to one token per byte. A fixed regex pretokenizer keeps leading spaces
attached to words and prevents merges across category boundaries.

**Vocab decomposition, verified by running tiktoken:**

```
n_vocab = 50257,  eot_token = 50256,  max_token_value = 50256
mergeable ranks: 50,256 entries → 256 single-byte + 50,000 multi-byte
special tokens: {'<|endoftext|>': 50256}
→ 50000 + 256 + 1 = 50257  ✓
```

`<|endoftext|>` (id 50256) is what `data/fineweb.py:78-82` prepends to every document as a
delimiter, and it is why tokens fit in `uint16`.

**Why pad 50257 → 50304.** `50304 = 393 × 128`, i.e. 47 dead rows. The vocabulary is one dimension
of the embedding and LM-head matmuls. Tensor-core GEMMs want dimensions that are multiples of 8/16,
and cuBLAS tile scheduling wants 64 or 128. 50257 forces a ragged tail tile, misaligned loads, and a
slow fallback path. Padding costs ~0.1% extra FLOPs and buys a materially faster kernel. The padded
logits are never targets, so they simply learn large negative values. Implemented at
`train_gpt.py:1216` as `self.vocab_size = next_multiple_of_n(vocab_size, n=128)`.

**tiktoken vs HuggingFace tokenizers.** tiktoken is OpenAI's minimal Rust-backed BPE, encode-only,
fast. HF `tokenizers` is a general framework (normalizers, trainers, alignment offsets). For a fixed
published vocabulary they produce identical ids. tiktoken is used here for speed and to match
`llm.c` exactly.

## 2.8 The dataset: FineWeb

**FineWeb** (Penedo et al. 2024, arXiv:2406.17557) is ~15T GPT-2 tokens distilled from 96
CommonCrawl snapshots (2013–2024). The pipeline: URL blocklist → `trafilatura` text extraction from
WARC → fastText language ID (English, score ≥0.65) → quality/repetition heuristics adapted from C4
and Gopher plus custom filters → **per-snapshot MinHash deduplication** (5-grams, 14 bands × 8
hashes) → PII anonymization.

The notable finding: *global* cross-snapshot dedup performed **worse** than per-snapshot dedup,
because it preferentially retained low-quality outliers. FineWeb is considered high quality because
each filtering decision was validated by training identical small models and comparing downstream
benchmarks, not by inspection.

- `sample-10BT` / `sample-100BT` are random subsets of ~10B / ~100B tokens.
  `data/fineweb.py:60-67` selects between them via `--version 10B|100B`.
- **FineWeb-Edu** is FineWeb re-filtered by an educational-quality classifier: Llama-3-70B-Instruct
  scored 450k samples 0–5, a lightweight classifier was trained on embeddings of those annotations
  and applied at scale, documents scoring ≥3 kept → ~1.3T tokens. Trains better models per token on
  knowledge/reasoning benchmarks by concentrating expository prose. Shipped as
  `data/cached_finewebedu10B.py` but **the speedrun target is defined on plain FineWeb.**

**Lineage in one line each.** *CommonCrawl*: the raw free web crawl everything derives from.
*C4* (2019): one CC snapshot with heuristic cleaning, ~750 GB, the T5 corpus. *The Pile* (2020):
825 GiB of 22 curated sources, not CC-only. *RefinedWeb* (2023): the argument that aggressively
filtered+deduped CC *alone* beats curated mixtures; powered Falcon. *Dolma* (2024): 3T tokens with
a fully open pipeline.

## 2.9 The `.bin` shard format, exact

Defined in `write_datafile`, `data/fineweb.py:28-52`. Designed to be readable from C (inherited
directly from `llm.c`).

| Byte offset | Type | Content |
|---|---|---|
| 0 | `int32` | **magic = 20240520** (`fineweb.py:37`) |
| 4 | `int32` | **version = 1** (`fineweb.py:38`) |
| 8 | `int32` | **token count** (`fineweb.py:39`) |
| 12–1023 | `int32[253]` | zero padding |
| **1024** → | `uint16[]` | token payload, little-endian |

Header is `np.zeros(256, dtype=np.int32)` = **1024 bytes**. Guards: `assert len(toks) < 2**31`
(fits the int32 count field) and per-token `0 <= t < 2**16` (`fineweb.py:42-44`).

**Filenames.** `f"fineweb_{split}_{shard_index:06d}.bin"` where
`split = "val" if shard_index == 0 else "train"`: **shard 0 is always the validation shard.**

**Shard size** is 100,000,000 tokens (`--shard_size` default `10**8`, `fineweb.py:57`). Documents
spill across shard boundaries; the final partial shard is flushed at `fineweb.py:122-126`.

**Reading it back** (`train_gpt.py:1698-1706`): read 1024 bytes as `int32[256]`, assert
`header[0]==20240520` and `header[1]==1`, take `n = header[2]`, read `n` `uint16`. The read goes
into a **pinned** buffer via `f.readinto(tokens.numpy())` (`train_gpt.py:1703-1705`). This avoids
both a `bytes`→array copy and a later page-lock copy. See Part 7.8 for why pinning matters.

## 2.10 Getting the data

| Script | HF repo | Val file | Train chunks | Local dir |
|---|---|---|---|---|
| `cached_fineweb10B.py` | `kjj0/fineweb10B-gpt2` | `fineweb_val_000000.bin` | 103 | `fineweb10B` |
| `cached_fineweb100B.py` | `kjj0/fineweb100B-gpt2` | `fineweb_val_000000.bin` | 1030 | `fineweb100B` |
| `cached_finewebedu10B.py` | `kjj0/finewebedu10B-gpt2` | `finewebedu_val_000000.bin` | 99 | `finewebedu10B` |

Each downloads exactly **1 val shard** plus `num_chunks` train shards, skipping files already
present. Pass an integer to limit: `python data/cached_fineweb10B.py 9` fetches 900M tokens.
At 2 bytes/token, one shard ≈ 200 MB; the full 10B set ≈ 20 GB. The comment says this *"saves about
an hour of startup time compared to regenerating them."*

**Token budget math.** The val set is exactly 10 Mi tokens = 20 MiB. Track 1 now reaches target in
**under 400M tokens** over 1285 iterations, under 4% of the 10B shard set, which is why the
quickstart downloads only 9 shards. Track 3's baseline runs 3250 steps × 524,288 tokens ≈ 1.7B
tokens, so `python data/cached_fineweb10B.py 20` (2B tokens) is the right fetch there.

## 2.11 Evaluation: HellaSwag (optional, not scored)

HellaSwag (Zellers et al. 2019) is 4-way multiple-choice commonsense sentence completion. A context
from ActivityNet captions or WikiHow, one human-written continuation, three machine-generated
distractors. Construction used **Adversarial Filtering**: wrong endings were LM-generated and
iteratively resampled until a discriminator ensemble could no longer separate them from the truth,
then human-validated. So the wrong answers are fluent and on-topic; surface statistics don't solve
it. Humans >95%, random 25%, GPT-2 small ~30%, GPT-3 175B ~79%, frontier models >95% (saturated).

**Implementation** (`evals/hellaswag.py`): loads `Rowan/hellaswag` validation split, 10042 tasks
(`:145`, `:172`). Packs many tasks into one long sequence delimited by `<|endoftext|>`
(~800 tasks per ~260k-token sequence, `:32`), runs one forward pass, computes per-token
cross-entropy on the *ending* tokens only, takes the **mean** over each candidate's span
(`:135`), and picks the `argmin` (`:136`). Reports plain accuracy (`:190`). Tasks are sharded
across ranks after `shuffle(seed=42)` (`:146-147`) and counts are `all_reduce`-summed
(`:161-166`).

**Why length normalization is required.** Total log-probability decreases monotonically with token
count, since every extra token multiplies in a probability < 1. Unnormalized summed loss therefore
systematically prefers the *shortest* ending regardless of plausibility, a pure tokenization
artifact. Dividing by ending length converts total surprisal into average surprisal per token. It
is a heuristic, not a Bayes rule; PMI-style normalization against an unconditional baseline is the
principled alternative.

**It is not part of scoring.** Nothing in `hellaswag.py` feeds the record criterion; it only prints
(`:194`). The repo uses HellaSwag only in the *scale-up* argument (Part 1.6).

## 2.12 Validation loss vs downstream benchmarks

They diverge because (a) loss is dominated by high-frequency low-information tokens while benchmarks
hinge on rare decisive tokens; (b) architectural tricks can buy loss without buying capability (the
README names logit softcapping); (c) distribution shift (FineWeb val loss says nothing about code or
math); (d) tokenizer-dependence.

The Goodhart risk is real and the repo names it. The structural defenses: the data pipeline is
frozen, the val set is never trained on, the target is a *matched-quality bar* rather than unbounded
loss minimization, and the counter-evidence offered is downstream.

## 2.13 Chinchilla, and why <400M tokens is not a violation

Hoffmann et al. (2022) showed, via three independent methods, that parameters and tokens should
scale roughly equally for a fixed compute budget, the famous **~20 tokens per parameter**
compute-optimal ratio. Chinchilla (70B/1.4T) beat Gopher (280B/300B) at equal compute.

For 124M parameters, compute-optimal ≈ **2.5B tokens**. `llm.c` used **10B** (~80 tokens/param,
deliberately past optimal, since GPT-2 *replication* was the goal). The speedrun reaches the same
loss in **under 400M tokens** (~3.2 tokens/param), ~8× *below* Chinchilla-optimal.

**What that trade actually is.** Chinchilla's frontier assumes a specific recipe: AdamW, standard
architecture, standard schedules. It describes optimal allocation *on that curve*, not a law about
the information content of text. The speedrun moves the curve: better optimization (Muon's ~1.5×
sample efficiency), better architecture, better schedules. It spends **more compute per token**
(five Polar Express matmul-pairs per Muon step, extra embedding tables, auxiliary MTP and
prefix-prediction losses) to need far fewer tokens. Sample-efficiency win, not a scaling-law
violation.

## 2.14 Training vocabulary you will hit in the code

**Global batch size in tokens** is the only meaningful batch quantity. Micro-batch is what fits in
one GPU; gradient accumulation sums micro-batch gradients before stepping; data parallelism splits
across ranks. All three should leave the math invariant, and this repo makes it explicit:
`grad_accum_steps = 8 // world_size` (`train_gpt.py:57`) so 4 GPUs accumulate 2 micro-steps and
reproduce the 8-GPU math. "Batch size 64" is meaningless for LLMs; "524,288 tokens/step" is the
number that determines gradient noise scale and therefore the usable learning rate.

**Document packing.** Documents are variable-length, so the pipeline concatenates them into one flat
stream separated by `<|endoftext|>` and slices fixed-length windows. Zero tokens wasted on padding.
The cost is cross-document contamination inside a window, addressed here with FA3 varlen +
document boundaries, EoS-aligned batch starts (record #26), and a `max_doc_len` cap (records #29,
#78).

**Steps vs epochs vs tokens.** `tokens = steps × global_batch_tokens`. LLM pretraining is typically
**<1 epoch**: the speedrun sees ~4% of the shard set exactly once. There is essentially **no
memorization-style overfitting** in this regime, so train and val loss track each other closely, and
"validation loss" measures optimization progress far more than generalization gap.

**Run-to-run variance.** Identical code with a different seed gives a different final loss:
different init, different data order, and nondeterministic reduction order in bf16/fp8 kernels. The
spread is ~±0.005 nats, comparable to or larger than what individual records claim. **A single run's
val loss is not evidence.** This is why record directories contain dozens of `.txt` logs.

---

# Part 3: The Hardware

*You cannot reason about this repo without knowing what the machine does. Every optimization in
Parts 6–8 is an exploitation of something below.*

## 3.1 What an H100 is, in numbers

The 8-GPU servers this benchmark runs on use the **SXM5** variant (mezzanine module, NVLink-attached,
700 W), not the PCIe card.

| Property | H100 SXM5 |
|---|---|
| Architecture | Hopper (GH100), TSMC 4N, 80B transistors, 814 mm² die |
| Streaming Multiprocessors (SMs) | **132** |
| FP32 CUDA cores | 16,896 (128/SM) |
| Tensor Cores | 528 (4/SM, 4th generation) |
| Boost clock | ~1.98 GHz |
| Memory | **80 GB HBM3**, 5120-bit bus, **~3.35 TB/s** |
| L2 cache | 50 MB |
| Shared memory / L1 per SM | 256 KB unified (up to ~227 KB usable as shared per block) |
| Register file per SM | 256 KB (65,536 × 32-bit registers) |
| NVLink 4 | 900 GB/s bidirectional per GPU (18 links × 50 GB/s) |
| PCIe | Gen5 ×16, 128 GB/s |
| TDP | 700 W |

**Peak throughput** (dense / with 2:4 structured sparsity):

| Format | Dense | Sparse |
|---|---|---|
| FP64 (tensor) | 67 TFLOPS | - |
| FP32 (CUDA cores) | 67 TFLOPS | - |
| TF32 (tensor) | ~495 TFLOPS | ~989 TFLOPS |
| **BF16 / FP16 (tensor)** | **~989 TFLOPS** | ~1979 TFLOPS |
| **FP8 (tensor)** | **~1979 TFLOPS** | ~3958 TFLOPS |

"Sparsity" here means NVIDIA's **2:4 structured sparsity**: if in every group of 4 weights exactly 2
are zero, the tensor core can skip them and double throughput. It requires a pruned-and-retrained
model. **This repo does not use it**; those numbers are marketing headroom, not achievable here.

**The number that matters most for kernel design:** BF16 peak ÷ HBM bandwidth
≈ 989e12 ÷ 3.35e12 ≈ **295 FLOPs per byte**. Any operation doing fewer than ~300 FLOPs per byte
moved is **bandwidth-bound**, not compute-bound. Almost every elementwise op, every normalization,
and every transpose is bandwidth-bound. This single ratio explains why Part 6's kernels exist.

## 3.2 The execution model

```
thread  →  warp (32 threads)  →  thread block / CTA  →  thread block cluster (Hopper)  →  grid
```

- **Thread**: one lane of execution with its own registers and program counter.
- **Warp (32 threads)**: the real scheduling unit. All 32 lanes issue the *same* instruction each
  cycle (**SIMT**). If an `if` sends some lanes one way and some the other, the hardware executes
  *both* paths with the inactive lanes masked off; this is **warp divergence**, and a fully
  divergent branch costs 2× (or 32× for a 32-way switch). Each SM has 4 warp schedulers.
- **Thread block / CTA**: a group of warps (up to 1024 threads) guaranteed to run on **one SM**,
  sharing that SM's shared memory and able to `__syncthreads()`.
- **Thread block cluster (new in Hopper)**: a group of CTAs guaranteed co-resident in the same
  **GPC** (Graphics Processing Cluster), able to read each other's shared memory
  (*Distributed Shared Memory*) and synchronize. Extends the "fast local cooperation" radius beyond
  one SM.
- **Grid**: all blocks of one kernel launch.

**Occupancy** is the ratio of resident warps per SM to the hardware maximum (64 warps on H100). It is
limited by whichever of these runs out first: registers per thread (65,536 total per SM), shared
memory per block (up to ~227 KB), or blocks per SM (32). Occupancy matters because it is how the SM
hides latency: when one warp stalls on an HBM load (~500 ns), the scheduler issues from another. Low
occupancy means nothing to switch to. But *high occupancy is not the goal*: a kernel with big
register tiles and few warps can beat a high-occupancy one if it has enough instruction-level
parallelism.

## 3.3 The memory hierarchy

| Level | Size (H100) | Latency | Bandwidth |
|---|---|---|---|
| Registers | 256 KB/SM | ~1 cycle | enormous |
| Shared memory / L1 | 256 KB/SM unified | ~30 cycles | ~20+ TB/s aggregate |
| L2 | 50 MB (chip-wide) | ~200 cycles | ~7 TB/s |
| HBM3 | 80 GB | **~450–600 ns** | **3.35 TB/s** |
| Host RAM over PCIe5 | - | µs | 128 GB/s |

*(HBM latency is from third-party microbenchmarks, not an NVIDIA-published figure.)*

**Memory coalescing.** The memory controller services a warp's loads in 128-byte transactions. If
the 32 lanes of a warp touch 32 *consecutive* 4-byte addresses, that is one transaction. If they
touch addresses separated by a large stride (as happens when you read a matrix column-wise, or
write the output of a transpose), each lane triggers its own 32-byte sector fetch, and you move up
to **32× the traffic for the same logical bytes**.

This is the single most important hardware fact for reading `triton_kernels.py`. The repo's
`_transpose_copy_kernel` exists precisely because PyTorch's generic elementwise transpose has
non-coalesced writes. The fix is the classic one: read a tile coalesced into registers/shared
memory, transpose *on chip*, write out coalesced.

**Shared-memory bank conflicts.** Shared memory is 32 banks × 4 bytes wide, striped: address `a`
lives in bank `(a/4) mod 32`. If two lanes of a warp hit different addresses in the *same* bank, the
accesses serialize. The classic trap is a `tile[32][32]` float array accessed column-wise: every
lane hits bank 0. The classic fix is padding to `tile[32][33]`. Triton and TMA handle swizzling
automatically; hand-written CUDA does not.

## 3.4 Hopper-specific features (all of these appear in this repo)

**4th-gen Tensor Cores and `wgmma`.** Ampere's `mma.sync` is a *warp*-level instruction: 32 threads
cooperatively hold operand fragments in registers and issue a synchronous matrix multiply. Hopper
adds `wgmma`, **warpgroup** matrix-multiply-accumulate, operating across 4 warps (128 threads) and
crucially **asynchronous**: the instruction is issued, and the warpgroup can do other work while it
completes, waiting on it later. It can also read the **A operand directly from shared memory**
instead of requiring a register staging step. This is what makes producer/consumer pipelining
(below) possible.

**TMA (Tensor Memory Accelerator).** Dedicated hardware that copies a multi-dimensional tile between
global and shared memory given a *descriptor* (base pointer, tensor shape, strides, box size). One
thread issues `cp.async.bulk.tensor`; the TMA unit handles all address generation, out-of-bounds
clamping, and swizzled shared-memory layout. Completion is signalled via an `mbarrier`.

Why it wins, mechanically:
- **No registers burned** holding in-flight data or addresses. On Ampere a large tile load costs
  hundreds of index-arithmetic instructions occupying issue slots.
- **One thread issues it**: 127 of 128 threads in the warpgroup stay free.
- **Hardware swizzling** removes shared-memory bank conflicts for free.
- Writes back to global can carry a fused elementwise reduction (add/min/max/and/or).

In this repo TMA appears as Triton `TensorDescriptor.from_tensor` in
`linear_relu_square_kernel` (`triton_kernels.py:613-623`); the loads there are
`a_desc.load([offs_am, offs_k])` with **no pointer arithmetic and no mask**, because TMA does bounds
handling in hardware.

**Thread block clusters and Distributed Shared Memory**, described in 3.2. Not explicitly used by
this repo's hand-written kernels; the Triton compiler may exploit them.

**The Transformer Engine and FP8.** Hopper's tensor cores natively support two 8-bit float formats.
Their exact bit layouts matter because the constant `448` is hardcoded all over this codebase:

| Format | S/E/M | Bias | Max finite | Min normal | Rel. precision |
|---|---|---|---|---|---|
| **E4M3** (`float8_e4m3fn`) | 1/4/3 | 7 | **448** | 2⁻⁹ ≈ 1.95e-3 | ~6.3e-2 |
| **E5M2** (`float8_e5m2`) | 1/5/2 | 15 | **57344** | 2⁻¹⁴ ≈ 6.10e-5 | ~0.125 |

**Why E4M3's max is 448**: this trips people up. Under normal IEEE convention the all-ones exponent
field is reserved for Inf/NaN, which would cap E4M3 at `2^7 × 1.875 = 240`. But the OCP FP8 spec
(and Hopper hardware) **deviates for E4M3**: it has **no infinities**, and reserves only the single
bit pattern `S.1111.111` for NaN. That frees exponent field 15 for normal numbers. So the largest
finite value uses unbiased exponent `15 − 7 = 8` and the largest non-reserved mantissa `110` →
significand `1 + 1/2 + 1/4 = 1.75`. Hence **2⁸ × 1.75 = 448**. The `fn` in `float8_e4m3fn` literally
means "**f**inite, **N**aN-only".

E5M2 *does* follow IEEE (Inf and NaN in exponent field 31), so its max is `2^15 × 1.75 = 57344`.

**The convention (E4M3 forward, E5M2 gradients) and why.** Forward activations (post-normalization)
have a fairly tight distribution; what limits accuracy is *precision*, so you want the mantissa bit.
Gradients have a much wider, heavier-tailed distribution that shifts by orders of magnitude over
training; what limits them is *dynamic range*, so you want the exponent bit. This repo follows it
exactly: E4M3 at `train_gpt.py:78-79`, E5M2 at `train_gpt.py:114`.

**Asynchronous execution / warp specialization.** With `wgmma` and TMA both asynchronous, you can
split a CTA's warpgroups into *producers* (issue TMA loads for tile n+1) and *consumers* (run wgmma
on tile n), synchronizing through a shared-memory circular buffer. Data movement and math genuinely
overlap rather than alternating. This is the core of FlashAttention-3 (Part 5.3).

**DPX instructions**: hardware acceleration for dynamic-programming inner loops (min/max with
saturation). Not relevant to this repo.

## 3.5 Multi-GPU: NCCL and the collectives

**NCCL** (NVIDIA Collective Communications Library) implements MPI-style collectives as CUDA
kernels, topology-aware, using NVLink/NVSwitch where available. Critically: **NCCL collectives run
as kernels on SMs.** They compete with your compute kernels for the machine. This is the mechanism
behind the kernel comment in `triton_kernels.py:785`:

> Replaces PyTorch's elementwise `copy_` which uses a naive 75k-block kernel with non-coalesced
> writes, saturating all SMs and blocking NCCL.

Unpacked: PyTorch's generic `TensorIterator` elementwise kernel launches ~75,000 tiny blocks for a
large tensor. Those blocks occupy all 132 SMs for the kernel's whole duration. The NCCL all-reduce
that was supposed to run *concurrently* cannot be scheduled, so communication stops overlapping with
compute. A well-tiled transpose finishes in a fraction of the time and leaves headroom.

**The collectives, worked.** Four ranks, four elements each:

```
rank0: [ 1,  2,  3,  4]      rank2: [ 9, 10, 11, 12]
rank1: [ 5,  6,  7,  8]      rank3: [13, 14, 15, 16]
```

| Collective | Result |
|---|---|
| `all_reduce(SUM)` | every rank: `[28, 32, 36, 40]` |
| `all_reduce(AVG)` | every rank: `[7, 8, 9, 10]` |
| `reduce_scatter_tensor(AVG)` | rank0 `[7]`, rank1 `[8]`, rank2 `[9]`, rank3 `[10]` |
| `all_gather_into_tensor` (from the above) | every rank: `[7, 8, 9, 10]` |
| `all_to_all_single` | rank0 `[1,5,9,13]`, rank1 `[2,6,10,14]`, … (a distributed transpose) |
| `reduce(dst=0, AVG)` | rank0 only: `[7,8,9,10]`; others undefined |
| `broadcast(src=0)` | every rank: `[1,2,3,4]` |

**Ring all-reduce, mechanically.** Arrange the N ranks in a ring. Split each rank's tensor into N
chunks. *Phase 1 (reduce-scatter)*: for N−1 steps, each rank sends one chunk to its neighbour and
adds the chunk it receives; after N−1 steps rank *i* holds the fully-reduced chunk *i*.
*Phase 2 (all-gather)*: for another N−1 steps, pass the completed chunks around the ring until
everyone has all of them. Total bytes moved per rank: `2(N−1)/N · S`, independent of N in the
limit, which is why ring all-reduce scales.

**Why `reduce_scatter` + `all_gather` = `all_reduce`, and why splitting them is better.** Compare
rows 3 and 4 of the table: they compose exactly to row 2. That is not a coincidence; it is how ring
all-reduce is implemented internally. But *splitting the halves apart* buys something all-reduce
cannot: **between them, each rank holds only 1/N of the gradient, and that is the only part it
needs.** So each rank can keep optimizer state (momentum, second moments) for only its 1/N shard,
compute the update for that shard, and then `all_gather` the updated **parameters** instead of the
gradients. Optimizer memory drops N×, and the optimizer's arithmetic (Adam's elementwise ops,
Muon's Polar Express iterations) is divided 8 ways instead of redundantly recomputed on all 8 GPUs.
**This is ZeRO stage 1.** It is record #24 in the leaderboard, and it is exactly what
`train_gpt.py:600-619` implements.

**`ReduceOp.AVG`** is not "sum then divide in Python"; it is a NCCL-native op (`ncclAvg`), so the
division happens inside the reduction kernel, saving a pass over memory and avoiding intermediate
overflow. Note: NCCL-only; Gloo does not implement it.

## 3.6 Where H100 sits

| | A40 | A100 SXM | **H100 SXM** | H200 SXM | B200 |
|---|---|---|---|---|---|
| Arch | Ampere (GA102) | Ampere (GA100) | **Hopper** | Hopper | Blackwell |
| Memory | 48 GB GDDR6 | 80 GB HBM2e | **80 GB HBM3** | 141 GB HBM3e | 192 GB HBM3e |
| Bandwidth | 696 GB/s | 2.0 TB/s | **3.35 TB/s** | 4.8 TB/s | ~8 TB/s |
| BF16 dense | 149 TFLOPS | 312 TFLOPS | **989 TFLOPS** | 989 TFLOPS | ~2250 TFLOPS |
| FP8 | **not supported** | **not supported** | **1979 TFLOPS** | 1979 TFLOPS | ~4500 TFLOPS |
| FP4 | no | no | no | no | ~9000 TFLOPS |
| GPU↔GPU | NVLink3 bridge, pairs only | NVLink3, 600 GB/s | **NVLink4, 900 GB/s** | 900 GB/s | NVLink5, 1.8 TB/s |
| TDP | 300 W | 400 W | 700 W | 700 W | 1000 W |

Two consequences worth internalizing:

1. **The A40 has no FP8 tensor cores, no TMA, no wgmma, no clusters.** Track-3 experiments that ran
   on A40s (e.g. record #46's eight seed logs are `A40_seed0..7`) genuinely cannot use the Hopper
   paths, and that is *fine*, because track 3 scores steps, not wall-clock.
2. **H200 is the same compute die as H100**; only memory changed. Every FLOP-bound optimization
   transfers unchanged; only bandwidth-bound kernels improve.

## 3.7 MFU: Model FLOPs Utilization

From the PaLM paper (Chowdhery et al. 2022):

```
MFU = (model FLOPs per token × tokens/sec) / (num GPUs × peak FLOPs/sec)
```

The numerator counts only the FLOPs the *mathematical model definition* requires. The standard
approximation for a dense model with N non-embedding parameters over D tokens is **6ND** (2ND
forward + 4ND backward), plus attention's `12·L·H·S·d` per sequence. It **excludes**
activation-recomputation FLOPs; the variant that includes them is **HFU** (Hardware FLOPs
Utilization), and HFU ≥ MFU always.

Denominator here: 989 TFLOPS BF16 × 8 GPUs = **7.9 PFLOPS**. Well-tuned large-scale training
typically achieves **35–50% MFU**. Record #10 in the leaderboard is literally named `MFUTweaks`.

## 3.8 Cost context

An 8×H100 node rents for roughly **$15–25/hour** in the 2025–2026 spot market (PrimeIntellect,
Lambda, RunPod, Modal). A 75-second record run therefore costs on the order of **$0.35** of GPU
time. A track-3 baseline run (~15 min on 8×H100) costs roughly **$5**. That price point is precisely
why this benchmark exists in a form individuals can attack, and it is the number to quote when
someone asks whether they can afford to reproduce a record.

---
# Part 4: The Optimizer Story

> Muon is the headline contribution of this repo. This part explains
> the math, its origin, and (critically) the gap between what the README describes and what the
> code actually runs.

---

## 4.0 The headline warning

**The README describes classic Muon. `train_gpt.py` does not implement classic Muon.**

- `README.md` contains `zeropower_via_newtonschulz5` with the famous fixed quintic coefficients
  `(3.4445, -4.7750, 2.0315)`, 5 iterations, bf16.
- `train_gpt.py` implements a class called **`NorMuonAndAdam`** (`train_gpt.py:374`) whose
  orthogonalizer is **`polar_express`** (`train_gpt.py:177`) using **five *different* coefficient
  triples**, one per iteration (`train_gpt.py:168-174`), followed by a **NorMuon** per-row variance
  correction.
- `records/track_3_optimization/train_gpt_simple.py` (the frozen track-3 baseline) implements yet a
  third thing: `zeropower_via_newtonschulz5` with `a, b, c = 2, -1.5, 0.5` and **12 iterations**
  (`train_gpt_simple.py:163-186`), the *convergent* classical coefficients, because track 3 does
  not care about wall-clock.

If you read the README and then read the code expecting a match, you will be confused. All three
are correct for their context. Verified directly against source.

---

## 4.1 Optimization foundations

**Gradient descent.** A network has parameters θ and a scalar loss `L(θ)`. The gradient `∇L(θ)` says,
for each knob, which way and how hard the loss moves. Descent walks downhill: `θ ← θ − η∇L(θ)`.
Computing `∇L` over the whole dataset is prohibitive, so we estimate it on a random mini-batch.
That is **SGD**. One forward + one backward + one update = **one step**. In track 3 a step consumes
524,288 tokens, and the metric is how few steps reach 3.28.

**Why curvature is the whole problem.** The gradient gives the steepest *direction* but says nothing
about how far you can safely travel. If the loss surface is a long narrow ravine (steep across,
nearly flat along), the largest learning rate that doesn't diverge across the steep axis is far too
small to progress along the flat axis. The ratio of steepest to shallowest curvature is the
**condition number** κ, and plain SGD's convergence rate degrades directly with it. **Every
algorithm in this repo is an attempt to fix ill-conditioning cheaply.**

**Momentum.** Heavy-ball (Polyak 1964): `m_t = μ·m_{t-1} + g_t`, then `θ ← θ − η·m_t`. Oscillations
across the ravine cancel; consistent progress along it accumulates.

**Nesterov** differs in *where the gradient is evaluated*, at the extrapolated look-ahead point:
`g_t = ∇L(θ_{t-1} − ημ·m_{t-1})`. The practical difference is a correction proportional to the
*change* in gradient: Nesterov brakes as the landscape turns; heavy-ball overshoots. In convex
problems this improves the rate from O(1/k) to O(1/k²).

**Adaptive methods.**
- **AdaGrad** (Duchi, Hazan & Singer, JMLR 2011; no arXiv): per-coordinate step, divide by √(running
  *sum* of squared gradients). The denominator only grows, so the effective LR decays to zero.
- **RMSProp** (Tieleman & Hinton, Coursera 2012; never published): replace the sum with an EMA,
  `v_t = β₂v_{t-1} + (1−β₂)g_t²`, so old information is forgotten and the step size can recover.
- **Adam** (arXiv:1412.6980): RMSProp's second moment + momentum's first moment.
  ```
  m_t = β₁m_{t-1} + (1−β₁)g_t        v_t = β₂v_{t-1} + (1−β₂)g_t²
  m̂ = m_t/(1−β₁ᵗ)                    v̂ = v_t/(1−β₂ᵗ)          # bias correction
  θ ← θ − η·m̂/(√v̂ + ε)
  ```
  **Bias correction** exists because the EMAs start at zero and under-estimate early; without it the
  first hundreds of steps take absurdly small updates. **ε** is a numerical floor preventing division
  by ~0. Track-3 submissions tune ε as low as `1e-15`, deliberately pushing Adam toward pure
  sign-descent.
- **AdamW** (arXiv:1711.05101) fixed a real bug. Classic "weight decay" was L2 added to the *loss*,
  so the penalty gradient `2λθ` passes **through** the adaptive denominator. Parameters with large
  gradient history get *less* decay. That is an accident, not a design. AdamW decouples:
  `θ ← θ − η·(m̂/(√v̂+ε)) − ηλθ`. Decay is now uniform and its optimal value no longer moves when you
  change the learning rate. **This matters doubly for Muon**, where the preconditioner is a full
  orthogonalization; L2 would get orthogonalized along with everything else, which is meaningless.

**Second-order.** The **Hessian** `H = ∇²L` is the full local curvature map. Newton's method:
`θ ← θ − H⁻¹∇L`. It is *affine invariant*: it rescales each direction by its own curvature, turning
the ravine into a bowl. It is also infeasible: for 124M parameters `H` has ~1.5×10¹⁶ entries
(petabytes) and inversion costs O(n³). A **preconditioner** `P` is any cheap stand-in for `H⁻¹` (or
`H^{-1/2}`, or a whitening matrix), a change of coordinates applied to the gradient before
stepping. In this framing **Adam is a diagonal preconditioner**, and everything below buys more
off-diagonal structure without paying O(n²).

**Kronecker factorization** is the trick that makes this tractable for matrix-shaped parameters. A
weight `W ∈ ℝ^{m×n}` has `mn` parameters; its true preconditioner is `mn × mn`. The Kronecker
product `A ⊗ B` of an `m×m` and an `n×n` matrix produces exactly such a matrix from only `m² + n²`
numbers. Crucially `(A ⊗ B)⁻¹ = A⁻¹ ⊗ B⁻¹`, and applying it to `vec(G)` is just `A⁻¹ G B⁻¹`, two
small matmuls. For a 768×3072 layer: ~10M numbers instead of 5.6×10¹². **K-FAC**
(arXiv:1503.05671) was the ancestor, approximating each layer's Fisher block as
(activation covariance) ⊗ (backprop-gradient covariance). Every "Kronecker" method below
(Shampoo, SOAP, PSGD-Kron, PMuon) varies which two matrices you factor and what power you raise
them to.

---

## 4.2 Muon: the core idea in plain English

SGD and Adam are **coordinate-wise**: every scalar weight gets its own step from its own gradient
history. The fact that 589,824 of those scalars happen to be arranged as a 768×768 linear map is
completely invisible to the optimizer. Permute the rows and columns of the weight matrix and Adam's
update permutes identically; it has no notion of "this matrix stretches inputs a lot along
direction *v*".

**Muon** (**M**oment**U**m **O**rthogonalized by **N**ewton-Schulz) treats a 2-D parameter as a
*linear operator*. It computes the ordinary momentum matrix, then **replaces that matrix with the
nearest matrix whose action is a pure rotation/reflection** (same directions, uniform stretch) and
steps with that instead.

Reference implementation (github.com/KellerJordan/Muon):

```python
def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)                            # M ← βM + (1-β)G
    update = grad.lerp_(momentum, beta)                       # Nesterov blend
    update = zeropower_via_newtonschulz5(update, ns_steps)    # ≈ U V^T
    update *= max(1, update.size(-2)/update.size(-1))**0.5    # RMS / aspect-ratio match
    return update
# in step():  p.mul_(1 - lr*weight_decay);  p.add_(update, alpha=-lr)
```

Memory: **one** state tensor per parameter. Adam needs two.

---

## 4.3 The orthogonalization math

**SVD, in layman's terms.** Any matrix `G` (m×n) can be written `G = U S Vᵀ`. Read right-to-left:
*rotate* the input into a special frame (`Vᵀ`), *stretch* each of those axes by an independent factor
`σ₁ ≥ σ₂ ≥ … ≥ σᵣ` (the **singular values**, diagonal of `S`), then *rotate* into the output frame
(`U`). The singular values are "how much stretch in each direction", and they carry **all** the
magnitude information; `U` and `V` are pure rotations.

**Why snap all σ to 1.** Jordan's empirical observation, reproduced at 1.1B scale by the NorMuon
paper (arXiv:2510.05491, Fig. 1a): momentum/Adam updates for transformer 2-D parameters have **very
high condition number**: they are nearly low-rank, with a handful of directions carrying almost all
the energy. The pathology: your learning rate is a single scalar, so it gets set by the *largest*
singular direction to avoid divergence, which means the small-σ directions receive essentially zero
step. Those rare directions may be exactly the useful ones. Replacing `U S Vᵀ` with `U Vᵀ` keeps
every direction and gives them all equal budget. κ becomes 1.

Formally, `Ortho(G) = argmin_O { ‖O − G‖_F : OᵀO = I or OOᵀ = I } = U Vᵀ`. This is the **orthogonal
polar factor**: `G = (UVᵀ)(VSVᵀ)` splits `G` into "rotation" × "positive stretch". It is also
`sign(G)` in the matrix-sign sense, and equals `(GGᵀ)^{-1/4} G (GᵀG)^{-1/4}`, a form that matters in
4.9.

### The deep point: norms and steepest descent

A **norm** is a ruler, a rule assigning a single "size" to a vector or matrix. Gradient descent is
*not* norm-free. The real statement of a gradient step is:

> Δ = argmin over all Δ of ⟨G, Δ⟩ + (λ/2)‖Δ‖²

"move where the linearized loss drops fastest, *per unit of size*." **The answer depends entirely on
which ruler you use:**

| Ruler | Resulting update | Known as |
|---|---|---|
| Euclidean / Frobenius | `Δ ∝ −G` | vanilla SGD |
| Max-norm (ℓ∞) | `Δ ∝ −sign(G)` | sign-SGD ≈ Adam-without-EMAs |
| **Spectral norm** `‖A‖₂ = σ_max(A)` | **`Δ = η·U Vᵀ`** | **Muon** |

The spectral norm measures "the largest factor by which A can stretch any input vector." Solving
`max ⟨G,Δ⟩ s.t. ‖Δ‖₂ ≤ η` gives `Δ = η·UVᵀ`, with optimal value `η·‖G‖_*` (the nuclear norm, dual
to the spectral norm). **Muon is steepest descent under the spectral norm.** That is the theoretical
claim of Bernstein & Newhouse, *Old Optimizer, New Norm: An Anthology* (arXiv:2409.20325), extended
constructively in *Modular Duality in Deep Learning* (arXiv:2410.21265).

**Why the spectral norm deserves to be the ruler.** A hidden layer's job is to map activations to
activations. What controls whether the forward pass blows up or dies is the *operator norm* of each
layer, not the Euclidean length of its flattened weight vector. Bounding `‖ΔW‖₂` bounds the change
in the layer's output for the worst-case input. So "lr in units of spectral norm per update" is a
**scale-meaningful** learning rate in a way Adam's is not.

**Why not just call SVD?** Correctness isn't the issue; throughput is. SVD is iterative, branch-heavy
and poorly parallelized. GPUs are matmul engines (~989 TFLOP/s bf16 on H100 vs a tiny fraction of
that on LAPACK-style decompositions). Jordan also rejected the coupled Newton iteration used in
Shampoo implementations because it needs ≥ float32 to stay stable. **Newton–Schulz is matmuls only
and survives bf16.**

---

## 4.4 Newton–Schulz iteration

**History.** Schulz (1933) gave the iteration for matrix inverses. Its use for
orthogonalization/polar decomposition runs through Kovarik (1970), Björck & Bowie (1971), and
Higham's *Functions of Matrices* (SIAM 2008, eq. 5.22). It is a fixed-point iteration for the matrix
sign function using only matrix products.

**The iteration:**
```
X ← a·X + b·X(XᵀX) + c·X(XᵀX)²
```

**Why odd, why quintic.** Substitute `X = USVᵀ`; since `XᵀX = VS²Vᵀ`, everything telescopes:
`X ← U(aS + bS³ + cS⁵)Vᵀ`. **`U` and `V` never move.** The iteration is a scalar map
`φ(x) = ax + bx³ + cx⁵` applied to each singular value independently. Only **odd** polynomials in
`X` have this property (even powers of a rectangular `X` aren't even shape-legal, and the sign
function is itself odd). After N steps you get `U φ^N(S) Vᵀ`, where `φ^N` is the N-fold
**composition**: degree 5 composed 5 times reaches effective degree 5⁵ = 3125 for 15 matmuls.
Cubic costs 2 matmuls/step but converges slower per matmul; Jordan reports testing cubic and septic
and finding neither improved wall-clock.

**Pre-normalization.** Convergence requires all σ ∈ [0,1] to start. Dividing by `‖G‖_F` guarantees
this since `‖G‖_F ≥ ‖G‖₂ = σ_max`, and it is free because `Ortho(cG) = Ortho(G)` for `c > 0`. But it
is *conservative*: for high-rank `G`, `‖G‖_F` can be √r times larger than σ_max, crushing all
singular values toward zero, which is precisely why `φ` must climb steeply near x = 0. In this repo:

```python
X = X / (X.norm(dim=(-2,-1), keepdim=True) * (1 + 2e-2) + 1e-6)   # train_gpt.py:198
```

The `1 + 2e-2` is a safety factor guarding against bf16 rounding pushing σ above the polynomial's
stability basin.

**Non-convergent coefficients.** `φ'(0) = a`, so `a` governs how fast tiny singular values get
rescued. Jordan maximizes `a` subject to `lim φ^N(x)` landing near 1, and found a spread of ~±0.3 is
harmless to loss curves. The result, `(a,b,c) = (3.4445, −4.7750, 2.0315)`, is a polynomial that
**oscillates around 1 forever and never converges**. The output is `U S' Vᵀ` with S' spread around
1, not `U Vᵀ`. κ ≈ 1.5–2.2 instead of 10³, and that is enough.

**Why bf16 is safe.** You need ~1 decimal digit of accuracy, not 7. The iteration is a contraction
toward the fixed point at 1, so rounding error injected at step k is partially healed at step k+1.
There is no ill-conditioned inverse anywhere. Contrast Shampoo's inverse-fourth-roots, where small
eigenvalues amplify error.

---

## 4.5 Polar Express: what this repo actually runs

*The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm*
(Amsel, Persson, Musco & Gower, arXiv:2505.16932). Their framing of the prior art is blunt: classical
Newton–Schulz has slow *initial* convergence, and Jordan's tuned heuristic coefficients **fail to
converge at all**.

**The fix: stop using one polynomial repeatedly.** Since N steps compose into `p = p_T ∘ … ∘ p₁`,
treat each `p_k` as a separate design variable and choose it **greedily optimally**: given that the
current singular values lie in `[ℓ_k, 1]`, pick the odd degree-5 polynomial minimizing the worst-case
deviation `max_{x∈[ℓ_k,1]} |p_k(x) − 1|`. That is a **minimax (Chebyshev) problem**, solved via an
equioscillation/Remez-style characterization. Lineage: Chen & Chow; Nakatsukasa & Freund
(Zolotarev-type optimal sign methods). They also explicitly engineer finite-precision safety so it
runs in bf16.

**Consequence: early iterations get wildly aggressive coefficients, later ones relax toward the
classical values.** Verified verbatim at `train_gpt.py:167-174`:

```python
# Computed for num_iters=5, safety_factor=2e-2, cushion=2
polar_express_coeffs = [
    (8.156554524902461,  -22.48329292557795,  15.878769915207462),   # a~8.16: enormous slope at 0
    (4.042929935166739,   -2.808917465908714,  0.5000178451051316),
    (3.8916678022926607,  -2.772484153217685,  0.5060648178503393),
    (3.285753657755655,   -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377,  -1.7097828382687081, 0.42323551169305323),  # → classical (2, -1.5, 0.5)
]
```

Step 1 drags tiny singular values up from the Frobenius-normalization floor; the tail steps do
precision cleanup. **The coefficient set is tuned to the iteration count**; that is why it is a list
of five triples, not one triple used five times.

**Shape specialization.** The quintic is implemented in two branches to keep the intermediate small
(`train_gpt.py:202-253`):
- Tall (`rows > cols`): `A = XᵀX` (`train_gpt.py:216`), `B = b·A + c·A²` (`:217`), `X ← a·X + X·B`
- Wide: `A = XXᵀ` (`:244`), `B = b·A + c·A²` (`:245`), `X ← a·X + B·X`

All three matmuls (`XTX`, `XXT`, `ba_plus_cAA`) are **hand-written Triton kernels** exploiting output
symmetry. See Part 6.2. A code comment at `train_gpt.py:219-222` explains why they are separate
kernels: referencing `X` twice makes PyTorch "make a defensive copy, resulting in a
`cudaMemcpyAsync` in `baddbmm`."

The whole thing is `@torch.compile(dynamic=False, fullgraph=True)` (`train_gpt.py:176`) with the
docstring:

> Fused Nesterov momentum + Polar Express Sign Method. Nesterov momentum is applied in FP32, then the
> result is cast to BF16 for polar express orthogonalization, avoiding materialization of the FP32
> intermediate between graph breaks.

---

## 4.6 NorMuon: the second layer

*NorMuon: Making Muon more efficient and scalable* (Li, Liu, Liang, Chen & Zhao, arXiv:2510.05491).

**The diagnosis:** Muon fixes the *spectrum* but not the *rows*. Orthogonalization makes singular
values uniform, yet the L2 norms of individual **neurons** (rows of the update matrix) remain highly
non-uniform: some neurons get far bigger updates than others. AdamW, being coordinate-wise, is
naturally good at this and bad at the spectrum. **The two are complementary, not competing.**

**The fix:** after orthogonalization, keep an EMA of the **mean squared value per row** and divide
each row by its RMS.

```
v_t ← β₂·v_{t-1} + (1−β₂)·mean_cols(O_t ⊙ O_t)     # v_t in R^m: m scalars, not m x n
Ô_t ← O_t ⊘ (√v_t + ε)
η̂  ← 0.2·η·√(mn) / ‖Ô_t‖_F                          # RMS re-match
W   ← W − ηλW − η̂·Ô_t
```

**Correction to a common mis-statement:** this is *not* Adafactor's row×column rank-1
outer-product factorization. Adafactor (arXiv:1804.04235) stores an m-vector **and** an n-vector and
reconstructs `V ≈ (r cᵀ)/sum(r)`. NorMuon stores **only** the m-vector, a rank-1 statistic along one
axis. The paper describes it as "similar to Adam-mini's block-wise reduced-dimensional statistics,
where we treat each neuron (i.e. each row) as a block", citing Adafactor only as the memory-reduction
precedent. The shared takeaway is the same: *a full m×n second-moment tensor is mostly redundant; a
per-row summary captures most of the signal at 1/n the memory.* Cost: `m(n+1)` vs Muon's `mn` vs
Adam's `2mn`.

**The paper's own honest caveat:** for a *perfectly* orthogonalized full-rank update with m ≤ n,
every row norm is exactly 1 and the normalization is a no-op. **NorMuon only helps because the
Newton–Schulz / Polar-Express output is deliberately approximate.**

**In this repo** (`_apply_normuon_variance_reduction`, `train_gpt.py:934-947`): reduces along
whichever axis is larger (`red_dim = -1 if rows ≥ cols else -2`), maintains the `beta2` EMA (`:942`),
rescales by `rsqrt` (`:943`), then algebraically fuses the normalize-then-renormalize so the update's
Frobenius norm is preserved: `final_scale = step_size · v_norm/v_norm_new` (`:946`).

Reported: 21.74% fewer steps than Adam at 1.1B (vs Muon's 10.43%), 13.91% at 5.4B, ~3% per-iteration
latency over AdamW.

---

## 4.7 The `NorMuonAndAdam` class: how it actually runs

`train_gpt.py:374-947`. Note it does **not** subclass `torch.optim.Optimizer`; it uses per-parameter
configs keyed by a `.label` attribute (`:374-424`).

`step(do_adam=True)` (`:723-823`) runs **three phases**:

**Phase 1: Scatter (`:745-764`).** Iterate `scatter_order`; for each param launch an *async*
gradient reduction (`_launch_reduce`, `:588-628`). When embed and lm_head are tied, `transpose_add`
folds `embed.grad.T` into `lm_head.grad` (`:758`) and embed is skipped (`:761`).

**Phase 2: Work (`:766-801`).** Iterate `work_order`; wait on each param's reduce future (`:781`),
apply `_adam_update` (`:792`) or `_normuon_update` (`:794`), then launch an async all-gather for
sharded params (`:797`).

**Phase 3: Finalize (`:803-823`).** Wait lm_head gather first (`:806`), `transpose_copy`
lm_head→embed if tied (`:810`), wait remaining gathers, clear grads (`:823`).

**`scatter_order` and `work_order` are deliberately different orderings** (`:406-410`, `:2052-2063`)
so that large gradient reductions are launched early and overlap with the compute of small params.
There are **no backward hooks**; the schedule is explicit.

`_normuon_update` (`:869-912`): cast grad to FP32 (`:874`) → fused Nesterov + Polar Express (`:882`)
→ NorMuon variance reduction (`:889`) → cautious weight decay + in-place update with mantissa
tracking (`:902`/`:907`).

---

## 4.8 Every remaining mechanism in the optimizer

**Momentum schedule** (`get_muon_momentum`, `train_gpt.py:1995-2007`): linear warmup μ 0.85→0.95 over
the first 300 steps, hold, then linear cooldown 0.95→0.85 over the final 50. Passed as a **0-D CPU
tensor** (`_momentum_t`, `:463`, `:876`) to avoid `torch.compile` recompiles (see Part 7.5).

**Shape scaling** (`:510`): `shape_mult = max(1.0, chunk_shape[-2]/chunk_shape[-1])**0.5`, folded
into `lr_mul` (`:511`). This is the canonical Muon aspect-ratio correction. Why it's needed:
`Ortho(G)` for an m×n matrix has `min(m,n)` unit singular values, so its Frobenius norm is
`√min(m,n)` and its **per-entry RMS depends on the aspect ratio**. Without the correction, a square
update and a 4:1 update have systematically different per-entry sizes under one shared learning
rate. The correction makes one LR transferable across differently-shaped layers.

**The 0.2 RMS target.** The NorMuon/Moonshot variant `η̂ = 0.2·η·√(mn)/‖Û‖_F` targets a fixed update
RMS of 0.2, chosen to match the typical RMS of an *Adam* update. This is what makes Muon learning
rates directly comparable to a tuned AdamW LR. arXiv:2502.16982 identifies "carefully adjusting the
per-parameter update scale" as one of the two things needed to make Muon work out-of-the-box at
scale.

**Cautious Weight Decay** (arXiv:2510.12402). The idea: **decay a weight only where the update and
the weight have the same sign**, i.e. only where decay and the optimizer already agree on direction.
Elsewhere, decay is skipped, so the original loss is preserved rather than implicitly replaced by a
regularized one. The parent idea is *Cautious Optimizers* (arXiv:2411.16085), which gates the whole
update by sign agreement `mask = (u ⊙ g) > 0`, famously "one line of code". In this repo:
`mask = (grad * p_precise) >= 0`, then `p ← p − p·mask·wd·lr − grad·lr` (`train_gpt.py:929-930`).
This is leaderboard record #43 (track 1) and part of record #46 (track 3).

**The split-buffer FP32 master weight.** Parameters are stored in **bf16** (`:970`), but NorMuon
keeps a **7-bit mantissa side-buffer** as `uint16` (`:575`, `:916-932`). It reconstructs an FP32
value from `(bf16 << 16) | mantissa` (`:927`), applies the update in FP32, then writes back both the
bf16 high bits and the mantissa (`:931-932`). Net effect: a full FP32 master weight stored in 32
bits split across two buffers, with no stochastic rounding needed. Momentum buffers are plain FP32
(`:561`).

**No gradient clipping anywhere.** Verified: there is no `clip_grad` call in the optimizer.
Orthogonalization implicitly bounds update magnitude (spectral norm ≤1 after Polar Express).

**Gradient accumulation.** `grad_accum_steps = 8 // world_size` (`:57`),
`grad_scale = 1/grad_accum_steps` (`:58`), which keeps magnitudes device-count-invariant.

---

## 4.9 Which parameters get which optimizer, and why

`param_table`, `train_gpt.py:2026-2050`:

| Optimizer | Parameters |
|---|---|
| **NorMuon** | `qk_bank`, `vo_bank`, `mlp_bank`: the 2-D attention & MLP projection matrices **only** |
| **Adam** | `scalars`, `smear_gate`, `ve_gate_bank`, `lm_head`, `bigram_embed`, `post_lambdas`, `resid_lambdas`, `value_embeds`, `embed`, all `mudd_*` |

**Why the split** (docstring `:388-389`): orthogonalization is only meaningful for 2-D weight
matrices where the singular-value spectrum describes a linear map between two dense activation
spaces.

- **Biases, gains, scalars are 1-D**: there is no operator to orthogonalize.
- **Embeddings and the LM head are 2-D but are not linear maps between activation spaces.** An
  embedding row is selected by a one-hot index, so the natural norm on that weight space is a
  row-wise ℓ2/ℓ∞ norm, not the spectral norm. Bernstein & Newhouse's whole thesis is that **the norm
  should be assigned per-layer based on the layer's semantic role**, and their modular-duality
  framework derives a *different* dual map for `Embed` than for `Linear`. Jordan notes the embedding
  case follows from theory; the output-head case does **not** and is purely empirical.

**Adam hyperparameters** (`:2030-2076`). Defaults: `lr=0.008, eps=1e-10, weight_decay=0.005`.
NorMuon defaults: `lr=0.023, momentum=0.95, beta2=0.9, weight_decay=1.2`. Notable per-param overrides:
`scalars` β=(0.9,0.99) lr×5.0 wd×0; `lm_head`/`embed` β=(0.5,0.95) wd×150; `bigram_embed`
β=(0.75,0.95) lr×75 wd×5; `smear_gate` lr×0.01; `value_embeds` lr×75.

**Adam runs only on odd steps** (`_is_adam_step`, `:2104-2106`, used at `:750/:776/:821`): record
#39, "update Adam every other step". NorMuon steps every iteration.

**Embed/lm_head tying.** They are tied until 2/3 of training (`split_step`, `:1948`, forced odd,
`:2146-2147`). Before the split, embed is synced from lm_head each step via the transpose kernels
(`:737-741`, `:808-810`); at the split, `copy_lm_state_to_embed` (`:657-694`) all-gathers lm_head's
Adam state, transposes it, and re-shards it to embed's layout. Record #53.

---

## 4.10 Learning-rate schedule

- **Per-stage base multiplier** (`lr_mul`, `:1979-1989`): `1.0 → 1.52 (=(16/8)^0.6) → 1.73
  (=(24/8)^0.5)`, tied to the batch-size schedule 8→16→24. Batch-size-aware LR scaling.
- **Cooldown** (`get_lr`, `:1968-1976`): starting at `(1 − cooldown_frac)` of scheduled iterations
  with `cooldown_frac = 0.60` (`:1992`), LR decays linearly toward **0.15×** base:
  `lr·(1−t) + 0.15·t`. Record #19 was "decay to 0.1 not 0.0"; decaying to a nonzero floor beats
  decaying to zero.
- **No LR warmup ramp.** Warmup is handled by the *momentum* schedule instead, plus the
  compile-priming loop. This is unusual and worth knowing.
- Applied per-param at `:2138`: `p_cfg.lr = initial_lr · step_lr`, then each param's own `lr_mul`
  is applied inside its update.

This is a **WSD (warmup-stable-decay)** schedule in spirit (arXiv:2405.18392). WSD's advantage over
cosine is that it does not require committing to the total step count in advance: you can fork a run
at any point in the stable phase, which makes scaling studies vastly cheaper.

---

## 4.11 Muon's empirical record, and the honest criticisms

**Documented advantages.** (i) *Sample efficiency*: README claims ~1.5×; the Oct 2024 switch from
AdamW to Muon cut the speedrun 35% and Muon has survived every subsequent record. CIFAR-10 speedrun:
3.3 → 2.6 A100-seconds. (ii) *Memory*: one state tensor, ~half of Adam's. (iii) *Wall-clock*: FLOP
overhead is `T·m/B` (T = NS steps, m = model dim, B = tokens/batch): 0.7% for the speedrun, 0.5% for
a Llama-405B-shaped run. README claims <2%.

**Validated at scale, but only partially and with modifications.** Moonshot's *Muon is Scalable
for LLM Training* (arXiv:2502.16982) trained **Moonlight**, a 3B-active/16B-total MoE on 5.7T tokens,
reporting ~2× compute efficiency vs AdamW at compute-optimal, **but only after adding** (1) weight
decay and (2) per-parameter update-scale adjustment, neither of which vanilla Muon had. **Kimi K2**
(arXiv:2507.20534) trained a 1T-param/32B-active MoE on 15.5T tokens with **MuonClip** (Muon plus a
new "QK-clip" technique) and specifically advertises "zero loss spike". *The advertisement is the
tell:* attention-logit blowup was a real Muon-at-scale failure mode requiring a bespoke fix.

**Honest summary: the idea has been validated to trillion-parameter scale by an independent lab, but
plain Muon-as-published is not what shipped.**

**Criticisms and open questions:**
- *Numerical.* The Polar Express authors state outright that Jordan's coefficients fail to converge.
  A legitimate objection from numerical analysis, even though Jordan's counter (a ±0.3 spread
  doesn't hurt the loss) held up empirically for a year, and Polar Express's gains are real but
  modest.
- *Incompleteness.* NorMuon's entire premise is a documented deficiency: orthogonalization alone
  leaves neuron-norm imbalance on the table.
- *Distribution.* Jordan's own listed open question. Orthogonalization is inherently a non-sharded
  operation: you need the whole matrix. NorMuon notes the Moonshot distributed implementation
  gathers full momentum on **all** devices and replicates the orthogonalization; NorMuon's FSDP2
  version fixes this but adds 33–50% communication volume. This repo sidesteps it by reshaping
  parameters into "banks" whose leading dim is divisible by `world_size` (`train_gpt.py:504-508`) so
  whole matrices stay on one rank.
- *Scope.* Jordan explicitly flags that Muon may work only for pretraining, not finetuning or RL.
  Treat pretraining as the demonstrated regime.
- *Coverage.* Muon touches only hidden 2-D weights. You still need a tuned AdamW for embeddings, the
  head, and all scalars, so it's a hybrid, not a replacement, and it inherits AdamW's tuning burden
  on the parameters that are often most sensitive.

**The strongest defense, and worth quoting to a skeptic:** the evidence is not "our paper's table
beats a baseline we tuned." It is a *competitive* benchmark where Muon is **slower per step**, so any
researcher who could tune AdamW to match its sample efficiency would win the record by deleting
Muon. Dozens of records by many independent researchers have not done so. That is real, if indirect,
adversarial validation, stronger than most optimizer papers offer. It is also, by construction,
evidence about *small* models on *one* dataset.

---

## 4.12 The Shampoo equivalence

Shampoo (Gupta, Koren & Singer, arXiv:1802.09568) maintains accumulated preconditioners
`L = Σ GGᵀ`, `R = Σ GᵀG` and updates with `L^{-1/4} G R^{-1/4}`. Bernstein & Newhouse observed:
**delete the accumulation** (use only the current gradient) and the algebra collapses:

```
(GGᵀ)^{-1/4} G (GᵀG)^{-1/4}
 = (US²Uᵀ)^{-1/4}(USVᵀ)(VS²Vᵀ)^{-1/4}
 = (US^{-1/2}Uᵀ)(USVᵀ)(VS^{-1/2}Vᵀ)
 = U S^{-1/2} S S^{-1/2} Vᵀ
 = U Vᵀ
```

So **accumulation-free Shampoo ≡ orthogonalized gradient ≡ steepest descent under the spectral
norm.** Add momentum before orthogonalization and you have Muon. Rohan Anil (a Shampoo author)
publicly endorsed this reading.

Track 3 gives this a clean empirical statement: result #22 proves the identity
**SpectralDescent(lr, wd) ≡ Muon(mu=0) ≡ Shampoo(betas=(0,0), power=−1/4)**, and it takes
**8225 steps** vs the baseline's 3250. That is the single most informative negative result in the
leaderboard: **momentum is essential; orthogonalization alone is ~3× worse.**

---

## 4.13 Why the `-1/4` power in Shampoo

Two quarter-powers, one on each side, *compose* to make the combined Kronecker preconditioner an
approximate `-1/2` power of the full-matrix AdaGrad preconditioner. Some implementations use `-1/2`
per side, which is more aggressive and behaves closer to full whitening; track-3 result #21
parameterizes this explicitly. Cost is dominated by inverse-pth-roots (eigendecompositions), which
is why implementations amortize them with a **preconditioner update frequency** (recompute every k
steps; track-3 results sweep `precond_freq` from 1 to 5).

---
# Part 5: The Model Architecture

> Everything here is `train_gpt.py` (track 1). The track-3 baseline
> `train_gpt_simple.py` is a deliberately stripped version (see Part 9).

**Configuration** (`train_gpt.py:2234-2241`):
`vocab_size=50257` (padded to 50304), `num_layers=11`, `num_heads=6`, `head_dim=128`,
`model_dim=768`. Note `6 × 128 = 768`, an unusual choice: **six heads of width 128**, where GPT-2
small had twelve heads of width 64.

---

## 5.1 Attention from first principles

A transformer layer lets token *i* pull information from tokens *j*. Attention is a
**differentiable, content-addressed lookup**.

Each token's residual vector `x_i` is projected three ways: `q_i = x_i W_Q`, `k_i = x_i W_K`,
`v_i = x_i W_V`. Read as a soft dictionary: `k_j` is the *address* token *j* advertises, `v_j` is the
*payload* it hands over, `q_i` is the *query* token *i* issues. Nothing intrinsically makes them
query/key/value. Those roles emerge because the loss rewards weight matrices that behave that way.

```
Attention(Q,K,V) = softmax(QKᵀ / √d_k) V
```

**Why `1/√d_k`.** If `q` and `k` entries are roughly independent with unit variance, `q·k` is a sum
of `d_k` products, so its variance grows like `d_k` and its magnitude like `√d_k`. For `d_k = 128`
raw scores sit around ±11. Softmax on ±11 is nearly a hard argmax: one weight ≈1, the rest ≈0, and
the gradient through a saturated softmax row is ≈0. Dividing by `√d_k` renormalizes to O(1) so
gradients survive. **It is a variance-control constant, not a modelling choice.**

**Softmax** turns scores into a distribution: `p_ij = exp(s_ij)/Σ_j exp(s_ij)`. Output
`o_i = Σ_j p_ij v_j` is a convex combination of payloads. Softmax makes attention *competitive*:
mass given to one key is taken from another.

**Causal masking.** An autoregressive LM factorizes `P(x_1..x_n) = Π P(x_t | x_<t)`. Training
computes the loss for all `n` positions in one pass, so position `t` must not see `t+1..n` or it
would trivially copy the answer. The mask sets `s_ij = -∞` for `j > i`; `exp(-∞) = 0`.

**Multi-head.** A *single* softmax row can express only one weighting pattern per token. A
syntactic-agreement head and a coreference head want incompatible weightings. `H` heads give `H`
independent lookups at the same FLOP cost (since `H · d_head = d_model`). The tradeoff: each head's
score matrix is rank-limited by `d_head`, the **low-rank bottleneck** that DCFormer attacks (5.10).

**The O(n²) cost.** The score matrix is `n×n` *per head, per layer*. At `n = 65,536` (this repo's
original FlexAttention context), one head's score matrix has 4.3×10⁹ entries, or 8.6 GB in bf16.
Materializing it is absurd. This is why everything in 5.3–5.5 exists.

---

## 5.2 What this repo's attention actually is

`CausalSelfAttention` (`train_gpt.py:1082-1162`). **It holds no weights** (`:1091` comment). Q/K/V/O
weights are passed in from external parameter *banks* per forward call, so the optimizer can shard
them freely (Part 4.11). `__init__(dim, head_dim, num_heads, paired=False)` asserts
`num_heads*head_dim == model_dim` (`:1090`).

**Backend: FlashAttention-3, not FlexAttention, not SDPA.** `train_gpt.py:1069` loads
`kernels-community/flash-attn3`; `:1139-1141` calls `flash_attn_interface.flash_attn_varlen_func`.
A comment at `:1138` records the switch from FlexAttention.

The full ordered pipeline for one non-paired layer:

1. Fused QKV projection (scaled by `sa_lambdas[0]`, pre-multiplied into the weight), reshaped to
   `(1, T, 3H, d_head)`, chunked.
2. **QK-norm**: `q, k = norm(q), norm(k)` (`:1106`).
3. **Half-truncated RoPE + YaRN**: rotate the first 64 head dims; leave the last 64 stationary.
4. **Partial key offset** (long-window layers, `:1111-1113`): shift the stationary half of `k`
   forward one position.
5. **Value embedding injection**: `v = v + aux_v` (`:1115-1116`).
6. **FA3 varlen** with `causal=True`, `window_size=(bm_size, 0)`, `softmax_scale=yarn.attn_scale`,
   `cu_seqlens=seqlens` (`:1139-1141`).
7. **DCMHA post-only no-DD correction** (layer 10, `:1142-1149`): a Triton kernel over a 112-token
   window, added to the FA3 output.
8. **XSA** (`:1151-1157`, layers [1,3,4,7]): subtract `tanh(α)·proj_v(y)`.
9. **Sparse attention gate** (`:1158-1159`, layers [3,10]): per-token per-head multiplicative gate.
10. Concatenate heads, output projection (`sa_lambdas[1]` pre-multiplied into `W_O`).

---

## 5.3 FlashAttention: the family

**FA1: tiling + online softmax.** The insight is that attention is **memory-bound, not
compute-bound**. A naive kernel writes `S = QKᵀ` (n²) to HBM, reads it back for softmax, writes `P`
(n²), reads it back for `PV`. The arithmetic is cheap; the HBM round-trips are not.

FA never materializes `S` or `P`. It tiles `Q` into `B_r`-row blocks and `K,V` into `B_c`-column
blocks sized to fit SRAM, loops over key blocks, and maintains **running statistics** per query row.
For one row with running max `m`, denominator `l`, and unnormalized accumulator `o`, all initialized
`m=-∞, l=0, o=0`:

```
for each key block:
  s = block scores
  m_new      = max(m, max(s))
  correction = exp(m - m_new)            # retroactive rescale of the old state
  l          = l * correction
  o          = o * correction
  p          = exp(s - m_new)            # subtract the max, else exp overflows fp32
  l          = l + sum(p)
  o          = o + p @ V_block
  m          = m_new
output = o / l
```

This is **exactly equal** to the full softmax, algebraically identical, not an approximation. The
trick: softmax's normalizer is a sum, sums are associative, and the max-subtraction can be
retroactively corrected by a single scalar multiply. HBM traffic drops from O(n²) to O(n²·d/M) where
M is SRAM size: 2–4× wall-clock and **O(n) memory instead of O(n²)**.

**LSE and why backward is cheap.** At the end the kernel writes one extra number per query row:
`LSE_i = m_i + log(l_i)`, the log-sum-exp of that row's scores. That single scalar reconstructs the
whole probability row on demand: `p_ij = exp(s_ij - LSE_i)`. The backward pass recomputes `s_ij` from
`Q,K` (a cheap matmul) and normalizes with the stored LSE: **no second max pass, no stored n²
matrix**. You trade a recomputed GEMM (cheap, tensor cores) for O(n²) memory traffic (expensive).
Storage per sequence: `n` floats instead of `n²`.

**FA2** is the same math with better scheduling: defer the `1/l` division to the end of the row
rather than rescaling every block (fewer non-matmul FLOPs); parallelize over the *sequence*
dimension not just batch×heads (fills the GPU for long-sequence/small-batch); split work across
warps so K/V tiles are shared. ~2× over FA1, ~70% of A100 peak.

**FA3** (arXiv:2407.08608) is Hopper-specific. FA2 hit only ~35% utilization on H100 because it did
not use Hopper's asynchrony. FA3 adds:
- **Warp specialization / producer-consumer.** Warpgroups split into *producers* issuing TMA loads
  of the next K/V tile and *consumers* running WGMMA, synchronized through a shared-memory circular
  buffer. Data movement and math genuinely overlap.
- **Interleaved (ping-pong) softmax and GEMM.** Softmax runs on the multi-function unit
  (exponentials); GEMM runs on tensor cores, *different hardware*. FA3 schedules block *i*'s softmax
  concurrently with block *i+1*'s GEMM, hiding the softmax latency FA2 exposed.
- **FP8** with per-block scale factors and *incoherent processing*, a random orthogonal (Hadamard)
  transform applied to Q and K to spread outliers across dimensions before quantizing.

Reported: **1.5–2.0× over FA2 on H100; ~740 TFLOP/s FP16 (75% utilization); ~1.2 PFLOP/s FP8, with
2.6× lower numerical error than baseline FP8 attention.**

**varlen and `cu_seqlens`.** Training data is many short documents. Padding to a common length wastes
FLOPs. Varlen packs documents end-to-end into a single flat `(total_tokens, H, D)` tensor. This repo
asserts `B == 1` at `:1095` precisely because varlen replaces the batch dimension. `cu_seqlens`
("cumulative sequence lengths") is the boundary index: documents of length 300, 500, 200 give
`[0, 300, 800, 1000]`. The kernel binary-searches it to find each query block's document and clamps
its key loop to `[doc_start, doc_end)`. **No cross-document leakage, and no wasted work computing
masked-out scores.**

---

## 5.4 FlexAttention, and why the repo left it

FlexAttention is a `torch.compile`-based attention compiler. You supply two callbacks and it fuses
them into a FlashAttention-style Triton kernel:

- **`score_mod(score, b, h, q_idx, kv_idx) -> score`**: arbitrary elementwise transform of the
  pre-softmax score. ALiBi, softcapping, relative-position biases are each one line.
- **`mask_mod(b, h, q_idx, kv_idx) -> bool`**: a *pure predicate*. Because it is boolean and
  position-only, it can be evaluated **ahead of time** on a coarse grid.

That precomputation is the **BlockMask**: the n×n mask summarized at block granularity (e.g.
128×128) into fully-masked / fully-unmasked / partial. Fully-masked blocks are skipped entirely;
fully-unmasked blocks skip per-element mask arithmetic. For a sliding window at n = 64K this is an
enormous saving.

The repo used FlexAttention from record #12 (Nov 2024, "1024-ctx dense → 64K-ctx FlexAttention",
7.2 → 5.03 min, the single largest jump in the leaderboard) and moved to FA3 at record #29
(Sep 2025, 2.731 min).

**The tradeoff.** FlexAttention buys expressiveness at the cost of being a *generated* kernel: it
cannot use Hopper warp specialization, TMA producer/consumer pipelining, or FP8. FA3 is a
hand-written CUTLASS kernel: much faster on H100, but you get only the knobs it exposes. The repo's
evolution is exactly that trade: once the needed pattern was just "causal + sliding window +
varlen", flexibility stopped being worth the throughput. **Anything FA3 doesn't support, the repo
now writes as a separate Triton kernel and adds as a correction term**, which is precisely what
DCFormer became (5.10).

---

## 5.5 Sliding window, warmup, and attention sinks

**Sliding window.** Restrict token `i` to attend to `[i-W, i]`. Cost drops O(n²) → O(n·W).
Information still propagates globally, but only via *depth*: after `L` layers the effective receptive
field is ~`L·W`. What you lose is any single-layer long-range copy: a retrieval needing token 5000
to look directly at token 3 must be relayed layer by layer, and each relay is lossy.

The per-layer schedule (`:1494`) is
`[short, short, short, LONG, short, short, None, short, short, short, LONG]`. Layers 3 and 10 get
the long window; **layer 6 is `None` because attention is skipped there entirely** (5.8). This is
Gemma-2-style interleaved local/global layers.

**Attention window warmup** (record #13). Windows grow across 4 training stages in units of
`block_size`: `(1,3) → (3,7) → (5,11) → (6,13)` (`:1980-1987`), extended to 20 for the final
validation (`:1928`). Two benefits: (1) early steps are dramatically cheaper (the quadratic term is
tiny) and early steps learn trivial local statistics (bigrams, morphology) that don't need long
range anyway; (2) it acts as a **curriculum**, forcing local structure to be learned before the model
can lean on long-range shortcuts. On every window change, YaRN is re-applied (`:2115-2116`).

**Attention sinks.** Softmax rows must sum to 1: a head has *no way to say "nothing here is
relevant"*. Models resolve this by dumping surplus mass onto a semantically empty, always-visible
token, typically BOS/position 0. This is the **attention sink**. It matters operationally: naive
sliding-window inference that evicts position 0 from the KV cache causes catastrophic perplexity
blowup, because the head's excess mass suddenly redistributes onto real tokens. (StreamingLLM's fix
is to always retain the first few tokens.)

**Sparse attention gate** (record #28). Instead of a sink, give the head an explicit off switch. At
layers 3 and 10 the head output is multiplied by a learned per-token per-head scalar:

```python
y = y * attn_gate_w.type_as(y).view(B, T, self.num_heads, 1)   # train_gpt.py:1159
```

Note this gates the **output**, not the softmax: the distribution still sums to 1, but the head can
contribute ~0 to the residual stream for a given token. **That is a true learned no-op**, and it
removes the need to waste window budget and KV capacity on a sink.

---

## 5.6 Positional encoding: RoPE, YaRN, and the half-truncation

**Why any.** Attention is a sum over `j` of `p_ij v_j`. Permute the input tokens and the sum is
unchanged; self-attention is *permutation-equivariant*. Without injected position, "dog bites man"
and "man bites dog" are the same object.

**Absolute** encodings (learned or sinusoidal) add a position vector at layer 0. Learned tables
cannot extrapolate at all. **Relative** encodings (T5 bias, ALiBi) inject `i−j` into the score, which
is what language actually cares about.

**RoPE** implements relative encoding as an *absolute* operation. Split each head vector into `d/2`
coordinate pairs; rotate pair `m` of the vector at position `t` by angle `t·θ_m`:

```
[x_2m  ]   [cos(tθ_m)  -sin(tθ_m)] [x_2m  ]
[x_2m+1] = [sin(tθ_m)   cos(tθ_m)] [x_2m+1]
```

The magic: a 2-D rotation by α composed with the inverse of a rotation by β is a rotation by α−β. So

```
⟨R(i)q , R(j)k⟩ = ⟨q , R(j-i)k⟩
```

The dot product depends **only on `i−j`**. You apply a cheap, cacheable, KV-cache-friendly absolute
transform per token and get exact relative behaviour for free.

**Base / theta.** `θ_m = base^(-2m/d)`. With `base = 10000`, `m=0` rotates once per ~6 tokens; the
last pair has wavelength ~10000·2π tokens. Low-index pairs are high-frequency "which token exactly"
detectors; high-index pairs are low-frequency "roughly how far" detectors.

**This repo uses base 1024, not 10000** (`train_gpt.py:1006`):

```python
angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=head_dim//4)
```

Sensible: the context here is 2048 tokens, not 100K, so wavelengths tuned for 10K-token contexts
would leave the slowest dimensions nearly constant and wasted.

### Length extrapolation and YaRN

RoPE fails beyond trained length because at test-time position `t > L_train` the high-frequency pairs
have wrapped into angle regions the model never saw. Attention logits go out of distribution and
perplexity explodes.

- **Position Interpolation (PI).** Rescale `t → t·L_train/L_new`. Every frequency divided by the same
  `s`. Guaranteed in-distribution, but it crushes high-frequency dimensions: adjacent tokens now
  differ by a tiny angle, losing local resolution.
- **NTK-aware scaling.** Change the *base* instead (`base → base·s^(d/(d-2))`). Interpolates low
  frequencies a lot and high frequencies almost not at all, but the very highest still get pushed
  slightly out of range.
- **YaRN ("NTK-by-parts")** (arXiv:2309.00071) makes the split explicit and per-dimension. For each
  frequency, compute how many **full rotations** it completes over the original context:
  `r_m = L_train · θ_m / 2π`. Then:
  - `r_m > β` (many rotations → high frequency → local detail): **do not interpolate.** The model has
    already seen every phase of this dimension.
  - `r_m < α` (fewer than α full turns → low frequency → global position): **fully interpolate** by
    `1/s`. The model never saw a full period, so extrapolation here is genuinely unseen territory.
  - In between: linear ramp.

  **α and β are wavelength thresholds measured in "rotations over the training context."** LLaMA's
  published values are `α=1, β=32`; this repo uses exactly those (`train_gpt.py:1037-1041`):

  ```python
  rotations = old_window * self.angular_freq / (2 * torch.pi)
  scaling_factor = old_window / new_window
  interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
  self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
  ```

  Weight 0 (low freq) → multiplied by `scaling_factor`; weight 1 (high freq) → untouched. Exactly
  the paper.

- **Attention temperature.** YaRN's second component: extending context means each query attends to
  more keys, so softmax entropy rises and logits should be sharpened. The paper multiplies the
  attention scale by `√(1/t) = 0.1·ln(s) + 1`. This repo uses a tuned coefficient of 0.2 (`:1055`):

  ```python
  self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1
  ```

  starting from `attn_scale = 0.1` (`:1035`), vs the textbook `1/√128 = 0.088`.

**The non-obvious inversion worth flagging.** YaRN was designed for *post-hoc context extension of a
frozen model*. This repo uses it **during training**, every time the sliding window grows. Each
window expansion is a mini context-extension event, and YaRN keeps the RoPE phases and logit
temperature consistent across the discontinuity so the model isn't shocked. That is record #31.

### Half-truncated RoPE

`train_gpt.py:1006-1009` (credited @YouJiacheng):

```python
angular_freq = (1/1024) ** linspace(0, 1, steps=head_dim//4)   # 32 distinct frequencies
angular_freq = angular_freq.repeat_interleave(2)               # → 64 rotating dims
angular_freq = cat([angular_freq, zeros(head_dim//2)])         # → 64 STATIONARY dims
```

With `head_dim = 128`: half the dimensions rotate, half have frequency exactly zero.

**What that buys.** RoPE forces every dimension's contribution to the logit to oscillate with
distance. A head that wants a purely *content-based* match ("find the token that is the word
'Paris', wherever it is") has to fight the rotation. Reserving half the head dimensions as stationary
gives a clean content-only subspace and a clean position-modulated subspace, rather than making every
feature do both jobs.

**Partial key offset** (record #49) then exploits the stationary half (`train_gpt.py:1111-1113`):

```python
if key_offset:
    k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:]
```

The non-rotating half of each key is shifted one position forward, so a query matching against it is
really matching against *the previous token's* content. **That is a hardwired induction-head
primitive** ("find where token X was last followed by something"), available in a **single** layer,
whereas standard induction heads require two layers to compose.

---

## 5.7 Normalization and stability

**LayerNorm** `y = γ ⊙ (x − μ)/√(σ² + ε) + β`, statistics over the feature dimension.
**RMSNorm** `y = γ ⊙ x / √(mean(x²) + ε)`. Drop the mean subtraction and the bias. Cheaper: one
reduction pass instead of two. Empirically the re-centering does essentially nothing for
transformers; only the re-scaling matters.

**Pre-norm vs post-norm.** Post-norm (original paper): `x ← LN(x + Sublayer(x))`. Pre-norm:
`x ← x + Sublayer(LN(x))`. Pre-norm leaves the residual stream as a **clean identity path** from
embedding to output. The gradient reaches layer 0 without passing through any normalizer's Jacobian.
Post-norm requires LR warmup and careful init to avoid divergence at depth; pre-norm just trains. The
cost: pre-norm's residual magnitude grows with depth, so late layers contribute proportionally less,
the "representation collapse" that MUDD (5.11) targets.

**Parameter-free norm.** `train_gpt.py:952-953`:

```python
def norm(x): return F.rms_norm(x, (x.size(-1),))
```

**No `γ`, no LayerNorm anywhere in the model.** This works because every norm here is immediately
followed by a linear layer, and a per-feature gain `γ` is exactly a diagonal matrix the following
weight can absorb. The gain is redundant parameterization. Removing it saves parameters, removes an
optimizer state, and removes one more thing that can drift.

Placement is pre-norm: `norm(cache.get(7,x))` before attention (`:1598`), `norm(x)` before MLP
(`:1643`), input norm (`:1549`), final `norm(x)` before the LM head (`:1682`).

**QK-normalization** (record #5, credited @Grad62304977). `train_gpt.py:1106`: `q, k = norm(q),
norm(k)`. The failure mode it prevents: nothing bounds `‖q‖` or `‖k‖` during training. If they grow,
logits grow like `‖q‖‖k‖`, softmax saturates, gradients through attention vanish, and (because a
saturated head cannot correct itself) **the growth is self-reinforcing**. In large runs this shows
up as loss spikes and attention-entropy collapse. RMS-normalizing `q` and `k` makes the logit a
scaled *cosine similarity*, bounded by construction; the scale is then supplied explicitly and
controllably by `softmax_scale = yarn.attn_scale`.

**Logit softcapping.** Gemma 2 uses `cap · tanh(z/cap)`: identity for small `z`, asymptotically
clamped for large `z`. It keeps final logits from running away, stabilizing the cross-entropy
gradient.

This repo (`train_gpt.py:1690`, eval path) uses:

```python
logits = 23 * torch.sigmoid((logits + 5) / 7.5)
```

**Here is the fact a skeptic will want.** This is not a different family of function. Since
`tanh(x) = 2σ(2x) − 1`:

```
23·σ((z+5)/7.5)  =  11.5·tanh((z+5)/15) + 11.5
```

So it is an affine-shifted tanh softcap with **half-width 11.5, input shift +5, input scale 15**.
Three deliberate deviations from Gemma's form:

1. **Cap and temperature are decoupled.** Gemma ties them (`cap·tanh(z/cap)`); here the saturation
   half-width (11.5) and the input scale (15) are independent knobs.
2. **The output is strictly positive**, range `(0, 23)` rather than `(−11.5, +11.5)`. Softmax is
   shift-invariant so the distribution is unchanged, but the *numerics* of the fused kernel and the
   FP8 head improve, because a one-sided range quantizes better.
3. **The +5 shift makes it asymmetric**: record #54's "asymmetric logit rescale". Saturation is
   applied more aggressively on one side, matching the fact that in language modelling the
   interesting logits are the large-positive ones.

Evolution in the record history: tanh softcap at 30 (record #9) → 15 (record #18) → sigmoid
reparameterization (credited @YouJiacheng) → the current asymmetric form. It is also **fused into the
cross-entropy kernel** (records #60, #79), which is a further practical reason to prefer the sigmoid
form: `σ' = σ(1−σ)` recomputes the derivative from the already-computed forward value, so the fused
fwd/bwd kernel needs no second transcendental.

The training path uses the fused kernel (`:1687`); the eval path uses the explicit expression plus
fp32 `F.cross_entropy` (`:1688-1692`).

---

## 5.8 Skip connections, U-Net structure, and the residual stream

**U-Net skips** (record #11). Treat the layer stack like a U-Net encoder-decoder: layer `L−k`'s input
receives a direct skip from layer `k`'s output, with a learnable weight. Rationale: early layers hold
surface/lexical features that late layers need to produce the output token, and forcing that
information through the whole residual stream is lossy.

Current form: **layer 6 skips attention entirely** (`train_gpt.py:1591-1595`) and instead adds a
gated skip from `cache[3]`:

```python
x + sigmoid(skip_lambda) * post_skip_gate * cache[3]
```

`skip_lambda` init −1.5 → σ ≈ 0.18 (`:1346`). Cache snapshots are taken at layers [3, 7]
(`:1249`, `:1667-1668`). Gating added in record #55.

**Lambda scalars.** The repo is full of learned scalar mixers:
- `resid_lambdas` init √1.1 per sublayer (`:1338`), cumulative 1.1 growth per layer
- `post_lambdas` init 1 (`:1334`)
- `sa_lambdas` [0.5, 1.0] per layer, scaling QKV and O (`:1344`, applied `:1103`/`:1161`)
- `scalars` bank holds smear + skip lambdas (`:1341-1350`)

Record #47 ("multiply attn lambda with weight not data") and #74 ("removed post-attn lambdas") are
both about folding these into the weights so they cost nothing at runtime.

**Smear gate** (record #34, `:1326-1327`, `:1546-1548`): mixes each token's embedding with a
sigmoid-gated fraction of the *previous* token's embedding: a learned bigram smear, originally
"token += 0.07 · prev-token".

---

## 5.9 The MLP block

**Activation: ReLU-squared (ReLU²), not GELU.** `ReLUSqrdMLP = FusedLinearReLUSquareFunction.apply`
(`train_gpt.py:48`); the kernel comment confirms `relu(x@W1.T)^2 @ W2.T`. GELU *is* used, but only
inside the tiny MUDD/gate helper networks (`:1399`, `:1466`).

Hidden dim = 4×model_dim = 3072 (`:1299`). Weights live in `mlp_bank(12, 2, 3072, 768)` (`:1300`).
Note **12 slots for 11 layers**: index 11 is padding that exists purely so the bank shards evenly
across 8 GPUs (`:1298`). `c_fc` uniform-init, `c_proj` zero-init (`:1307-1308`).

Records #30 and #35 dropped the first MLP layer and the first attention layer respectively; the
model is 11 layers because layers were *removed*, not because 11 was chosen.

FP8 path at `:1403-1434`, `:1644-1658`.

---

## 5.10 Embeddings: five different tables

**Token embedding.** `nn.Embedding(vocab_size, model_dim)` (`:1228`), vocab padded 50257→50304 via
`next_multiple_of_n(vocab_size, n=128)` (`:1216`).

**Tying.** Embed is tied to `lm_head.weight.T` at init (`:1229-1231`) and synced during the tied
phase (`:1531`); untying happens at 2/3 of training, optimizer-driven (Part 4.9).

**Value embeddings** (records #14, #63, #65, #70). `value_embeds = 0.01*randn(5*vocab_size,
model_dim)` in bf16 (`:1261`): **five distinct VE tables** injected into attention V at layers
[1,2,7,8,9] (`:1543`), gated by `2*sigmoid(...)` (`:1611-1616`) so the default is 1.0.

The background: **value residual learning** (arXiv:2410.17897, "ResFormer"). Observation: the
residual stream carries hidden states forward, but **values are recomputed from scratch at every
layer**, so original token-level information gets progressively laundered away. ResFormer adds a
value residual `V_l + λ·V_1`, a shortcut from layer 1's values. SVFormer is the extreme variant
where all layers share layer 1's values (halving KV cache).

**This repo goes one step further**: instead of routing layer 1's values, it learns a **dedicated
embedding table indexed by token id** and adds it directly into `V` (`:1115-1116`). Each token
injects a learned, layer-specific vector into the value stream regardless of what the residual stream
has become.

**FP8 LM head** (`:1225`, record #19). `CastedLinearT(..., use_fp8=True)` with hand-tuned per-tensor
scales `x_s=100/448, w_s=1.6/448` (448 = E4M3 max). LM-head weight normal-init std 0.005 (`:1226`).

**`CastedLinearT`** (`:956-983`) stores its weight **transposed** as `(in_features, out_features)` in
bf16 (`:970`), zero-initialized (`:975`). The comment (`:958-959`) explains: transposed storage is a
kernel-speed fix for gradient accumulation. Record #27 and #61.

**Bigram hash embedding** (record #62, `:1329-1332`, `:1533-1538`, `:1902`). A huge
`bigram_vocab_size = 50304*15//2 = 377,280`-row embedding, combined with a random ±1 **sign table**
(`:1331`, 8192 rows) to compress colliding bigrams: the "sign trick", record #83. Injected into the
**first `bigram_dim=768` channels** of the residual (`:1562`, `:1636`, `:1640`). This is the biggest
single-record jump of the 2026 era (−5.6 s). It also required a custom sparse gradient path
(record #71) because the table is enormous but each step touches few rows.

---

## 5.11 The exotic mechanisms

### DCFormer / DCMHA (arXiv:2405.08553)

**The problem.** Each head's score matrix `A_h = QW_h^Q (KW_h^K)ᵀ` has rank at most `d_head`. With
`d_head = 128` that is a hard ceiling on expressible attention patterns, the **low-rank
bottleneck**. Separately, heads are empirically redundant; many can be pruned with no loss.

**The fix.** Let heads *borrow* each other's attention matrices. Define a composition map
`C ∈ ℝ^{H×H}` and produce `A'_h = Σ_j C_hj A_j`. The paper proves two equivalences: composing
**score** matrices (pre-softmax) is exactly equivalent to QK projections with `H`-fold larger head
dimension (Thm 2.1), directly breaking the rank bottleneck; composing **weight** matrices
(post-softmax) is equivalent to `H`-fold expanded OV projections (Thm 2.2), widening the cross-token
information bandwidth. In circuits language: head 1 keeps its own OV circuit ("what to do with what
it finds") while borrowing head 3's QK circuit ("where to look"), forming a hybrid head that exists
in neither.

**Why "dynamic" matters.** If `C` is a fixed parameter, this is *static* composition and is exactly
equivalent to a bigger projection. DCMHA makes `C` **input-dependent**: `C = C(Q_i, K_j)`, generated
on the fly. That has *no* projection equivalent; it is strictly more expressive.
- *Query-wise composition*: weights generated from `Q_i` alone, shared across keys. O(T·H·R).
- *Key-wise*: generated from `K_j` alone, shared across queries.

Results: DCFormer matches Transformers with ~1.7–2.0× the compute; DCPythia-6.9B beats Pythia-12B.

**What this repo runs: "post-only, no-DD"** (record #85, `dc_triton_kernels.py`). Reading the
reduction backwards:

- **post-only**: only the post-softmax composition is applied; the pre-softmax composition is
  dropped. **This is forced by the architecture**: FA3 never materializes the score matrix, so you
  *cannot* insert a pre-softmax transform without abandoning FA3. Post-softmax composition can be
  added afterward as a correction over a narrow window (`window=112`) and added to the FA3 base
  output. That is literally what `dc_attention_postonly_nodd_correction_add_base_triton`
  (`dc_triton_kernels.py:1266`) does; the `add_base` in the name is the fused residual add.
- **no-DD**: the reference `Compose` has `dw1/dw2` (low-rank dynamic outer product) *and* `dd`
  (dynamic diagonal, per-head self-gating). This kernel takes only `(dc_w1, dc_w2)`, both shape
  `(B,T,H)` indexed by *query* position (hence **query-wise, rank-1**) and no diagonal. Dropping
  `dd` is cheap to justify: **the repo already has an independent per-head gate** (5.5's
  `attn_gate_w`), which is what a dynamic diagonal *is*.
- `post_w1` is RMS-normalized across the head dimension before use (`train_gpt.py:1079`), the same
  stability logic as QK-norm.

"Lightweight" is doing real work here: full DCMHA's cost is prohibitive when the baseline is a
hand-tuned Hopper kernel.

### MUDD / MUDDFormer (arXiv:2502.12170, same authors as DCFormer)

**The problem.** The residual stream is a bottleneck: every layer reads and writes the same additive
channel, and in a pre-norm network the stream's magnitude grows with depth, so deep layers'
contributions get proportionally diluted. DenseNet-style dense connections help but use *static,
shared* connection weights.

**Two ideas.** (1) **Dynamic**: connection weights are generated from the hidden state *at each
sequence position*, so different tokens route differently. (2) **Multiway**: a transformer block has
four *decoupled* input streams (Q, K, V, and the residual), and each gets its **own** set of
connection weights. Layer `l`'s query input can be sourced primarily from layer 3 while its value
input comes from layer 7. Reported: performance of Transformers trained with 1.8–2.4× compute, for
+0.23% parameters and +0.4% FLOPs.

**In this repo** (records #81, #85; `train_gpt.py:1352-1484`): small GELU MLPs (`mudd_w1/w2`, dim 64,
scale 0.1) produce **per-token, data-dependent** skip coefficients. Invoked at the last layer
(`:1603`, 14 coefficients feeding V-injection and residual recombination) and post-loop (`:1678-1680`,
5 coefficients mixing `cache[0], cache[7], cache[9], ve_bank0, cache[3]`).

**Note what the MUDD gate has become.** Its output slots are unpacked into several *different*
consumers (`unpack_pre_mudd_gate`):

```python
xsa_alphas[1] = gate[..., 0:6]     # 6 heads
xsa_alphas[3] = gate[..., 6:12]
attn_gates[3] = gate[..., 12:18]
```

The MUDD machinery is now the model's **general-purpose data-dependent gate supply**, feeding XSA
strengths, sparse attention gates, `x0` skip gates, and bigram-embedding gates. That is worth knowing
before you touch it.

### XSA: Exclusive Self Attention (arXiv:2603.09078, Shuangfei Zhai)

**Premise.** Token `i`'s attention output `o_i` almost always contains a large component parallel to
`v_i`, the token's own value. That component is redundant, because the residual stream already
carries `x_i` forward. Worse, part of the head's output bandwidth is spent re-transmitting
information the network already has. XSA constrains attention to capture only what is **orthogonal to
the token's own value vector**, forcing the head to spend its output on genuine context.

**This repo's learnable partial version** (`train_gpt.py:1151-1157`, record #82):

```python
dot   = (y * v).sum(-1, keepdim=True)
denom = v.square().sum(-1, keepdim=True).clamp_min(1e-8)
alpha = torch.tanh(attn_args.xsa_alpha)          # per token, per head
y = y - alpha * (dot / denom) * v
```

At `α = 1` this is exactly the Gram–Schmidt projection removing the `v`-parallel component. At
`α = 0` it is a no-op. `tanh` bounds `α ∈ (−1,1)`, so the model can also *amplify* the self-component
if that helps. And `xsa_alpha` is **not a free parameter**: it comes from the MUDD gate, so it is
data-dependent per token per head. Record #86 is an algebraic rewrite giving identical math with
fewer ops. It is disabled for paired-head layers because `v`'s shape no longer aligns.

### Paired-head attention (record #58)

`train_gpt.py:1118-1136`, layers [0,2,5,9] (`:1252`). Adjacent head pairs share a doubled `head_dim`
and the sequence is interleaved to 2× length with half the window, so **adjacent heads attend to each
other's keys**. RoPE uses an even/odd-position-interleaved frequency table (`Yarn(paired=True)`,
`:1019-1031`). Cost: halved effective window for those layers. XSA and value-embedding injection are
skipped there.

Record #80 then made Muon orthogonalize Q/K **in head-pairs** rather than as one 6-head matrix
(`:1272-1279`) because the paired structure means the natural operator is per-pair, not per-layer.

### Multi-token prediction (arXiv:2404.19737, Meta) and prefix prediction

Standard training predicts token `t+1` from position `t`. **MTP** adds `n` heads on a shared trunk
predicting `t+1 … t+n` simultaneously, as an **auxiliary loss at training time only** (the extra
heads are discarded, or repurposed for speculative decoding, at inference).

Why it helps: next-token prediction rewards *myopic* features. A model can score well on `t+1` by
pattern-matching the last few tokens. Forcing the same trunk representation to also support
`t+2, t+3` requires encoding information about the *future*. The paper reports gains that grow with
model size (13B: +12% HumanEval, +17% MBPP) and that MTP **accelerates induction-head formation**.

In this repo: `mtp_weights` (record #53), annealed over training via `mtp_weights_start`/`_end` per
stage (`:1981-1985`), fused into `FusedSoftcappedCrossEntropy` (`:1687`).

**Prefix token prediction** (record #88, `train_gpt.py:1175-1198`) is a sibling: the model also
predicts `T'`, the longest proper prefix of the target token that is *itself* in the GPT-2
vocabulary. A purely tokenizer-derived auxiliary target: it gives partial credit for "nearly right"
tokens and injects sub-token structure that BPE otherwise hides. Weight annealed 0.25 → 0.

---

## 5.12 Parameter count: an honest non-answer

**The code never prints a total parameter count, and I will not quote a number it does not state.**
It must be computed from: `qk_bank`, `vo_bank`, `mlp_bank(12,2,3072,768)`, two 50304×768 embeddings,
five 50304×768 value-embedding tables, and the **377,280 × 768 bigram embedding**, which alone is
~290M parameters, far larger than the "124M" transformer. "GPT-2 small class" here refers to the
quality bar and the transformer trunk, not the total parameter count. If you need the number, compute
it; do not assume 124M.

---
# Part 6: GPU Kernels

## 6.1 What Triton is

Triton is a Python-embedded DSL and compiler for GPU kernels, created by Philippe Tillet (Harvard PhD
→ OpenAI). Academic origin: Tillet, Kung & Cox, *"Triton: An Intermediate Language and Compiler for
Tiled Neural Network Computations"*, MAPL 2019; OpenAI open-sourced the Python frontend in July 2021.

**The problem it solves.** CUDA C++ gives full control but forces you to hand-manage thread indexing,
shared-memory staging, bank conflicts, register blocking, async-copy pipelining, and tensor-core
fragment layouts. Getting within 20% of cuBLAS on a matmul in raw CUDA is a multi-week expert task.
Triton's bet: **the hard parts are mechanical once you fix a tiling strategy**, so you specify the
tiling and the compiler derives the rest.

**`@triton.jit` compiles nothing at decoration time.** It wraps the function in a `JITFunction` and
defers. Compilation triggers on first launch, keyed on: `tl.constexpr` argument *values*, tensor
dtypes, pointer alignment (Triton specializes on divisibility-by-16), integer args equal to 1, plus
`num_warps`/`num_stages`. Change a `constexpr` → fresh compilation. Results cache to `~/.triton/cache`.

**Pipeline:**

1. **Python AST**: Triton walks the AST (it does not trace by execution). Python control flow becomes
   IR control flow; `tl.constexpr` marks values that must be known at compile time so shapes and
   unrolling resolve statically.
2. **Triton IR**: MLIR since Triton 2.0: a `triton` dialect (tensor-of-pointers, `tt.dot`, `tt.load`)
   that is layout-agnostic, lowered to a `triton_gpu` dialect where every tensor gets an explicit
   **layout encoding** (blocked, shared, MMA/dot-operand). This is where the real work happens:
   layout assignment, coalescing analysis, software pipelining, shared-memory allocation and liveness.
3. **LLVM IR** with NVVM intrinsics.
4. **PTX**: *Parallel Thread eXecution*, NVIDIA's **virtual ISA**. Stable, forward-compatible,
   human-readable, with infinite virtual registers. A compiler target, not machine code.
5. **SASS**: *Streaming ASSembler*, the actual per-architecture binary. `ptxas` compiles PTX → SASS,
   doing real register allocation, instruction scheduling and dual-issue packing. SASS is
   architecture-specific (sm_90 SASS won't run on sm_100) and is not officially documented; inspect
   with `nvdisasm` or Nsight Compute. When the driver loads PTX for an unknown arch it JITs PTX→SASS
   at load time. That is the forward-compatibility mechanism.

**Practical consequence:** Triton owns steps 1–4 and delegates 5 to NVIDIA. **Occupancy surprises
usually come from `ptxas` register allocation, which Triton does not control.**

## 6.2 The programming model: block-level vs thread-level

This is the single most important conceptual difference.

**CUDA is SIMT / thread-level.** You write the program for *one thread*:
`int i = blockIdx.x*blockDim.x + threadIdx.x;` and operate on scalars. Parallelism inside a block is
your problem: you decide which thread loads which element, allocate `__shared__` tiles, call
`__syncthreads()`, write warp shuffles for reductions. This repo's one raw-CUDA kernel
(`triton_kernels.py:904+`) shows it: explicit `threadIdx.x`, `extern __shared__ __nv_bfloat16 smem[]`,
a `__shfl_down_sync(0xFFFFFFFF, thread_max, offset)` warp-reduction ladder, then `__syncthreads()` and
a cross-warp reduction. **~40 lines to compute one max and one sum.**

**Triton is block-level (tile-level).** You write the program for *one block* (Triton calls it a
"program"), and your variables are **tiles**, small dense arrays with static shapes like
`(BLOCK_M, BLOCK_N)`. There is no `threadIdx`. `tl.max(x, axis=0)` is one line and the compiler
generates the shuffle ladder. Shared memory is never declared by you: the compiler inserts it where a
layout conversion or a `tl.dot` operand requires it, and allocates/frees via liveness analysis.
Vectorization (emitting `ld.global.v4.b32` instead of four scalar loads) falls out of coalescing
analysis on your index expressions.

`tl.program_id(axis=0|1|2)` returns this instance's coordinate in the **grid**. `num_warps` is the one
place the thread model leaks: it tells the compiler how many warps (×32 threads) to spread each tile
across. It changes the layout encodings, not your code.

## 6.3 Core primitives you will meet in this repo

**`tl.load` / `tl.store` with masks.** Triton has no bounds checking. You build a tile of addresses by
broadcasting index arithmetic over pointers:

```python
a_ptrs = A_ptr + (offs_m[:, None]*a_stride_r + offs_k[None, :]*a_stride_c)
a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
```

`mask` is an elementwise boolean tile; false lanes are **not issued** (no fault, no traffic) and
receive `other`. **Masking is the entire answer to non-divisible sizes.** Tile shapes must be
compile-time constants (generally powers of two), so a 50304-column tensor with `BLOCK_N=128` gives
393 tiles where the last is partially masked. `other=0.0` is chosen to be the accumulation identity.
Note `.to(tl.int64)` on offsets in the transpose kernels: index arithmetic defaults to int32, which
overflows past 2³¹ elements.

**`tl.dot(a, b, acc)`** emits tensor-core MMA (`mma.sync` / `wgmma` on Hopper). Operands must be ≥16
in each dim. The accumulator is FP32 even for BF16 inputs. `input_precision` (deprecated name:
`allow_tf32`) controls **FP32 inputs only**: `"tf32"`, `"tf32x3"` (3-pass error-compensated, ~FP32
accuracy at ~1/3 speed), `"ieee"` (no tensor cores). Every `tl.dot` in `dc_triton_kernels.py` passes
`input_precision="tf32"` (e.g. `:126`).

**`tl.atomic_add(ptr, val, sem="relaxed", mask=...)`**: read-modify-write in global memory, used when
multiple programs write the same output tile (`dc_triton_kernels.py:543`, accumulating `dv` across
query blocks). `sem` maps to PTX `.relaxed`/`.acquire`/`.release`/`.acq_rel`. **`"relaxed"` guarantees
atomicity of this location only**, with no ordering constraints on surrounding accesses, so no memory
fences are emitted. Correct here because the only consumer reads after a kernel-wide barrier (the
launch boundary). It is also **nondeterministic**: FP addition isn't associative, so bit-exact
reproducibility is lost.

**`tl.exp2` and the log2(e) trick.** Hardware has `ex2.approx.f32`, a single SFU instruction.
`exp(x)` compiles to `exp2(x · log2 e)`, an extra multiply per element. In softmax you already
multiply scores by a scale, so you fold the constant in for free:

```python
score = tl.dot(q, tl.trans(k), input_precision="tf32") * (scaling * 1.4426950408889634)
p = tl.exp2(score - m[:, None])
```

`1.4426950408889634 = log₂(e)`. Across a full score matrix this removes one multiply per element at
zero cost. FlashAttention's CUDA source uses the same trick.

**`tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)`** reorders the mapping from linear
program ID to 2-D tile coordinate. Row-major order means co-resident programs span an entire output
row, touching `num_pid_n` distinct B-tiles, which is poor L2 reuse. Swizzling walks a
`GROUP_SIZE_M × num_pid_n` group in column-major order, so co-resident programs share operand tiles
and hit in L2 (50 MB on H100). Typical win: **10–30% on large matmuls.**

**`TensorDescriptor` / TMA**: see Part 3.4. In `linear_relu_square_kernel` the loads are
`a_desc.load([offs_am, offs_k])`: no pointer arithmetic, no mask, because TMA does out-of-bounds
handling in hardware.

**`tl.static_range(0, N)`** forces full compile-time unrolling (vs `tl.range`, which may emit a real
loop). Used at `dc_triton_kernels.py:37` for a fixed-iteration binary search: no loop counter, no
branches, and the scheduler can interleave all iterations.

**`tl.trans`, `tl.reshape`, `tl.permute`, `tl.split`**: layout changes, usually free (folded into the
shared-memory access pattern or MMA operand layout). The `linear_relu_square_kernel` epilogue
(`:451-453`) does `reshape(BM,2,BN/2) → permute(0,2,1) → tl.split` to peel a `(BM,BN)` accumulator
into two half-tiles matching the register layout, avoiding cross-lane data movement.

## 6.4 Tuning knobs

| Knob | What it controls | Tradeoff |
|---|---|---|
| `BLOCK_M`/`BLOCK_N`/`BLOCK_K` | output tile per program; K-slab per inner iteration | Larger M/N = better arithmetic intensity (`M·N·K` FLOPs on `(M+N)·K` bytes) but more accumulator registers and fewer programs to fill the GPU. `BLOCK_M=BLOCK_N=128` with FP32 accumulator = 16384 floats = 64 KB of registers, near the 64K-register-per-SM budget. Larger `BLOCK_K` = fewer loop iterations but a bigger shared-memory working set, which caps `num_stages`. |
| `num_warps` | warps per program; threads = `num_warps × 32` | More warps = fewer registers each, more epilogue parallelism, but smaller MMA fragments. Typical: 4 for small tiles, 8 for 128×128+. |
| `num_stages` | software-pipelining depth | The compiler restructures the K-loop so iteration *i*'s compute overlaps iterations *i+1..i+n−1*'s async copies. Shared-memory cost = `num_stages × (BLOCK_M+BLOCK_N) × BLOCK_K × sizeof(dtype)`. **Direct trade: shared memory (hence occupancy) for latency hiding.** Too high → allocation fails or occupancy collapses; too low → tensor cores stall on HBM. |
| `GROUP_SIZE_M` | swizzle group height | Bigger = better L2 reuse until the group's working set exceeds L2. `8` throughout this repo. |
| `@triton.autotune` | brute-force config search | On first call with a new `key`, Triton compiles *every* config, benchmarks each, caches the winner. 20 configs × seconds of compile = 1–3 min **per shape, per process**. |

**There is no `@triton.autotune` anywhere in this repo.** Every config is hardcoded, with comments
like `# Hardcoded configs based on H100 autotuning` (`triton_kernels.py:113`), then branching on
`if K == 768:`. The author ran the search offline on target hardware and froze the answer. **Rational
for a speedrun where compile time counts against the record; brittle if you change GPU.**

## 6.5 Why hand-written beats stock PyTorch: five mechanisms

**(a) Fusion / eliminating HBM round-trips.** Eager `y = relu(x)**2` launches kernels that read `x`
from HBM and write `y` to HBM. Recall Part 3.1: H100 needs ~295 FLOPs per byte to be compute-bound.
Any op below that is bandwidth-bound, and every avoided round-trip of an M×N BF16 intermediate saves
`2·M·N` bytes each way. Latency per HBM access is ~450–600 ns vs ~30 cycles for shared memory.

**(b) Exploiting mathematical structure.** `C = A @ Aᵀ` is symmetric. `XXT_kernel`
(`triton_kernels.py:34-98`) computes only the lower triangle via `if skip_block_below_diag or
skip_block_above_diag: return` (`:54`), then writes each result tile twice, to `C[m,n]` and
transposed to `C[n,m]` (`:91-98`). **~2× fewer FLOPs than a generic gemm**, and cuBLAS's `syrk` path
is typically not as well-tuned as `gemm`. This matters because Muon's Polar Express iteration is
nothing but `A @ Aᵀ` chains. Attribution comment at `:7`: *"Triton kernel for symmetric matrix
multiplication by @byronxu99."*

`XTX_kernel` (`:147-224`) is the tall-matrix variant, with the docstring explaining the choice:
*"For tall matrices (M > K), this is more efficient than transposing and using XXT because the
intermediate products are smaller (K x K vs M x M)."*

`ba_plus_cAA_kernel` (`:281-356`) fuses `C = α·A@Aᵀ + β·A`: the quintic polynomial step itself. Its
comment (`:294-295`): *"mostly duplicated from XXT_kernel, but also loads and adds a block of A …
Performance is slightly slower than XXT_kernel, so we use two separate kernels."*

**(c) Recompute vs store.** `linear_relu_square_kernel` (`:402-499`) computes `post = relu(pre)²`. The
backward needs `relu(pre)`. Standard autograd saves `pre`, an M×N tensor. Here the forward stores
**only** `post`, and the backward recovers `relu(pre) = sqrt(post)` in-register (`:457-459`,
`:473-475`). Comment: *"forward stores ONLY post = relu(pre)^2 (drop the redundant `pre`
materialization). Backward reconstructs relu(pre) = sqrt(post) in-kernel, so the full (M, N) pre
tensor never round-trips HBM."* One M×N tensor never touches HBM. Cost: one `sqrt` per element, a
few SFU cycles against a ~500 ns memory round-trip. **Trade recompute for bandwidth whenever the op
is cheap.**

**(d) Coalescing.** Unpack the comment at `triton_kernels.py:783-786`:

> Uses coalesced reads from src and coalesced writes to dst via `tl.trans()`. Replaces PyTorch's
> elementwise `copy_` which uses a naive 75k-block kernel with non-coalesced writes, saturating all
> SMs and blocking NCCL.

- **"naive 75k-block"**: PyTorch's generic `TensorIterator` elementwise kernel launches one block per
  fixed chunk of elements; for a large tensor that's ~75,000 blocks each doing a trivial per-element
  copy with strided index math.
- **"non-coalesced writes"**: in a transpose, consecutive source elements land in destinations
  separated by a full row stride, so each thread triggers its own 32-byte sector fetch: **up to 32×
  the memory traffic** for the same logical bytes.
- **"saturating all SMs and blocking NCCL"**: the mechanism from Part 3.5.

**(e) FP8 gradients.** Writing the CE gradient directly as FP8-E5M2 halves the bytes for an
`n_rows × 50304` tensor.

## 6.6 Kernel inventory: `triton_kernels.py`

Imported at `train_gpt.py:42`.

| Kernel (line) | Computes | Called from | Notable technique |
|---|---|---|---|
| `_pid_to_block` (9) | linear pid → `(batch, m, n)` tile coord | helper | `tl.swizzle2d` grouped ordering for L2 |
| `XXT_kernel` (34) | `C = A@Aᵀ`, symmetric | `polar_express` `:244` | diagonal-skip + mirror store; fp32 accum |
| `XTX_kernel` (147) | `C = Aᵀ@A` (tall) | `polar_express` `:216` | smaller K×K intermediate |
| `ba_plus_cAA_kernel` (281) | `αA@Aᵀ + βA` | `polar_express` `:217,245` | fused GEMM + scaled-add (the quintic step) |
| `linear_relu_square_kernel` (402) | `relu(x@W1ᵀ)²` fwd+bwd | `ReLUSqrdMLP` `:48` | **TMA TensorDescriptor**, persistent NUM_SMS grid, FP8 emit, `sqrt(post)` reconstruction |
| `reduce_mlp_activation_scales_kernel` (502) | per-layer amax → FP8 scale | `:1671` | delayed-scaling calibration |
| `quantize_transpose_mlp_down_weights_kernel` (522) | fused FP8 quantize + transpose | `:1428` | one pass instead of two |
| `_transpose_copy_kernel` (788) | `dst = srcᵀ` | `:810, 1278, 1281` | coalesced R/W; avoids NCCL-blocking |
| `_transpose_add_kernel` (850) | `dst += srcᵀ` | `:758` | fused transpose-add |
| `ce_fwd_bwd_kernel` (914, **raw CUDA C**) | softcap + softmax + CE, grad in FP8 | `FusedSoftcappedCrossEntropy` `:1687` | inline PTX fp8 convert, warp-shuffle softmax, MTP + prefix loss |

**Configs**: `XXT`/`XTX` use `BM,BN,BK = 128,128,64` when `K==768`, else `64,128,128`; both
`num_stages=4, num_warps=8`. `linear_relu_square` uses `BM=128, BN=256, BK=128` (fp8) or `64` (bf16),
`num_stages` 4 fwd / 3 bwd, `num_warps=8`, and a **persistent grid capped at `NUM_SMS`** with
`tl.range(..., NUM_SMS, flatten=True)`, a stream-K-style persistent kernel.

**There are no reference implementations or unit tests in the file.** The only "references" are the
plain-PyTorch equivalents named in comments. If you need correctness evidence, it comes from the
records' multi-seed logs, not from this file.

## 6.7 The raw CUDA cross-entropy kernel

`ce_fwd_bwd_kernel` (`triton_kernels.py:914-1171`) is **not Triton**. It is a hand-written CUDA C
string compiled at runtime:

```python
ce_fwd_bwd_kernel = torch.cuda._compile_kernel(
    CE_KERNEL_DECLS + CE_KERNEL_SOURCE, "ce_fwd_bwd_kernel",
    compute_capability="90",
    nvcc_options=["-lineinfo", "--use_fast_math"])
ce_fwd_bwd_kernel.set_shared_memory_config(CE_KERNEL_VOCAB_SIZE * 2)   # ~98 KB
```

**What it computes.** One CTA per row, `BLOCK_SIZE=256` threads. Applies the sigmoid-based logit
softcap (`:1004`, using fast `__tanhf`), runs an **online softmax** (block-max via
`__shfl_down_sync` warp reductions `:1018-1030`, then log-sum-exp `:1064`), computes MTP +
prefix-token losses (`:1071-1090`), and writes the gradient **directly as FP8-E5M2** via inline PTX
(`:923-933`, `:1118-1121`), with a sparse target-column correction (`:1131-1168`). Vectorized 8-wide
bf16 loads (`__nv_bfloat168`, `:999`). Logits themselves come from `torch._scaled_mm` in FP8-E4M3
(`:1217-1224`); backward `grad_x`/`grad_w` also via `_scaled_mm` (`:1268-1290`) using
`transpose_copy` for the fp8 transposes.

**Why:** it fuses the entire logits→loss→grad chain, **never materializing the `(rows, 50304)` softmax
probabilities in fp32/bf16 HBM**; it emits the gradient straight to fp8.

**The runtime-compilation machinery.** `torch.cuda._compile_kernel` is a **private** PyTorch API (no
stability guarantee) using **NVRTC**, NVIDIA Runtime Compilation, a shared library that compiles
CUDA C++ to PTX **in-process**: no `nvcc` subprocess, no temp files, no full toolchain required.
PyTorch then loads the PTX via the driver API (`cuModuleLoadData` + `cuModuleGetFunction`). NVRTC
emits PTX; the driver JITs it to SASS at load.

**`compute_capability="90"`** targets Hopper. The SM map:

| SM | Generation | Representative GPU |
|---|---|---|
| sm_70 | Volta | V100 |
| sm_75 | Turing | T4, RTX 2080 |
| sm_80 | Ampere | A100 |
| sm_86 / sm_89 | Ampere consumer / Ada | RTX 3090 / 4090, L40S |
| **sm_90 (sm_90a)** | **Hopper** | **H100, H200** |
| sm_100 | Blackwell datacenter | B100/B200 |
| sm_120 | Blackwell consumer | RTX 50xx |

The trailing `a` in `sm_90a` means "architecture-specific": enables `wgmma` and TMA, **not**
forward-compatible to later archs.

**`set_shared_memory_config(50304 * 2)`** requests ~98 KB dynamic shared memory per block (one full
BF16 vocabulary row of logits). CUDA defaults to a 48 KB static limit; above that requires an explicit
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` opt-in. H100 has ~227 KB usable per block, so
98 KB permits **2 blocks/SM**, matching `__launch_bounds__(BLOCK_SIZE, 2)`, which tells `ptxas` to
cap registers accordingly.

**`--use_fast_math`** trades IEEE conformance for speed. It implies `--ftz=true` (flush denormals to
zero), `--prec-div=false` (approximate reciprocal division, ~2 ulp instead of correctly-rounded),
`--prec-sqrt=false`, `--fmad=true`, and remaps `expf`/`logf`/`sinf` to the `__expf`/`__logf`/`__sinf`
hardware intrinsics (~2 ulp, single SFU instruction, vs ~10× more instructions for accurate
versions). **The risk it accepts:** denormal flushing can turn tiny gradients into exact zeros.

**Inline PTX**, decoded:

```cpp
asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(x), "f"(0.0f));
```

- `cvt` = convert
- `.rn` = round-to-nearest-even
- `.satfinite` = **saturate to the largest finite value** instead of producing infinity on overflow,
  which is critical for FP8, whose max is small
- `.e5m2x2` = destination is **two packed** FP8-E5M2 values
- `.f32` = FP32 sources
- `"=h"` = 16-bit output register; `"f"` = 32-bit float input
- `volatile` prevents reordering or elision

It exists because no portable C++ intrinsic exposes this exact rounding+saturation combination, and
because it is **one instruction** instead of a branchy software conversion.

## 6.8 `dc_triton_kernels.py`: the DCFormer correction

**"dc" = Dynamically Composable**, confirmed by three sources: the project README's feature list
("Lightweight Dynamically Composable MHA"), the introducing record
`records/track_1_short/2026-05-27-MuddGatedAndDC/README.md`, and the file docstring
(`dc_triton_kernels.py:2`): *"Triton implementation for the best-config post-only no-DD DC
correction."*

**Zero function-name overlap with `triton_kernels.py`**, completely disjoint namespaces. Roles:
`triton_kernels.py` = optimizer + FP8 matmul + transpose + loss infrastructure;
`dc_triton_kernels.py` = **one feature**, the DC attention correction and its hand-written backward.

11 `@triton.jit` functions. Module constants are hand-set (`:16-21`): `BLOCK_K=128`, `PRE_WARPS=8`,
`QK_WARPS=4`, `PRE_STAGES=1`, `QK_STAGES=2`. **No `@triton.autotune`.**

| Kernel (line) | Computes | Notable |
|---|---|---|
| `_dc_postonly_doc_bounds` (24) | binary-search doc `[start,end)` per token | `tl.static_range` log-search |
| `_dc_postonly_build_doc_bounds_table_kernel` (48) | precompute `int32[T]` doc-bound tables | amortizes the search out of the attention loop |
| `_dc_postonly_probs_wsmall_head` (90) | per-head windowed softmax + LSE | stable base-2 softmax via `exp2` |
| `..._no_lse` (144) | same, no LSE | inference path |
| `_dc_postonly_store_corr_wsmall_head` (196) | `post_w2·a_acc @ V + BASE` | **fused residual add** (`ADD_BASE`) |
| `_dc_postonly_corr_fwd_wsmall_cached_kernel` (252) | **main forward**, 6 heads manually unrolled | window ≤128 ⇒ **a single 128-wide key tile covers the whole window**: no inner K loop. That is what "wsmall" encodes. |
| `_dc_postonly_probs_loop_head` (442) | recompute probs from cached LSE | skips the max/renormalize pass |
| `..._bwd_pre_wsmall_dm_head` (489) | `dV`, `gpost_w2`, `da_acc` | relaxed `atomic_add` on dV |
| `..._bwd_pre_wsmall_soft_head` (558) | softmax-Jacobian pre-reduction | |
| `_dc_postonly_corr_bwd_pre_wsmall_kernel` (594) | **backward pass 1** | materializes `da_buf`/`soft_dot` to HBM for pass 2 |
| `_dc_postonly_corr_bwd_qk_wsmall_kernel` (781) | **backward pass 2**: `dQ` (store), `dK` (atomic) | 2-D grid `(T-blocks, H)` |

Hard shape guard (`_is_supported_postonly_triton`, `:898`): `B==1`, `H==6`, `D==128`,
dtype ∈ {fp16, bf16}. Public entry `dc_attention_postonly_nodd_correction_add_base_triton` (`:1266`,
default `window=112`), called at `train_gpt.py:1144` when `dc_w is not None` (layer 10, non-paired).

**Notable contrast with `triton_kernels.py`:** this file uses **no TMA, no wgmma intrinsics, no warp
specialization, no clusters, no async copy**: plain masked `tl.load`/`tl.store`, tf32 `tl.dot`, and
relaxed atomics. Any wgmma mapping is left implicit to the Triton compiler.

**One flag:** the docstring cites `machine2_exps/train_mudd.py`, which does not exist in this repo;
the sole importer is `train_gpt.py:43-44`. Leftover path reference from the kernel's origin repo.

## 6.9 Numerical formats: the reference table

| Format | S/E/M | Total | Bias | Max finite | Min normal | Rel. precision | Use |
|---|---|---|---|---|---|---|---|
| FP32 | 1/8/23 | 32 | 127 | ~3.40e38 | ~1.18e-38 | ~6e-8 (~7 digits) | Master weights, optimizer state, accumulators |
| TF32 | 1/8/10 | 19 (in 32-bit reg) | 127 | ~3.40e38 | ~1.18e-38 | ~4.9e-4 (~3 digits) | Tensor-core input for FP32 matmul |
| **BF16** | 1/8/7 | 16 | 127 | ~3.39e38 | ~1.18e-38 | ~3.9e-3 | **Default training dtype**: FP32 range, low precision |
| FP16 | 1/5/10 | 16 | 15 | 65504 | ~6.10e-5 | ~4.9e-4 | Inference, legacy AMP (needs loss scaling) |
| **FP8-E4M3** | 1/4/3 | 8 | 7 | **448** | ~1.95e-3 | ~6.3e-2 | **Forward activations & weights** |
| **FP8-E5M2** | 1/5/2 | 8 | 15 | **57344** | ~6.10e-5 | ~0.125 | **Gradients** |
| FP4 (E2M1) | 1/2/1 | 4 | 1 | 6.0 | 1.0 | ~0.25 | Blackwell MXFP4 inference; block-scaled |

See Part 3.4 for the derivation of 448 and the E4M3/E5M2 forward/gradient convention. The literal
`448.0` appears throughout this codebase, in `tl.minimum(c0_post * inverse_activation_scale, 448.0)`
(`triton_kernels.py:464`) and `(448.0 / amax)` (`train_gpt.py:1646`), because **FP8 requires explicit
scaling**: compute the tensor's amax, scale so amax maps to (just under) 448, cast, and carry the
reciprocal scale to dequantization. Without this, E4M3's ~1 significant decimal digit and 448 ceiling
would destroy the tensor.

---

# Part 7: Distributed Training and the PyTorch Runtime

> Part 3.5 covered the collectives at the hardware level; this part
> covers how PyTorch exposes them and every runtime knob this repo touches.

---

## 7.1 `torchrun --standalone --nproc_per_node=8`

`run.sh` is one line. `torchrun` is a console script for `torch.distributed.run`, the elastic
launcher. It forks 8 child processes, each a **fresh Python interpreter** (no threads, no GIL
contention), and injects environment variables so each can identify itself:

- `RANK`: global index 0–7
- `LOCAL_RANK`: index within this node
- `WORLD_SIZE` / `LOCAL_WORLD_SIZE`: 8
- `MASTER_ADDR` / `MASTER_PORT`: where the rank-0 TCPStore lives
- plus `GROUP_RANK`, `ROLE_RANK`, `TORCHELASTIC_RUN_ID`, `TORCHELASTIC_RESTART_COUNT`

The repo reads exactly these at `train_gpt.py:54-60`, then `torch.cuda.set_device(device)` so every
subsequent allocation lands on the right GPU.

**Rendezvous** is the barrier where the N processes discover each other and agree on ranks, world
size, and a shared key/value store. `--standalone` means "single node, `--nnodes=1`,
`--rdzv-backend=c10d`, endpoint localhost on a **randomly chosen free port**." The random port is the
point: you can run several standalone jobs on one box without collisions. No etcd, no external
coordinator, and no elasticity in practice; if a worker dies, the job dies. The launcher reaps the
whole worker group on failure and reports the first-by-timestamp exception as root cause.

**One oddity worth knowing** (`train_gpt.py:31-33`): a throwaway `torch.empty(...).backward()` on the
local CUDA device runs *before* importing dynamo/dist, with a comment that it "prevents a bug on some
systems."

## 7.2 Process groups and the backend string

```python
dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)   # train_gpt.py:62
```

The multi-backend string format is documented PyTorch. Why it is needed: **NCCL does not support CPU
tensors at all**, and Gloo is slow-to-nonexistent for GPU collectives. Registering both means a
collective on a CUDA tensor dispatches to NCCL and one on a CPU tensor dispatches to Gloo,
transparently. (This repo needs the CPU path for the sparse-comms count tensors.) MPI is a third
option only if you build PyTorch from source against an MPI implementation; nobody does for
single-node GPU work.

**`device_id=` is a real optimization, not decoration.** It makes NCCL build the communicator eagerly
(`ncclCommInit*` at init rather than lazily at first collective) and lets sub-groups use
`ncclCommSplit`. It moves hundreds of milliseconds of NCCL handshake **out of the timed region** and
surfaces NCCL init errors immediately. `dist.barrier()` at `:63` then guarantees the communicator is
warm before anything is measured.

## 7.3 Data parallelism, and what this repo is *not* doing

Data-parallel replicates the entire model on every device. Each rank pulls a different slice of the
global batch, runs an independent forward/backward, and produces gradients correct for *its* slice
only. Before the optimizer step the ranks must agree: the mathematically-required gradient is the
mean over the global batch, so ranks exchange and average. Then each applies the same update to its
own copy. Because copies started identical and receive identical updates, they stay bit-identical
(modulo 7.10). The only communication is O(model size) per step, **independent of batch size**,
which is why DP is the default when the model fits in one GPU.

Contrast, one paragraph each, so you know what this repo is *not*:

- **Tensor parallelism** splits individual layers: an `(in,out)` weight is column-sharded so each rank
  computes a slice of the output, then an all-reduce stitches activations back. Communication happens
  *inside* every layer, twice per block, on activation-sized tensors. For models whose single layer is
  too big. Demands very fast interconnect.
- **Pipeline parallelism** assigns different *layers* to different devices, streaming micro-batches
  through stages. Small communication (one activation per stage boundary) but you pay a "bubble"
  where stages idle. Good for many-node setups with slow links; useless for 8 GPUs on one NVSwitch
  board.
- **FSDP / ZeRO-3** keeps DP's structure but shards parameters, gradients, *and* optimizer state.
  Before each layer executes it all-gathers that layer's parameters, computes, then discards them.
  Memory drops ~N×; communication rises because parameters are gathered twice per step.

This repo does DP, plus a hand-rolled ZeRO-1 (sharded optimizer state) per parameter.

## 7.4 The four per-parameter comms policies

`_launch_reduce`, `train_gpt.py:588-628`. Each parameter's `ParamConfig` names one of:

| Policy | Mechanism | Used for |
|---|---|---|
| `none` | no communication | params that need none |
| **`replicated`** (`:597-599`) | `dist.all_reduce(grad, AVG)`: every rank computes the full update | small params where sharding overhead exceeds the win |
| **`sharded`** (`:600-619`) | `reduce_scatter_tensor(AVG)` → each rank owns rows `[rank·chunk : (rank+1)·chunk]`, updates only its shard, then `all_gather_into_tensor` (`_launch_gather`, `:630-642`) rebuilds the full param | the big weight banks. **This is ZeRO-1.** |
| **`sharded_sparse`** (`:620-628`, `:258-346`) | `all_to_all_single` exchanging **only the gradient rows each rank actually touched** | the 377,280-row bigram embedding |

**Why the banks exist.** NorMuon reshapes parameters into "banks" whose leading dimension is divisible
by `world_size` (`:504-508`) so that **whole matrices stay on one rank**; Polar Express needs the
complete matrix to orthogonalize it (Part 4.11).

**The sparse path is conditional.** It is gated on `world_size == 8 and grad_accum_steps == 1`
(`:259-261`) and falls back to plain `sharded` otherwise (`:479-480`). **If you run on 4 GPUs, you
silently get a different (slower) communication path.** Know this before you benchmark.

**Overlap.** Async futures everywhere (`async_op=True` … `.get_future()`). `scatter_order` launches
reductions early; `work_order` processes small params first so large bf16 reductions finish
underneath (`:2052-2063`). **No backward hooks**: the schedule is explicit and hand-ordered.

**`async_op=True` semantics, precisely.** It returns a `Work` handle immediately; `.get_future()`
converts it to a `torch.futures.Future`. The NCCL kernel runs on a separate CUDA stream so the GPU
can execute compute kernels concurrently with the transfer. **"Waiting" does not mean "block the CPU
until bytes arrive"**; it means "insert a stream dependency so subsequent kernels on the compute
stream don't read the buffer before the collective's stream has written it." The CPU usually returns
instantly.

## 7.5 Why not DDP, and the graph-guard trick

`torch.nn.parallel.DistributedDataParallel` broadcasts rank 0's parameters at construction, then
registers autograd hooks on each parameter's gradient accumulator. Parameters are assigned to
**buckets** (~25 MB) in approximately reverse-construction order; when every gradient in a bucket has
arrived, DDP fires an async `all_reduce` for the whole bucket. So DDP already gets bucketing and
overlap for free, plus `no_sync()` and `find_unused_parameters`.

**What hand-rolling buys:**

- **Choice of collective.** DDP does `all_reduce`, full stop. It cannot do reduce_scatter + sharded
  optimizer + all_gather; that's FSDP, a different wrapper with different overheads. Record #24 is
  literally "replace all_reduce with reduce_scatter."
- **Per-parameter policy.** Four policies coexist, including the sparse `all_to_all_single` path. No
  wrapper expresses that.
- **Bucketing you control.** DDP's bucket boundaries are heuristic and dtype-homogeneous. Record #36
  fuses the MLP and attention reduce-scatters into a shared call because the author knows the model's
  structure.
- **No module wrapper.** DDP wraps `forward`, adds `.module` indirection, and interacts awkwardly with
  `torch.compile(fullgraph=True)`: the reducer's hooks and the `_DDPSink` autograd node have
  historically caused graph breaks. Raw collectives on raw tensors don't.
- **Reduction dtype and op.** Reduce in the gradient's native dtype with `ReduceOp.AVG` directly.

## 7.6 torch.compile / TorchInductor

**The pipeline.** (a) **TorchDynamo** hooks CPython's frame-evaluation API, symbolically executes
bytecode, and extracts an FX graph plus a set of **guards**, runtime predicates that must hold for
the compiled artifact to be valid. Anything Dynamo can't trace (a `print`, a data-dependent
`if tensor.item() > 0`, an unsupported call) causes a **graph break**: Dynamo ends the graph, runs the
offending code eagerly, and starts a new graph after it. (b) **AOTAutograd** traces the forward *and*
generates the backward ahead of time, producing a joint representation, then functionalizes and
decomposes into a smaller ATen set. (c) **TorchInductor** lowers that into a scheduling IR, does loop
fusion and buffer reuse, and emits **Triton** for GPU.

**So Inductor is a competitor to the hand-written kernels in Part 6, playing the same game.** It
excels at fusing elementwise/reduction chains; it is weaker at structured matmul variants (it mostly
falls back to cuBLAS or templated Triton matmuls), exactly the gap `XXT`/`XTX`/`linear_relu_square`
fill.

**`fullgraph=True`** turns graph breaks from a warning into a hard error. Two reasons this matters.
First, correctness of expectation: a graph break silently costs the fusion across the break: every
intermediate must be materialized to global memory instead of staying in registers. `polar_express`'s
own docstring says exactly this: it fuses Nesterov momentum and orthogonalization "avoiding
materialization of the FP32 intermediate between graph breaks." Second, in a speedrun you want the
build to fail loudly when a refactor accidentally introduces a break, rather than losing 3% silently.

**`dynamic=False`** forces static-shape specialization. By default Dynamo compiles the first call with
static shapes, and if it later sees a different shape it recompiles with **symbolic** shapes so one
artifact handles all sizes. Symbolic shapes cost you: Inductor cannot constant-fold loop bounds,
cannot pick block sizes tuned to exact dimensions, loses divisibility guarantees (so masks can't be
elided), and must emit guard arithmetic. `dynamic=False` says "never generalize; recompile per shape."
The comment at `train_gpt.py:176` (*"Must use dynamic=False or else it's much slower"*) is exactly
this.

**But this repo deliberately varies batch size and sequence length on a schedule**, so it raises
`dynamo.config.recompile_limit = 64` (`train_gpt.py:50`; formerly `cache_size_limit`, default 8).
**Exceeding the limit makes Dynamo give up and fall back to eager for that frame, a catastrophic and
easy-to-miss performance cliff.** If you add a new schedule stage, check this number.

### The 0-D CPU tensor trick (understand this before you touch hyperparameters)

Guards decide whether a cached compilation can be reused.

- For a **tensor** argument, guards check *metadata*: dtype, device, shape (or symbolic constraints),
  stride/layout, `requires_grad`. **They do not check values.**
- For a **Python float or int**, Dynamo by default treats the value as a compile-time constant, bakes
  it into the graph (so Inductor can constant-fold it into the generated Triton source), and installs
  a guard `arg == 0.0031`. **Change the learning rate next step and the guard fails → recompile.**

A schedule over 1285 steps would therefore trigger ~1285 recompiles, blow past `recompile_limit`, and
drop you to eager.

**The fix:** make the hyperparameter a tensor so the guard checks metadata rather than value. This
repo keeps LR, weight decay, step size, and momentum as **0-D CPU float32 tensors**
(`train_gpt.py:459-463`), documented in-line: *"0-D CPU tensors to avoid recompilation"* and
*"`momentum_t` is a 0-D CPU tensor to avoid triggering graph recompilations when the value changes"*
(`:187`). Each step does an in-place `.copy_()` of the new scalar. Shape `()`, dtype float32, device
cpu never change, so the guards hold and the compiled kernel is reused for all 1285 steps. **The
scalar becomes a runtime kernel argument instead of an immediate.** CPU rather than CUDA so updating
it needs no device sync or H2D copy in the hot loop.

**`coordinate_descent_tuning`.** Inductor normally picks Triton configs from a small candidate list.
This flag adds a greedy coordinate-descent search: hold all knobs fixed but one, sweep it, benchmark
on real hardware, keep the winner, move to the next, repeat. Typically a few percent on reduction and
pointwise kernels. It also benchmarks every candidate for every kernel in the model. `train_gpt.py:38`
has it commented out with the reason: *"we have banned this flag for new records because it causes
compilation to take 30min."* The README quantifies the trade: ~25 min of untimed pre-run compilation
to save ~3 s of a ~75 s run. Still allowed on the Medium track, where the trade flips.

## 7.7 Custom ops

```python
@torch.library.custom_op("nanogpt::mm_t", mutates_args=())   # train_gpt.py:70
```

**Why not just call the function?** Because `torch.compile` traces *through* plain Python. Dynamo
would inline the body, AOTAutograd would functionalize and decompose the FP8 casts and the
`.T.contiguous().T` layout gymnastics, and Inductor would be free to rewrite them, which either
breaks the strict layout contract `torch._scaled_mm` requires, or produces a graph break. Registering
an op makes it a **single opaque node** in the FX graph with a stable schema. The compiler treats it
as a black box: it will not look inside, will not reorder around it in ways the schema forbids, and
knows from `mutates_args=()` that it has no side effects (so it can be DCE'd or CSE'd).

But a black box breaks tracing, because tracing runs on **FakeTensors**: metadata, no storage.
Calling the real kernel on a FakeTensor is impossible. Hence `@mm_t_op.register_fake` (`:97-103`): a
"meta" implementation taking fake inputs and returning fake outputs with **correct shapes, dtypes and
devices**, doing no real computation. It also asserts the layout preconditions, so a violation fails
at *compile* time rather than producing garbage. Finally
`mm_t_op.register_autograd(backward_t, setup_context=setup_context_t)` (`:162`) tells autograd how to
differentiate the opaque node, by calling a *second* custom op, `nanogpt::mm_t_backward`, which
reuses the already-quantized FP8 activations saved in the forward, avoiding re-quantization.

Note the pattern inside: the custom op's body itself calls `@torch.compile`'d inner functions
(`:73`, `:107`). **The code is still Inductor-optimized, but in an isolated compilation unit whose
boundaries the author controls.**

## 7.8 FP8 in PyTorch

**`torch._scaled_mm`** (used at `train_gpt.py:85`). Signature:
`torch._scaled_mm(a, b, scale_a=, scale_b=, out_dtype=, use_fast_accum=)`. Semantically
`out = (scale_a * a) @ (scale_b * b)`, accumulating in FP32 inside the tensor cores and casting to
`out_dtype` (bf16 or fp32 here). **The scales are FP32 tensors, not Python floats.**

Why scales are mandatory: FP8 has ~4 decimal digits of *range*, not enough for a typical activation
tensor's distribution. You divide by a per-tensor scale so values land in the representable band,
cast, then multiply the scale back into the product. Because matmul is linear,
`(x/s_x) @ (w/s_w) · s_x·s_w = x @ w` exactly, up to quantization error. Per-**row** (or per-block)
scaling gives each row of A and column of B its own scale: much more robust with outlier channels,
at the cost of a more complex kernel. **This repo uses per-tensor.**

**Hardware constraints leak into the API.** On H100, `b` must be **column-major**, and M/N/K must be
multiples of 16. Hence `w_f8.T.contiguous().T` (`:83`), a round trip that keeps the logical
`(in, out)` shape while forcing column-major storage.

**`use_fast_accum=True`** lets the tensor core accumulate in reduced precision inside the MMA
pipeline: faster, slightly less accurate. Note the repo enables it in the forward (`:91`) and
**disables it in both backward matmuls** (`:123`, `:134`); gradients are more sensitive.

**The three amax strategies, all three of which appear in this repo:**

| Strategy | Mechanism | Tradeoff | Where used here |
|---|---|---|---|
| **Dynamic** (= "current scaling") | compute the tensor's amax immediately before casting, every time | exact, never clips; but a full extra read + reduction, and a data dependency that fragments fusion | MLP **up**-projection input: `amax = normed.detach().abs().max().clamp(min=1e-12)`, then `(normed * (448.0/amax)).to(float8_e4m3fn)` (`:1645-1647`) |
| **Delayed** | derive this step's scale from a rolling history of previous steps' amax (× headroom) | no extra pass, no dependency stall, kernels fuse cleanly; but the scale is stale so a sudden spike clips | MLP **down**-projection: layers accumulate `_mlp_down_partial_amax` during forward; `reduce_mlp_activation_scales(...)` (`:1671-1675`) turns this step's amax into *next* step's scales. **This is record #89.** |
| **Static** | hand-tuned fixed constants, never updated | zero runtime cost, maximum fragility | LM head: `x_s=100/448, w_s=1.6/448, grad_s=grad_scale*0.75/448` (`:1225`); each is `expected_amax / 448`, measured once and baked in. **Record #51 is "retune fp8 scales", which is what maintaining this costs you.** |

**Terminology correction worth carrying:** in NVIDIA Transformer Engine and torchao,
**"current scaling" and "dynamic scaling" are the same thing** (`Float8CurrentScaling`). The genuine
third option is static/constant scaling. If you see a source list "delayed / dynamic / current" as
three strategies, it is wrong.

**"amax"** is `t.abs().max()`, the input to every strategy, because the scale must satisfy
`amax / scale ≤ FORMAT_MAX` or you clip. The **headroom factor** is a fudge multiplier > 1 applied
before computing the scale, reserving room for the next step's tensor being larger. This repo passes
`headroom=1.80` for MLP down-projection activations (`:1674`) and `1.12`/`1.25` defaults elsewhere.
**Too little headroom clips outliers; too much wastes precision by pushing everything into the low
bits.**

FP8 is togglable via env `DISABLE_FP8` (`:1224`, `:1498`), useful when debugging on non-Hopper
hardware.

## 7.9 Memory and data movement

**Pinned memory.** Normal host allocations are *pageable*: the OS may relocate or swap the physical
pages at any time. The GPU's DMA engine addresses physical memory and cannot tolerate that, so a copy
from pageable memory forces the driver to first stage into an internal pinned bounce buffer with a CPU
memcpy, which is serialized and blocks. `pin_memory=True` calls `cudaHostAlloc`, page-locking the
allocation so its physical address is fixed. Now the DMA engine reads it directly, without CPU
involvement, on a copy engine that runs concurrently with the SMs.

**`tensor.to(device, non_blocking=True)` is only genuinely asynchronous if the source is pinned.** On
pageable memory PyTorch honours the flag syntactically but the copy is effectively synchronous. The
second requirement: you must not mutate the source buffer until the copy completes, which is why
async loaders use a *ring* of pinned staging buffers, not one.

**CUDA streams and timing.** Almost every PyTorch GPU call is asynchronous: it enqueues a kernel and
returns. So `t0=perf_counter(); model(x); t1=perf_counter()` measures **launch overhead**, often 100×
too small. `torch.cuda.synchronize()` blocks the CPU until all work on all streams of the device has
completed. The repo brackets every timing boundary with it (`:2311`, `:2327`, `:2344`), **the only
correct way to produce the numbers the leaderboard is built on.** Operations that read GPU data into
Python (`.item()`, `float(t)`, `print(t)`, `if t > 0`) synchronize implicitly, which is a hidden
stall; the 0-D-CPU-tensor trick partly exists to keep such reads out of the loop.

**The caching allocator.** `cudaMalloc`/`cudaFree` are expensive and device-synchronizing, so PyTorch
never calls them per tensor. It requests large **segments** and sub-allocates, keeping freed blocks in
size-bucketed pools. **Fragmentation** is when total free bytes are ample but no single free block is
contiguous enough: you OOM with `reserved_memory` far above `allocated_memory`. This is precisely the
failure mode for a workload whose tensor shapes change over time, **and this repo has both a
batch-size schedule and a max-sequence-length schedule** (records #46, #72).

`PYTORCH_ALLOC_CONF=expandable_segments:True` (`train_gpt.py:25`, set **before** `import torch`;
it's read at allocator init) switches to CUDA's virtual-memory-management API. Instead of fixed-size
physical segments, it reserves a large *virtual* address range per pool and maps physical pages in on
demand, growing and shrinking. A segment can expand to serve a larger request rather than forcing a
new segment plus a stranded old one. Fragmentation from varying shapes largely disappears. (Older
releases named it `PYTORCH_CUDA_ALLOC_CONF`.)

**Layout, contiguity, and `.T`.** A tensor is storage + shape + strides. `.T` allocates nothing, a
view with swapped strides. That is free, but **downstream kernels are not layout-agnostic**.
cuBLAS/cuBLASLt have separate kernels per transposed-operand combination and they are not equally
fast; `torch._scaled_mm` on Hopper flatly *requires* column-major B. When a kernel meets a layout it
can't use, PyTorch inserts a materializing copy (a transpose kernel that is pure memory traffic, zero
FLOPs) or falls back to a slower generic path.

Hence `.T.contiguous().T` (`:83`, `:130`): the middle `.contiguous()` performs the transpose copy
**once, deliberately and visibly**, and the outer `.T` restores the logical shape over now-column-major
storage. Hence also records #74/#76 (dedicated Triton `transpose_copy`/`transpose_add`, then reusing
them) and record #27 ("transpose one of the MLP matrices"), which lays weights out as `(in, out)`
instead of `(out, in)` so gradient accumulation and the FP8 matmul both hit fast paths without a copy.

## 7.10 Precision knobs, and which ones this repo deliberately skips

- **bfloat16**: 1/8/7. Identical exponent field to FP32 ⇒ identical dynamic range. Conversion from
  FP32 is a truncation of the low 16 bits.
- **float16**: 1/5/10. More mantissa but max 65504, smallest normal ~6.1e-5. Transformer gradients
  routinely live below 1e-5, so they flush to zero. The workaround is **loss scaling**: multiply the
  loss by a large constant before backward, divide it out of the gradients before the step, with
  `GradScaler` adjusting dynamically and skipping inf/NaN steps. **bf16 needs none of this**, which is
  why record #10 ("bfloat16 activations") was a clean win.
- **TF32**: a tensor-core *internal* format, 1/8/10, 19 bits. `torch.backends.cuda.matmul.allow_tf32 =
  True` (or `torch.set_float32_matmul_precision("high")`) makes FP32 matmuls silently round inputs to
  TF32 and accumulate in FP32. `torch.backends.cudnn.allow_tf32` is a separate switch (defaults True);
  the matmul one has defaulted **False** since PyTorch 1.12.
- **Autocast/AMP** maintains a per-op dtype policy (matmuls and convs in bf16/fp16,
  softmax/normalization/reductions in FP32), inserting casts automatically.

**This repo skips all of it.** No `autocast` context, no `GradScaler`. Every cast is explicit and
hand-placed: bf16 activations and attention/MLP weights (`:2242-2254`), FP8 for the LM head and MLP
projections, FP32 kept only where needed (Nesterov momentum in `polar_express`, `grad_w` accumulation
at `:131`, Adam moments). `allow_tf32` is never touched because **there are essentially no FP32
matmuls left to accelerate**. Loss scaling is unnecessary in bf16, and the FP8 `x_s`/`w_s`/`grad_s`
scales are a per-tensor generalization of the same idea, tuned by hand rather than adapted by a
scaler.

Manual casting is more work and more fragile, but **it is the only way to get FP8 into specific
matmuls while keeping a normalization in FP32 two lines away**; autocast's per-op policy has no
vocabulary for that.

## 7.11 Determinism, and why this benchmark is honest about it

Floating-point addition is not associative: `(a+b)+c ≠ a+(b+c)`. Every source of nondeterminism here
reduces to that plus a nondeterministic ordering.

- **Collectives.** NCCL chooses its algorithm (ring, tree, NVLS) and chunk sizes at runtime from
  topology, message size, and tuning heuristics. Different chunking = different reduction tree =
  different rounding. **NCCL guarantees all ranks get the *same* answer, not the same answer as last
  time.**
- **Atomics.** Kernels that scatter into overlapping outputs (embedding-gradient accumulation, and
  this repo's `tl.atomic_add` in the DC backward) sum contributions in whatever order thread blocks
  retire.
- **Autotuning.** cuDNN benchmark mode and Triton autotuning select kernels by *timing candidates at
  runtime*. Ambient noise can flip the winner, and different kernels have different reduction orders
  and split-K strategies.
- **Compiler.** Inductor's fusion decisions and Triton's split-K choices change accumulation order
  relative to eager.
- **This repo specifically.** The async-collective design means the *order* in which gradient shards
  land is timing-dependent **by construction**.

`torch.use_deterministic_algorithms(True)` would force deterministic kernels where they exist and
error where they don't, at a real speed cost, which is why no speedrun uses it.

**The consequence for benchmarking:** two runs with the same seed on the same hardware will not
produce the same final val loss, and the spread is on the order of the improvements being claimed
(records separated by 5–20 ms of step time and a few thousandths of loss). **A single run showing
3.279 proves nothing.**

Hence the rules requiring *statistical* evidence. You can see the discipline in the directory layout:
`records/track_1_short/2026-07-17_FP8DownProjection/` contains `baseline/` and `this_pr/`
subdirectories with ~15 logs each plus a `statistics.md`, a **paired comparison against a re-baselined
control on the same hardware**, not a comparison against a number someone published last month.

---
# Part 8: How a Run Actually Executes

## 8.1 The linear timeline

```
torchrun spawns 8 ranks
  → env vars read; device bound; NCCL+gloo process group; barrier      :54-64
  → model built, cast to bf16, broadcast from rank 0                   :2234-2257
  → torch.compile(model, dynamic=False, fullgraph=True)                :2260
  → WARMUP over transition steps, then FULL STATE RESET                :2264-2301
  → gc.collect()                                                       :2308
  → torch.cuda.synchronize(); t0 = perf_counter()   ← CLOCK STARTS     :2312-2313
  → prefix table build (inside the timed region, deliberately)         :2314-2317
  → 1285-step loop: advance_schedule → grad_accum fwd/bwd → opt.step   :2320-2369
       ↳ periodic UNTIMED validation with cross-rank AVG reduce        :2323-2346
  → final window extension + last validation print                     :2325-2342
  → peak/reserved memory report; destroy_process_group                 :2380-2382
```

## 8.2 The configuration (`Hyperparameters`, `train_gpt.py:1881-1906`)

`@dataclass(slots=True)`.

| Field | Value | Meaning |
|---|---|---|
| `data_path` | env `DATA_PATH`, default `"."` | |
| `train_files` | `.../fineweb10B/fineweb_train_*.bin` | |
| `val_files` | `fineweb_val_*.bin` | |
| `val_tokens` | **10,485,760** | fixed, for comparable val loss |
| `val_batch_size` | `4*64*1024*8 = 2,097,152` | ⇒ 5 validation batches |
| `num_scheduled_iterations` | **1270** | steps over which lr/window/batch schedules complete |
| `num_extension_iterations` | **15** | extra steps at the final lr/window ⇒ **1285 total** |
| `run_id` | `uuid4()` | names the logfile |
| `val_loss_every` | 250 | |
| `save_checkpoint` / `run_evals` | False / False | |
| `bigram_vocab_size` | `50304*15//2 = 377,280` | |
| `bigram_dim` | 768 | |
| `bigram_sign_table_rows` | 8192 | |

**There is no explicit val-loss-target constant in code.** The ≤3.28 threshold is a *benchmark rule*
enforced externally; the code just prints the loss (`:2342`).

## 8.3 The warm-up-then-reset trick (this is what makes the timing honest)

`train_gpt.py:2264-2301`. Before the clock starts, the script runs full forward/backward/optimizer
steps on a **curated set of `warmup_steps`**: step 0, step 1, and ±2 around every stage transition
(`:2274-2277`). The point is to trigger *all* `torch.compile` graphs and all CUDA kernel autotuning.
Then it **restores the initial weights and optimizer state** (`:2295-2298`).

So compilation and autotuning cost is excluded from the record, but the *steady-state* kernels are
already hot when timing begins. Combined with:

- the clock starting only after `torch.cuda.synchronize()` (`:2312`),
- validation being explicitly **stopped out** of the clock (`:2328-2329` accumulates
  `training_time_ms` before val, then restarts at `:2345-2346`),
- `gc.collect()` immediately before the clock (`:2308`),

…this is a clean measurement of the training loop only. The rule change after record #21 (Part 1.2)
formalized exactly this procedure.

**One honest touch:** the prefix-token table is built **inside** the timed region (`:2314-2317`) even
though the tokenizer is preloaded at import (`:1173`). The expensive part is untimed but the cheap
table build is charged. That is an honesty measure, not a speedup.

## 8.4 The data loader

`train_gpt.py:1697-1876`. This is more sophisticated than it looks.

**Shard loading** (`_load_data_shard`, `:1698-1706`): reads the header, validates magic and version,
then `f.readinto(tokens.numpy())` into a **pinned** `torch.empty(..., pin_memory=True)`, avoiding
both a `bytes`→array copy and a later page-lock copy.

**BOS indexing** (`Shard`, `:1712`): scans for BOS positions (`BOS_ID = 50256`, `:1709`). Here is the
trick: it computes a *partial* index over the first 6M tokens synchronously (`:1720`) **plus a
background thread** computing the full index (`:1723-1729`), hot-swapped in at `_maybe_switch`
(`:1731-1735`). **Training starts without waiting for the full scan.**

**Batch construction** (`next_batch`, `:1737-1760`): packs, per rank, **whole documents**
(BOS-aligned) until `num_tokens_local` tokens accumulate, truncating documents to `max_seq_len`.

**Sharding across ranks** (`distributed_data_generator`, `:1796`):
`num_tokens_local = num_tokens // world_size` (`:1816`); rank *r* takes `seq_starts[rank]` /
`seq_ends[rank]` (`:1822`).

**Prefetch**: the next shard loads via `Shard.load_async` on a thread (`:1762-1777`, launched `:1811`
/ `:1828`). Record #33.

**The generator is a coroutine.** It `yield`s GPU tensors and can *receive* new
`(batch_size, max_seq_len, grad_accum)` via `.send()` (`:1862-1875`). **That is how batch/sequence
schedule changes reach the loader without recreating it.** All host→device copies use
`non_blocking=True` on pinned memory (`:1863-1866`).

**Validation** uses `align_to_bos=False`: contiguous slabs (`:1839-1849`), not document-aligned.

## 8.5 The four training stages

`TrainingStage` / `TrainingSchedule`, `train_gpt.py:1909-2007`, stages defined `:1979-1989`. Each of
the first three covers 1/3 of scheduled steps; the fourth is the extension.

| Quantity | Stage 1 | Stage 2 | Stage 3 | Extension |
|---|---|---|---|---|
| Batch (×2048×8 tokens) | 8 | 16 | 24 | 24 |
| Sequence length | 896 | 2048 | 2048 | 2048 |
| Attn window (short, long) in 128-blocks | (1,3) | (3,7) | (5,11) | (6,13) → **(6,20)** for final val |
| `lr_mul` | 1.0 | 1.52 = (16/8)^0.6 | 1.73 = (24/8)^0.5 | 1.73 |
| MTP weights | [1, .5, .25→0] | [1, .5→0] | [1] | [1] |
| Prefix-loss weight | 0.25→0 | 0 | 0 | 0 |

Plus, orthogonally: **Muon momentum** ramps 0.85→0.95 over 300 steps and back down over the last 50
(`:1995-2007`); **LR cooldown** decays linearly to 0.15× over the final `cooldown_frac = 0.60`
(`:1968-1976`, `:1992`); **embed/lm_head untie** at 2/3 (`split_step`, forced odd, `:1948`,
`:2146-2147`).

**On every window change, YaRN is re-interpolated** (`:2114-2116`), which also rescales `attn_scale`.

## 8.6 The step itself

`train_gpt.py:2356-2365`:

```python
for _ in range(grad_accum_steps):
    batch = train_loader.send((batch_size, max_seq_len, grad_accum))   # :2358
    sparse_index_update(...)                                           # :2359
    loss = model(batch).sum() * grad_scale                             # :2360
    sparse_index_share(...)                                            # :2361
    loss.backward()                                                    # :2362
training_manager.step_optimizers(step)                                 # :2364
model.quantize_mlp_fp8(bootstrap_down=(step < 16))                     # :2365
```

`sparse_index_update`/`sparse_index_share` (`:2172-2198`) track which bigram-embedding rows this
micro-batch touched, so the sparse `all_to_all` path knows what to exchange.
`quantize_mlp_fp8` refreshes the FP8 weight copies for the *next* step (delayed scaling), with a
bootstrap phase for the first 16 steps.

## 8.7 Validation

`train_gpt.py:2323-2346`. Triggered on `last_step or step % 250 == 0`.
`val_steps = grad_accum_steps * val_tokens // val_batch_size` (`:2332`). With the defaults, that is 5
batches. A **fresh non-BOS-aligned loader** is created (`:2333`), the model set to `eval()`, and under
`no_grad` it accumulates the per-token mean loss (`:2336-2339`), divides, then reduces across ranks
with `dist.reduce(val_loss, 0, op=AVG)` (`:2341`).

In eval mode the loss path uses the real LM head with the explicit softcap and **fp32
`F.cross_entropy`** (`:1688-1692`), versus the fused softcapped CE with MTP + prefix aux terms during
training. **The training loss and the validation loss are not the same function.** Know that before
you compare them.

On `last_step`, `apply_final_ws_ext()` bumps the long window to 20 before the final validation
(`:2325-2326`, `:2092-2093`): record #33/#35, "extend final-layer attention window for val."

## 8.8 Logging, and why every log is a reproducible artifact

Only rank 0 logs. Logfile = `logs/{uuid4}.txt` (`:2207-2212`); `print0` appends (`:2213-2218`).

At startup it dumps, **before the timed region**:
1. **The complete source of `train_gpt.py` + `triton_kernels.py` + `dc_triton_kernels.py`**: read at
   `:5-12` into `code`, printed at `:2221`.
2. Python / PyTorch / CUDA / Triton versions (`:2224-2226`).
3. Full `nvidia-smi` output (`:2228-2231`).

Then per-step `step/train_time/step_avg` (`:2369`) and per-validation
`step/val_loss/train_time/step_avg` (`:2342`). Peak/reserved memory at the end (`:2380-2381`).

**This is the single most important process design decision in the whole project.** Because the
trainer prints its own source, **any log file is a runnable trainer**. To reproduce any record: take
everything before the `====` separator, save it as a `.py`, and run it. No version drift, no "which
commit was this", no missing config. Copy this idea.

## 8.9 Every wall-clock trick, catalogued

| # | Trick | Where |
|---|---|---|
| 1 | Warm-up-then-state-reset excludes compile & autotune from the clock | `:2264-2301` |
| 2 | Clock excludes validation entirely | `:2328-2329`, `:2345-2346` |
| 3 | FP8 matmuls with transposed weight layout for fast grad-accum | `:67-68`, `:956-960` |
| 4 | Adam runs only every *other* step | `:2013`, `:2134` |
| 5 | Sparse reduce-scatter for the bigram embedding (only touched rows) | `:257-346` |
| 6 | Explicitly scheduled, overlapped collectives: no backward hooks | `:406-410`, `:723-816`, `:2052-2063` |
| 7 | 0-D CPU scalar tensors to dodge recompiles | `:187`, `:459-463` |
| 8 | Async/threaded data prefetch + background BOS scan | `:1723-1729`, `:1762-1777` |
| 9 | Pinned-memory zero-copy shard reads | `:1703-1705` |
| 10 | `gc.collect()` immediately before the clock | `:2308` |
| 11 | Eager NCCL communicator init via `device_id=` | `:62` |
| 12 | `expandable_segments` allocator to survive the shape schedule | `:25` |
| 13 | Hardcoded (offline-autotuned) Triton configs; no runtime autotune | `triton_kernels.py:113` |

---

# Part 9: The Optimizer Research Environment (Track 3)

> **This is almost certainly your working area.** Everything below is operational.

## 9.1 What it is, in its own words

`records/track_3_optimization/README.md:3-4`:

> The goal of this benchmark is to collaboratively|competitively find efficient neural network
> optimizers. Unlike the main NanoGPT speedrun which seeks to minimize *wallclock time* by any means,
> here we aim to minimize *step count* by improving the optimization algorithm (⇒ methods that are
> slow in terms of wallclock are perfectly OK).

**The question it answers:** which optimizer reaches a fixed target loss in the fewest optimizer
steps, holding data, batch size, and architecture constant.

**Why it exists:** to "filter signal from noise" across the hundreds of SOTA-claiming optimizer
papers, by providing **one shared, well-tuned baseline** so undertuned-baseline confounds are
exposed.

## 9.2 How the model differs from track 1

Track 3 deliberately **removed** the speedrun's non-standard parameters (value embeddings, skip
lambdas) and all Triton kernels; **added back** standard-but-slow parameters (RMSNorm gains, Linear
biases); and replaced local-global attention with **plain causal attention over 1024-token contexts**
(`train_gpt_simple.py:109-110`: `F.scaled_dot_product_attention(..., is_causal=True, scale=0.12)`).

It is a **vanilla-architecture sandbox where only the optimizer is the variable.**

## 9.3 The rules, verbatim

> For a new result to be considered valid, it must satisfy the following constraints:
> 1. The dataset, batch size, and architecture must be kept the same as the baseline.
> 2. The trainer cannot perform multiple forward-backward passes per step.
> 3. (**Target loss and statistical significance**) The submitted run(s) must attain below 3.28 val
>    loss… the run(s) are required to pass a one-sided z-test assuming σ=0.0013 that achieves p<.001
>    (hence 3.09σ = 0.004 delta below the target). E.g., for a single non-cherry-picked run, any val
>    loss below 3.276 suffices, and for n=4 runs, any average below 3.278 suffices. **The precise
>    condition we require is `(3.28 - avg_loss) * num_runs**0.5 >= 0.004`**, where `avg_loss` is the
>    average result over `num_runs` non-cherry-picked runs.
> 4. (**Reproducibility**) … all code needed to reproduce the run must be included in the logfile. In
>    particular, third-party optimizer libraries must not be imported; instead, the necessary code
>    must be copied in its entirety into the train script…
> 5. (**No p-hacking using val spam**) Per-run early-stopping based on val loss (or any other form of
>    per-run decision based on val loss) is not allowed. On the other hand, it *is* permitted to print
>    the val loss every 25 steps near the end of training, and then select the earliest step that has
>    stat sig for reaching the target… early stopping is permitted as long as the stopping point is
>    selected the same across all trials.

**You are free to modify:** (1) the optimization algorithm, even if wall-clock-slow; (2) optimization
hyperparameters and their schedules; (3) **model initialization**.

**AI-authored submissions are explicitly allowed** ("no human need even be aware"), but authors are
asked to prune "barnacles": complexity that neither helps nor hurts.

## 9.4 The statistical protocol, decoded

**Validity (vs the target).** Validation loss varies run-to-run. The benchmark treats that as Gaussian
with a *known, pre-estimated* σ = 0.0013 per run. Under H₀ "this configuration's true mean loss is
exactly 3.28", the mean of n runs has standard error σ/√n. So:

```
z = (3.28 − avg_loss) / (σ/√n) = (3.28 − avg_loss)·√n / σ
p < 0.001 one-sided  ⟺  z ≥ 3.09  ⟺  (3.28 − avg_loss)·√n ≥ 3.09σ ≈ 0.004
```

Practical readings: **1 run must land below 3.276; 4 runs need a mean below 3.278; 16 runs need only
3.279.**

Using a *known* σ (a z-test) rather than estimating it from the sample (a t-test) is a deliberate
simplification: it makes small-n submissions legible and prevents a lucky low sample variance from
manufacturing significance.

**Pairwise significance (vs the prior record).** A run can be a valid new *record* (stat-sig below
3.28 at a lower step count) **without being stat-sig better than its predecessor.** The rule:

```
(final_loss_diff + exp_stepbased_loss_diff) / (1/n₁ + 1/n₂)**0.5  ≥  0.004
```

The step↔loss exchange rate is **empirically calibrated** (`README.md:226`): reducing result #12 by
200 steps raised the mean loss from 3.2790 to 3.2881, a gap of 0.0091 across 200 steps, i.e.
**0.0045 val loss per 100 steps.** Worked examples in the README: #11 vs #9 gives p=0.69 (not
pairwise-significant); #13 vs #11 gives LHS=0.00167 (not significant); #12's Muon-hparam change vs #6
gives LHS=0.00648 (**is** significant).

**Read the leaderboard with this in mind.** Several accepted records are *not* pairwise-significant
improvements over their predecessor: #11 (p=0.69), #16 (p=0.34), #34 (p=0.168), #13. **Treat step-count
drops of ~25 skeptically.**

**Why "no val spam" matters.** If you evaluate every 25 steps and stop the run the moment it dips
below target, you are not measuring your optimizer; you are measuring the **minimum of a noisy
sequence**, whose expectation is systematically below the true mean. With σ ≈ 0.0013 and thirty
evaluation points you can harvest a 2σ downward excursion essentially for free, and seed selection
does the same. The result is unreproducible.

The permitted alternative is precise and worth understanding: you may look at the **whole ensemble**
and pick one step count **for everyone**, because that decision is made once and its selection effect
is bounded by the number of candidate step counts, not by (runs × checkpoints).

**Skeptical results are first-class.** Keller does *not* reproduce before accepting; fake results are
possible. Statistically-grounded challenges that disprove old records are welcomed and may earn the
original submitter a ban. Any record can be pulled if it errors or shows mean >3.28.
**That inversion of incentives (making reproduction a publicized contribution rather than an
unpublishable chore) is the benchmark's most transferable idea.**

## 9.5 The baseline: `train_gpt_simple.py`

372 lines. Self-logging: reads its own source at startup (`:10-11`) and prints the entire file plus a
line of 100 `=` before any metrics. **That is what makes each logfile self-reproducing.**

**Model** (`:253`): `GPT(vocab_size=50304, num_layers=12, model_dim=768)`, GPT-2-small class,
head_dim 128 ⇒ 6 heads, MLP hidden 3072. Embedding is `.bfloat16()`. Plain SDPA causal attention,
scale 0.12. RMSNorm **with** gains, Linear **with** biases, half-truncated RoPE, ReLU² MLP, logit
softcap `15 * logits * (logits.square() + 15**2).rsqrt()` (`:155`).

**Default optimizer = leaderboard result #36, "tuned Muon + aux AdamW":**

```python
optimizer1 = AdamW([                                     # :287-290
    dict(params=[model.embed.weight],  lr=0.7),
    dict(params=[model.proj.weight],   lr=0.004),        # LM head
    dict(params=[p for p in ... if p.ndim < 2], lr=0.015)  # gains & biases
], betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)

optimizer2 = Muon([p for p in model.blocks.parameters() if p.ndim >= 2],   # :291-292
                  lr=0.025, weight_decay=0.05, mu=0.95)
```

**Muon internals here** (`:163-186`): Nesterov momentum lerp, **12-step** Newton-Schulz with the
*convergent* classical coefficients `a, b, c = 2, -1.5, 0.5` (`:172-173`, comment: "not optimizing
for wallclock speed"), aspect-ratio scale `max(1, rows/cols)**0.5`, distributed via round-robin
`all_gather`.

**Other hyperparameters:**

| | |
|---|---|
| `train_steps` | **3250** (`:267`), the quantity you minimize |
| `batch_size` | `8*64*1024 = 524,288` tokens; `mbs=64` micro-batch |
| `val_tokens` | `20 * 524288 = 10,485,760` |
| LR schedule (`set_hparams`, `:301-310`) | stable-then-linear-decay, `cooldown_frac=0.7` (flat until 30% remaining) |
| Init (`:270-284`) | `proj` zeroed; `embed` default normal; other weights `normal_(std=sqrt(0.33)/sqrt(fan_in))`; biases zeroed; gains exactly 1 |
| `num_trials` | `int(sys.argv[-1])` (`:257`): trailing arg = number of back-to-back seeds |
| Val frequency | `125` until 90% progress, then **every 25 steps** (`:328`), the grid used for the significance sweep |

## 9.6 The credibility problem this benchmark exists to solve

An optimizer paper's claim is always comparative: "ours beats AdamW by 1.5×." But the comparison has a
free parameter the author controls: **how hard they tried to tune the baseline.** Tuning your own
method exhaustively and the baseline perfunctorily produces a large, publishable, entirely fictitious
speedup. This is not hypothetical; it is the *modal* failure mode. The literature:

- **Choi et al., "On Empirical Comparisons of Optimizers for Deep Learning"** (arXiv:1910.05446)
  established the theoretical form: optimizers form **inclusion hierarchies**. Adam with the right
  hyperparameters can exactly simulate momentum SGD, which can simulate plain SGD. So with unbounded
  tuning the more general optimizer can never lose. **Any paper reporting that Adam loses to SGD has
  demonstrated a tuning artifact, not an algorithmic fact.**
- **Schmidt, Schneider & Hennig, "Descending through a Crowded Valley"** (arXiv:2007.01547) ran
  >50,000 runs over fifteen optimizers and found that *evaluating several optimizers at their defaults
  works about as well as tuning one*, and that no method consistently beat a well-tuned Adam.
- **Wen et al., "Fantastic Pretraining Optimizers and Where to Find Them"** (arXiv:2509.02046) reran
  ten optimizers with **per-optimizer tuning** at four scales: claimed 1.4–2× speedups shrank to
  **1.1× at 1.2B parameters**, with the gap *narrowing* as scale grows. Plus a second failure mode:
  **rankings flip if you compare intermediate checkpoints instead of end-of-training**, because LR
  decay affects methods differently.
- **AlgoPerf** (Dahl et al., arXiv:2306.07179) is MLCommons' institutional response: fixed hardware,
  multiple workloads, time-to-result, and an explicit tuning protocol so "how much did you tune"
  becomes a *controlled variable* rather than author discretion.

**Track 3 is the low-friction version of the same idea:** one workload, ~15 minutes, ~$5, and a
standing public invitation to beat the incumbents, **including an invitation to beat Muon, authored
by the person who invented Muon.**

## 9.7 The leaderboard: all 46 results

Format: **#, steps-to-3.28, evidence(n), description**. `(!)` = new record at the time.
Ⓧ = did not satisfy the significance rule (Keller's own first three).

| # | Steps | Evidence | Method |
|---|---|---|---|
| 1 | 3600 | 3.2777 (n=1)Ⓧ | Muon + aux Adam, lr=.02 wd=.01 |
| 2 | 5625 | 3.2790 (n=1)Ⓧ | **Adam** lr=.0015 betas=(.9,.95) warmup=250; self-labelled undertuned |
| 3 | 3500(!) | 3.2767 (n=1)Ⓧ | Muon + aux Adam, lr=.025 wd=.0125 |
| 4 | 4875 | 3.2741 (n=5) | **AdamH** (Adam + hyperball), per-module init std |
| 5 | 3325(!) | 3.2782 (n=10) | **MuonH** (Muon + hyperball) |
| 6 | 3375 | 3.2788 (n=20) | Muon + aux Adam lr=.025 wd=.025 |
| 7 | 3325 | 3.2752 (n=1) | **Muon²** + aux Adam |
| 8 | 3250 | 3.2778 (n=10) | **NorMuonH**, end 25 steps early |
| 9 | 3250(!) | 3.2771 (n=8) | NorMuon + **u/w-floor** (clamp ‖u‖/‖w‖→0.35), lr=.0375 |
| 10 | 3250 | 3.2789 (n=20) | NorMuon lr=.035 wd=.025 |
| 11 | 3225(!) | 3.2785 (n=16) | #9 + **Contra-Muon** *(p=0.69 vs #9, not pairwise-sig)* |
| 12 | 3325 | 3.2790 (n=20) | Muon + aux Adam lr=.035 wd=.025 |
| 13 | 3210(!) | 3.2785 (n=10) | NorMuonH inside **MuLoCo** outer Nesterov SGD |
| 14 | 3150(!) | 3.2776 (n=4) | #11 + **SOAP-Muon on MLP** |
| 15 | 3275 | 3.2785 (n=15) | **Newton-Muon** |
| 16 | 3125(!) | 3.2784 (n=8) | #14 + SOAP attn + trust gate *(p=0.34)* |
| 17 | 3175 | 3.2789 (n=20) | #11 + **Aurora** |
| 18 | 3225 | 3.2776 (n=9) | **PMuon** |
| 19 | 3125 | 3.2780 (n=6) | #8 with **KL-SOAP** + hyperball |
| 20 | 3030(!) | 3.2790 (n=30) | #16 + **PowerCool** + Contra/Soft-Muon interp |
| 21 | 4100 | 3.2776 (n=4) | **Shampoo** (power −1/4) |
| 22 | **8225** | 3.2774 (n=4) | **SpectralDescent = Muon(mu=0)**: the key negative result |
| 23 | 3075 | 3.2790 (n=30) | **Muown** + PowerCool + V-norm schedule |
| 24 | 3175 | 3.2782 (n=10) | #11 split LR cooldown |
| 25 | 3040 | 3.2781 (n=5) | #19 KL-SOAP-H + PowerCool floors |
| 26 | 3090 | 3.2785 (n=10) | **SinkSOAP** + NorMuon |
| 27 | 3125 | 3.2782 (n=6) | #19 with SOAP + hyperball |
| 28 | 3175 | 3.2790 (n=25) | **DynMuon** |
| 29 | 2990(!) | 3.2787 (n=11) | #20 + **radial brake** |
| 30 | 2930(!) | 3.2784 (n=16) | #29 + Aurora row-balanced polar on mlp.proj, extended Contra ramp, Muon momentum warm/cooldown |
| 31 | 2995 | 3.2789 (n=20) | #23 Muown + NorMuon + Contra-Muon |
| 32 | 3000 | 3.2778 (n=9) | #20 + **SODA** anchor correction |
| 33 | 3375 | 3.2779 (n=5) | **PSGD Kron** whitening + hyperball |
| 34 | 2925(!) | 3.2781 (n=8) | #30 + late **RRE extrapolation** *(p=0.168)* |
| 35 | 3375 | 3.2767 (n=2) | **One-Sided Shampoo** + pseudoinverse root + Adam grafting |
| **36** | **3250** | **3.2787 (n=10)** | **TUNED BASELINE: Muon + aux AdamW. This is `train_gpt_simple.py`.** |
| 37 | 3250 | 3.2786 (n=10) | MuonH with re-tuned aux Adam |
| 38 | 2900(!) | 3.2786 (n=9) | #30 + late-training tricks (backward EMA move, Tempered-Polar) |
| 39 | 3125 | 3.2786 (n=20) | #12 + **EMA-Nesterov** (γ=.99) |
| 40 | 2890(!) | 3.2788 (n=16) | #30 + EMA-Nesterov (β=.3·lr/max_lr) |
| 41 | 2875(!) | 3.2790 (n=20) | #40 + **Circuit-Muon** on attn V/O |
| 42 | 2860(!) | 3.2789 (n=16) | #40 + zero-init biases, fixed ref interp, dropped rademacher |
| 43 | 2850(!) | 3.2786 (n=13) | #41 + late trajectory transforms (BroadDelta, TrailDelta) |
| 44 | 2750(!) | 3.2789 (n=20) | **#41 + SOAP-Muon all hidden (freq=1), aux β₂ tuning, REMOVED Circuit/Contra/Aurora** |
| 45 | 2720(!) | 3.2786 (n=10) | #44 + final-step **Tail-EMA** (horizon 150) |
| **46** | **2690(!)** | **3.2783 (n=8)** | **CURRENT RECORD.** #45 + **RowUpdateFloor** + **Cautious Weight Decay** (CWD=0.025) |

**Current world record = 2690 steps, a 20.8% reduction vs the 3250-step tuned baseline.**

## 9.8 The three most instructive results

**#44: Clean SOAP-Muon (2750 steps). The pivotal result.** It took the 2875-step Circuit-Muon stack
and **removed seven modules** (Circuit-Muon, Contra-Muon, Soft-Muon, Aurora, NorMuon-lite
second-moment, SOAP denom-floor, V-blend), all of which ablated **neutral**. Then it applied three
real levers: **aux-Adam β₂ split (0.99→0.997; the single biggest lever, ≈−45 steps)**, SOAP on **all**
hidden matrices refreshed **every step** (≈−30), and a shorter LR-cooldown horizon (≈−25). The result
is smaller (910 lines vs the bloated predecessor), faster, and lower loss. **`train_gpt_clean_SOTA.py`
in `results/20260609_soap_f1_auxb2_clean/` is the recommended fork point if you want a readable base.**

**#45: Tail-EMA (2720). Purely an eval-time trick.** Keep a weight EMA (τ=150) over the cooldown
tail, then evaluate `(1−λ)·θ + λ·EMA(θ)` with λ=0.6. This is **Polyak/tail averaging**: with a
non-decaying learning rate, SGD iterates form a *noise ball* around the optimum rather than
converging to it. The error decomposes into bias (distance of the ball's centre from the optimum) and
variance (radius). Averaging the last T iterates **kills the variance term at rate 1/T while leaving
bias untouched**, and for strongly convex problems the averaged iterate attains the *statistically
optimal* asymptotic rate, matching what a second-order method achieves, using only first-order steps.
Empirically it also lands in flatter basins. Notably τ and λ here were **derived**
(τ = (t_end − S)/p; λ = σ²/(σ²+bias²) ≈ 0.5), not tuned. Bought −30 steps with one mechanism, at zero
training cost. *(Polyak & Juditsky 1992; SWA arXiv:1803.05407; model soups arXiv:2203.05482.)*

**#46: Current SOTA (2690).** Three levers, each on a *different* axis that survives the radius
pin: Tail-EMA (−42), **RowUpdateFloor**, per-output-row u/w-floor, the one genuinely novel piece
(−18), and post-pin **Cautious Weight Decay** at 0.025 (−19). Script:
`results/20260619_cwd_rowfloor_tailema/train_gpt_cwd_SOTA.py` (981 lines).

## 9.9 Negative results and dead ends: read this before you start

**These are the most valuable entries in the whole benchmark.**

1. **SpectralDescent / pure orthogonalization: 8225 steps** (result #22), ~3× worse than baseline.
   Muon *without momentum* is catastrophic. **Momentum is essential; do not drop it.**
2. **Pure Adam: 5625 steps** (result #2), self-labelled "most likely undertuned": the canonical
   illustration of the very problem the benchmark exists to fight.
3. **Classical second-order methods lose badly to the Muon lineage:** Shampoo 4100, PSGD 3375,
   One-Sided Shampoo 3375. Useful reference points, not paths forward.
4. **Stacking Muon-geometry variants is exhausted.** Soft-Muon and NorMuon-lite row/col variance were
   explicitly **pruned with negative deltas** (result 20260514: *removal improved the stack*), and
   the entire Contra/Circuit/Soft-Muon + Aurora + NorMuon-2nd-moment geometry stack ablated
   **neutral** in #44. Strong signal: **they cancel once every-step all-hidden SOAP is present.**
5. **KL-SOAP is β₁-sensitive.** The `b1090` tuple **FAILS** (3.28294, significance −0.0072); only
   `b1095/sh090` passed.
6. **SODA always-on acts as drag** (3.28102 vs 3.27916 at 3000 steps). Only the *cosine-faded-out*
   variant worked: **anchoring to initialization helps mid-run and hurts late.**
7. **Documented failed sub-ideas** (from the tail-tricks work): positive Tail-EMA / XEWA monotone is
   worse; train-loss EMA gating is too weak; further Tempered-Polar tuning is inert.
8. **The current record's own author lists what is unproven** (`README.md:166`): it is unknown whether
   the attention SOAP trust gate helps; the Muon momentum cooldown is *probably irrelevant* (scheduled
   over steps 2700–2900 but the run is accepted at 2690); PowerCool may be doing little (the run ends
   before the schedule diverges much from WSD); the Rademacher gain init and the depth-scaled `mlp.fc`
   init are below stat-sig (~0.00003 val loss). **The SOTA script contains known barnacles.**

## 9.10 Tuning wisdom, stated by the benchmark

- **Sensitivity ordering: weight decay > learning rate > everything else.**
- **Val loss at step 1000 does NOT predict final loss.** You must run to completion.
- **Recommended search strategy:** halve the step count, tune all hyperparameters on the short run,
  restore the full step count, then re-tune **only** wd + lr.
- Adam reaches the target within ~3B tokens; only "pathologically inefficient" optimizers need >4B.
- PSGD Kron starting point: `lr=.0005, wd=.625`, and from the result-#33 README:
  `precond_lr=1.0, beta=0.95`, linear LR decay from .025, hyperball instead of trust region.

## 9.11 Where the gains are actually coming from now

Reading the leaderboard trajectory: the 2026 gains are **not** from new Muon-geometry variants (those
are ablated-neutral / exhausted). They come from two places:

1. **Eval-time readout tricks**: Tail-EMA, reference/trajectory interpolation, RRE extrapolation.
   These cost *nothing during training* and exploit the fact that the final iterate is a noisy sample
   from a distribution, not the distribution's centre.
2. **Radius-pin-surviving shape levers**: RowUpdateFloor, post-pin Cautious Weight Decay. Once the
   update's global norm is pinned (by hyperball/floors), the remaining degrees of freedom are in the
   *shape* of the update across rows.

If you want to beat 2690, that is where the evidence points.

## 9.12 Operational commands

**Run the baseline or any experiment:**

```bash
git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt
pip install torch==2.11 huggingface_hub
python data/cached_fineweb10B.py 20     # 2B tokens: enough for ~4000 steps
torchrun --standalone --nproc_per_node=$(nvidia-smi -L | wc -l) \
    records/track_3_optimization/train_gpt_simple.py
```

Append an integer for multiple seeds (the script reads `sys.argv[-1]` as `num_trials`):

```bash
torchrun --standalone --nproc_per_node=8 records/track_3_optimization/train_gpt_simple.py 8
```

For runs longer than ~7600 steps: `python data/cached_fineweb10B.py 100` (max 10B tokens).

**⚠️ Hardware trap:** on **A100** with `torch==2.10` + `torch.compile`, you get **NaNs**. Use
`torch==2.11`. (`README.md:52`.) Note this contradicts the top-level `requirements.txt`, which pins
`torch==2.10` for track 1; the tracks want different torch versions.

Any `{1,2,4,8}` × `{A100, H100, A40, H200, GH200, …}` machine works; the script asserts
`8 % world_size == 0` (`train_gpt_simple.py:226`). **Because the score is step count, hardware is
irrelevant to the metric**; leaderboard logs come from H100, A40, A100, H200, H800 and GH200.

**Regenerate the figures:**

```bash
cd records/track_3_optimization && python make_figures.py     # optionally --colors extended
```

`make_figures.py` **parses `README.md` itself** (`:202-219`): it reads every 7-column results-history
row to get step count, evidence, date, and log path. It then loads the referenced logs
(`get_logfile_paths`: a bare `results/UUID.txt` link → one file; a link into a subdirectory → glob
**all** `*.txt` there, `:163-167`), regex-extracts `step:(\d+)/(\d+)\s+val_loss:([0-9.]+)`, treats a
reset to `step 0` as the start of a new run within a file, and averages losses per step across seeds.
Outputs `img/figure_{wr,best}.png` and zoomed variants (zoom windows: wr 2800–3650, best 2800–3400).

**⚠️ If you add a leaderboard row, `make_figures.py` will try to parse it.** Match the existing column
format exactly.

## 9.13 Reproducing any record

> simply rip their python script out of their logfile (take everything before `===`), and then run it
> using the quickstart above.

That is the entire procedure. Save the pre-`===` block as `train_gpt_simple.py` and `torchrun` it. If
it errors, or its mean is stat-sig above 3.28, file an issue; that is grounds for removal.

## 9.14 Log anatomy

A UUID `.txt` log is a **complete self-reproducing artifact**:

1. **Lines 1 → ~N:** the *entire Python source* the run executed (the script does
   `code = open(sys.argv[0]).read()`).
2. **A line of 100 `=` characters.**
3. **One environment line:**
   `Running PyTorch 2.12.0+cu130 ... on NVIDIA H100 80GB HBM3 with world_size 8`. **This is where you
   read the hardware**.
4. **Another line of 100 `=`.**
5. **Per-validation lines:**
   `step:{k}/{train_steps} val_loss:{x:.5f} train_time:{s:.3f}s step_avg:{ms:.2f}ms`, every 125 steps
   early and **every 25 steps near the end** (the grid used for the significance sweep). `step:0`
   shows the untrained ~10.97 loss.

**Important:** the *submitted* step is **not** necessarily the last line. It is the earliest
25-step-grid checkpoint clearing `(3.28−mean)·√n ≥ 0.004` across all seed logs.

**Multiple seeds in one file** are delimited by a fresh `step:0`; `make_figures.py` relies on this.

## 9.15 Results directory conventions

Two layouts coexist under `results/`:

1. **Bare-UUID logs** at top level, e.g. `results/1bd8db7a-....txt` (result #12).
2. **Dated subdirectories** `YYYYMMDD_shortslug/`, e.g. `20260619_cwd_rowfloor_tailema/` (#46),
   `20260527_psgd/` (#33). These bundle multi-seed evidence.

Naming inside subdirectories is either bare UUIDs or **hardware/seed-prefixed**:
`A40_seed0_<uuid>.txt … A40_seed7_<uuid>.txt` (#46, 8 seeds), `H100_<uuid>.txt`, or hyperparameter-
encoded names like `klsoap-h-b1095_sh090-K3125-seed-1.full.txt`.

**A per-result `README.md`** is present in most dated subdirectories and should contain: a title, the
**pseudocode of the update rule**, chosen hyperparameters and schedule rationale, and a **Validation**
block with the raw loss list, `np.mean`, and the explicit significance computation
`3.28 - 3.09*0.0013/sqrt(n)`. Plus references. Subdirectories may also carry `train_*.py` (the exact
ripped script), `figure.png`/`zoomed_figure.png`, and occasionally `stdout/`, `code_logs/`,
`remote_manifest.txt`.

**Follow this structure for your own submissions.** `results/20260527_psgd/README.md` is a good
template.

## 9.16 Your submission checklist

- [ ] Fork `20260619_cwd_rowfloor_tailema/train_gpt_cwd_SOTA.py` (SOTA, 2690) or
      `20260609_soap_f1_auxb2_clean/train_gpt_clean_SOTA.py` (cleaner base, 2750)
- [ ] Change **only** the optimizer, its hyperparameters/schedules, and/or model init
- [ ] Copy any third-party optimizer code **in full** into the train script, with no imports
- [ ] One forward-backward per step
- [ ] Choose your stopping step **once**, applied identically to all seeds
- [ ] Run `n ≥ 8` non-cherry-picked seeds
- [ ] Verify `(3.28 − mean_loss) · sqrt(n) ≥ 0.004`
- [ ] Also compute pairwise significance vs the current record, and **report it honestly even if it
      fails**; several accepted records did
- [ ] Create `results/YYYYMMDD_yourslug/` with README (pseudocode + validation block), all seed logs,
      and the exact script
- [ ] Add the leaderboard row in the exact 7-column format `make_figures.py` expects
- [ ] Prune your barnacles: ablate each component and delete anything neutral

---

# Part 10: Timeline

> The `records/` directory is this project's real history. Each dated
> directory documents one accepted change, with a README, the reproducible script, and multi-seed
> logs. Reading it is the fastest way to understand *why* the current design is what it is.

---

## 10.1 Track 1: the full record table

**Reported time is the record time on standardized hardware at the time of acceptance.** Absolute
seconds across 2024–2026 rows are not directly comparable because hardware was re-baselined several
times (explicitly documented in FlattenForward, XSAGatedLayers, PairedHeadMuon, RecursiveFromBest).
**Trust the per-record delta, not the raw seconds.**

| # | Time | Change | Date | Contributor |
|---|---|---|---|---|
| 1 | 45 min | llm.c baseline | 05/28/24 | @karpathy |
| 2 | 31.4 min | Tuned LR & rotary embeddings | 06/06/24 | @kellerjordan0 |
| **3** | **24.9 min** | **Introduced Muon optimizer** | 10/04/24 | @kellerjordan0, @jxbz |
| 4 | 22.3 min | Muon improvements | 10/11/24 | @kellerjordan0, @bozavlado |
| 5 | 15.2 min | Pad embeddings, ReLU², zero-init projections, QK-norm | 10/14/24 | @Grad62304977 |
| 6 | 13.1 min | Distributed the overhead of Muon | 10/18/24 | @kellerjordan0 |
| 7 | 12.0 min | Upgraded PyTorch 2.5.0 | 10/18/24 | @kellerjordan0 |
| 8 | 10.8 min | Untied embedding and head | 11/03/24 | @Grad62304977 |
| 9 | 8.2 min | Value & embedding skip connections, momentum warmup, logit softcap | 11/06/24 | @Grad62304977 |
| 10 | 7.8 min | Bfloat16 activations | 11/08/24 | @kellerjordan0 |
| 11 | 7.2 min | U-net skip connections & double lr | 11/10/24 | @brendanh0gan |
| **12** | **5.03 min** | **1024-ctx dense attn → 64K-ctx FlexAttention** | 11/19/24 | @KoszarskyB |
| 13 | 4.66 min | Attention window warmup | 11/24/24 | @fernbear.bsky.social |
| 14 | 4.41 min | Value Embeddings | 12/04/24 | @KoszarskyB |
| 15 | 3.95 min | U-net value embeddings + code opts | 12/08/24 | @leloykun, @YouJiacheng |
| 16 | 3.80 min | Split value embeds, block sliding window, separate block mask | 12/10/24 | @YouJiacheng |
| 17 | 3.57 min | Sparsify value embeds, improve rotary, drop an attn layer | 12/17/24 | @YouJiacheng |
| 18 | 3.4 min | Lower logit softcap 30→15 | 01/04/25 | @KoszarskyB |
| 19 | 3.142 min | FP8 head, offset logits, lr decay to 0.1 not 0.0 | 01/13/25 | @YouJiacheng |
| **20** | **2.992 min** | Merged QKV, long-short attn, attn scale, lower Adam eps, batched Muon | 01/16/25 | @leloykun, @fernbear, @YouJiacheng, @brendanh0gan, @scottjmaddox, @Grad62304977 |
| 21 | 2.933 min | Reduced batch size | 01/26/25 | @leloykun |
| 21* | 2.997 min | *Re-timed under updated rules (RuleTweak, not a new record)* | 02/01/25 | - |
| 21* | 3.014 min | *Re-timed with latest torch (StableTorch, not a new record)* | 05/24/25 | - |
| 22 | 2.990 min | Faster gradient all-reduce | 05/24/25 | @KonstantinWilleke et al. (Enigma) |
| 23 | 2.979 min | Overlap computation & gradient communication | 05/25/25 | @ryanyang0 |
| **24** | **2.966 min** | **Replace all_reduce with reduce_scatter** | 05/30/25 | @vagrawal |
| 25 | 2.896 min | Upgrade PyTorch 2.9.0.dev | 07/13/25 | @kellerjordan0 |
| 26 | 2.863 min | Align batch starts with EoS, cooldown frac→.45 | 07/13/25 | @classiclarryd |
| 27 | 2.817 min | Transpose MLP matrix + Triton symmetric-matmul kernel | 07/18/25 | @byronxu99 |
| 28 | 2.812 min | Sparse attention gate | 08/23/25 | @classiclarryd |
| **29** | **2.731 min** | **Flash Attention 3**, 2048 max_doc_len, ws schedule | 09/03/25 | @varunneal |
| 30 | 2.717 min | Drop first MLP layer | 09/05/25 | @EmelyanenkoK |
| 31 | 2.656 min | Dynamically incorporate YaRN in train+val | 09/10/25 | @classiclarryd |
| 32 | 2.625 min | Optimize distributed, skip-gating, bfloat16 usage | 09/11/25 | @bernard24 & AI system **hiverge.ai** |
| 33 | 2.565 min | Async data fetch/index, extend final-layer attn window for val | 09/15/25 | @classiclarryd |
| 34 | 2.547 min | Smear token embeddings 1 position forward | 09/18/25 | @classiclarryd |
| 35 | 2.527 min | Drop first attn layer, extend long windows for val | 09/21/25 | @classiclarryd |
| 36 | 2.495 min | MuonCustomSizing, shared mlp+attn reduce_scatter | 09/23/25 | @classiclarryd |
| 37 | 2.483 min | Compute cross entropy in BF16 during training | 09/27/25 | @Gusarich |
| **38** | **2.476 min** | **Polar Express** (Newton-Schulz replacement) | 09/29/25 | @varunneal |
| 39 | 2.447 min | Update Adam every other step, reduce batch size | 09/30/25 | @classiclarryd |
| 40 | 2.358 min | Backout, hyperparam tuning, optimize lambda padding | 10/04/25 | @classiclarryd |
| **41** | **2.345 min** | **NorMuon** | 10/24/25 | @li_zichong |
| 42 | 2.313 min | Update NorMuon LR, step logic | 10/27/25 | @varunneal |
| **43** | **2.284 min** | **Cautious Weight Decay** w/ schedule | 11/10/25 | @varunneal |
| 44 | 2.269 min | Backward hooks on Adam (Profiling 101) | 11/16/25 | @akash5474 |
| 45 | 2.248 min | Refine skip arch, update exponential decay init | 11/18/25 | @classiclarryd |
| 46 | 2.203 min | Batch size schedule | 11/29/25 | @varunneal |
| 47 | 2.193 min | Multiply attn lambda w/ weight not data, fix warmup | 12/10/25 | @roeeshenberg |
| 48 | 2.170 min | Speed up Muon, pre-multiply lambda, reshape matrices, NorMuon axis | 12/11/25 | @ChrisJMcCormick |
| 49 | 2.146 min | **Partial Key Offset** | 12/14/25 | @classiclarryd |
| 50 | 2.128 min | Extend Cautious Weight Decay to Adam params | 12/18/25 | @roeeshenberg |
| 51 | 2.075 min | Retie Embed to lm_head, retune fp8 scales | 12/19/25 | @varunneal |
| 52 | 2.037 min | Smooth scalars via beta increase, smear-gate lr, freeze scalars | 12/21/25 | @ChrisJMcCormick |
| **53** | **1.988 min** | **Multi-token prediction**, untie embed/lm_head at 2/3 | 12/22/25 | @varunneal, @classiclarryd |
| 54 | 1.940 min | Asymmetric Logit Rescale | 12/26/25 | @classiclarryd |
| 55 | 1.918 min | Gates on value embeds and skip connection | 12/29/25 | @classiclarryd |
| 56 | 1.894 min | Optimize/compile Adam, higher Adam buffer precision | 12/31/25 | @ChrisJMcCormick |
| 57 | 1.878 min | Bfloat16 attn/mlp weights, mixed-precision Muon, interweave Adam/Muon | 01/04/26 | @classiclarryd, @YouJiacheng, @ChrisJMcCormick |
| **58** | **1.820 min** | **Paired Head Attention** | 01/07/26 | @classiclarryd |
| 59 | 1.781 min | Fused Triton kernel for linear-relu-square MLP | 01/10/26 | @andrewbriand8, @Joshrav21 |
| 60 | 1.765 min | Fused Triton kernel for softcapped MTP cross-entropy | 01/16/26 | @soren_dunn_ & AI System **Locus** |
| 61 | 1.748 min | Unified Optimizers & Transposed LM Head | 01/18/26 | @ChrisJMcCormick |
| **62** | **1.655 min** | **Bigram Hash Embedding** (−5.6 s, biggest 2026 jump) | 01/19/26 | @classiclarryd |
| 63 | 1.650 min | Untie Value Embeds | 01/26/26 | @photon_mz |
| 64 | 1.630 min | Tuned nonzero Attn V and O init | 01/30/26 | @srashedll |
| 65 | 1.613 min | Group Value Embeds into single parameter | 01/30/26 | @varunneal |
| 66 | 1.595 min | Torch 2.10 | 01/31/26 | - |
| 67 | 1.540 min | Tune fused softcap kernels + fuse fp8 quant in LM head | 01/31/26 | @andrewbriand8 |
| 68 | 1.535 min | Move bigram hash to GPU | 01/31/26 | @dhrvji |
| 69 | 1.528 min | Kernel Optimizations | 02/02/26 | @EmmettBicker & AI System **Aster** |
| 70 | 1.521 min | Tune value-embed layout and ve_gates | 02/03/26 | @photon_mz |
| 71 | 1.516 min | Sparse bigram gradient comms + CPU loading | 02/06/26 | @roeeshenberg |
| 72 | 1.496 min | Increase min lr + add max_seq_len schedule | 02/10/26 | @dualverse-ai & AI System **Station** |
| 73 | 1.485 min | Partitioned Hyperconnections | 02/12/26 | @sisovicm |
| 74 | 1.468 min | Flattened GPT forward, removed post-attn lambdas, transpose kernels | 02/16/26 | @ChrisJMcCormick |
| 75 | 1.453 min | Cross Entropy Kernel Optimizations | 02/23/26 | @moof2x |
| 76 | 1.446 min | Reuse & tune backward transpose kernel | 02/28/26 | @samacqua |
| 77 | 1.435 min | Replace partitioned hyperconnections w/ single saved activation | 03/06/26 | @classiclarryd |
| 78 | 1.426 min | Tighten fa3 max_num_docs bounds to fineweb distribution | 03/22/26 | @ChrisJMcCormick |
| 79 | 1.411 min | Fuse CE Fwd/Bwd kernel (avoid softcap-sigmoid recalc) | 04/04/26 | @andrewbriand8 |
| 80 | 1.406 min | Muon orthogonalize Q/K in head-pairs, not full 6-head matrix | 04/08/26 | @samacqua |
| **81** | **1.363 min** | **MUDD Skip Connections** | 04/22/26 | @Lisennlp |
| 82 | 1.353 min | **Learnable XSA** | 04/29/26 | @_djdumpling |
| 83 | 1.328 min | Sign Trick on Bigram Embed | 05/20/26 | @TrianX |
| 84 | 1.320 min | FP8 on MLP up-projection forward pass | 05/21/26 | @sisovicm |
| **85** | **1.271 min** | **MUDD gates + Lightweight Dynamically Composable MHA** | 05/27/26 | @Lisennlp |
| 86 | 1.266 min | Algebraic rewrite of XSA (same math, faster) | 05/27/26 | @aryavohra |
| 87 | 1.256 min | Faster ReLU² kernel | 06/11/26 | @cong_ml & AI System **Recursive** |
| 88 | 1.243 min | Prefix token prediction auxiliary loss | 07/13/26 | @jvarho |
| **89** | **1.23 min** | **MLP down-projection in FP8 w/ delayed scaling metric** | 07/17/26 | @Mister-dev-oss, @CerovazS, @MarioPaerle, @GabrieleCirillo, @crisostomi |

## 10.2 The inflection points

```
45 min ──[Muon]──> 24.9 ──[modern arch]──> 15.2 ──[distributed Muon]──> 13.1
   ──[value/skip/softcap]──> 8.2 ──[FlexAttention]──> 5.03 ──[value embeds]──> 4.41
   ──[FP8 head]──> 3.14 ──[SUB-3-MIN]──> 2.99 ──[FA3]──> 2.73 ──[Polar Express/NorMuon]──> 2.31
   ──[MTP + paired heads]──> 1.82 ──[bigram hash]──> 1.655 ──[MUDD + DC]──> 1.27 ──> 1.23
```

Four eras:
1. **Optimizer era (2024 Q4).** Muon, distributed Muon, modern architecture. 45 → 8.2 min.
2. **Attention era (Nov 2024 – Jan 2025).** FlexAttention (the single biggest jump: 7.2 → 5.03),
   window warmup, value embeddings, softcap, FP8 head. 8.2 → 2.99 min. First sub-3-minute.
3. **Systems era (2025).** Communication (FasterReduce, reduce_scatter, noallreduce), FA3, Triton
   kernels, bf16/fp8 everywhere. 2.99 → 2.3 min.
4. **Fusion + exotic-architecture era (2026).** Kernel fusion, bigram hash embeddings, MUDD, DC-MHA,
   XSA, FP8 everywhere. 2.3 → 1.23 min.

**Note the shape of that curve.** The 2024 gains were 30-50% each. The 2026 gains are 0.5-3% each.
This is a maturing benchmark; it is exactly why track 3 (which measures a *cleaner* quantity)
became the more scientifically interesting instrument.

## 10.3 Track 2: GPT-2 Medium

| # | Time | Change | Date | Contributor |
|---|---|---|---|---|
| 1 | 5.8 hr | llm.c baseline (350M) | 05/28/24 | @karpathy |
| 2 | 29.3 min | Scale-up of small-track speedrun | 01/18/25 | @kellerjordan0 |
| 3 | 28.1 min | Added standard weight decay | 02/08/25 | @kellerjordan0 |
| 4 | 27.7 min | Tuned Muon Newton-Schulz coefficients | 02/14/25 | @leloykun |
| 5 | 27.2 min | Increased LR cooldown phase duration | 03/06/25 | @YouJiacheng |
| 6 | 25.95 min | 2× MLP wd, qkv norm, all_reduce/opt.step overlap | 03/25/25 | @YouJiacheng |
| 7 | 25.29 min | Remove FP8 head; ISRU logit softcap; sharded mixed-precision Muon | 04/16/25 | @YouJiacheng |
| 8 | 24.50 min | Cubic sliding window schedule, 2× max window | 04/22/25 | @jadenj3o |
| 9 | 24.12 min | Add two value embeddings | 08/28/25 | @snimu |
| 10 | 24.07 min | Second input embedding | 09/11/25 | @snimu |
| 11 | 23.45 min | Upgrade torch 2.7 → 2.10.0.dev | - | - |
| 12 | 23.28 min | **Snoo** optimizer (outer optimizer around Adam & Muon) | 09/16/25 | @dominikkallusky |
| 13 | 23.14 min | **EMA Wrapper on Muon** | 09/17/25 | @acutkosky |
| 14 | 23.08 min | Combine records 12 & 13 | 09/30/25 | @acutkosky |
| 15 | 23.03 min | Backout (skip 2/3-point → pre-lm_head) | 10/04/25 | @snimu |
| 16 | 22.99 min | Smear-MTP | 11/02/25 | @snimu |
| 17 | 22.98 min | Remove redundant mask op | 11/12/25 | @manikbhandari |
| **18** | **17.35 min** | **Bulk transfer short-track features** (16 layers, dim 1024) | 12/31/25 | - |

Note record 18: a single bulk port of the small track's accumulated wins bought **−5.63 minutes**,
more than the previous sixteen records combined. Track 2 is under-explored relative to track 1.

## 10.4 Reverts, re-certifications, and documented negatives

- **2024-11-09 Replicateleloykun**: an independent replication landed at **3.2824**, above the
  threshold. A documented failure, kept in the repo.
- **2024-11-14 QuantizedFP4**: near-neutral/negative, not carried forward.
- **2025-08-23 SparseAttnGate** explicitly **reverted** an earlier lm-head-scaling change for variance
  reasons.
- **SkipMLPBlocks appears twice** (2025-05-09 and 2025-09-05). The second is **not a new idea; it is
  a re-certification** with identical UUID logs, re-submitted and re-timed against the shifted
  baseline after the summer's FA3/Triton/YaRN stack landed, confirming the change still held.
- **The embedding-tie decision oscillated.** Record #8 untied embed and head. Record #51 **re-tied**
  them. Record #53 then untied them again, but **only at 2/3 of training**. The right answer changed
  as other pieces changed, a good reminder that these components interact.
- **ScaleUp1B and 50Bruns are not records**: they are scaling side-experiments (1B params, 50B-token
  runs), hence their huge times. Useful for provenance, outside the ladder.
- **AI systems are now regular contributors**: hiverge.ai (#32), Locus (#60), Aster (#69), Station
  (#72), Recursive (#87). Their records are held to identical standards.

---
# Part 11: Glossary

## Optimizers appearing in track 3

**With papers (all arXiv IDs verified to resolve to the stated title):**

| Name | Paper | One-line idea |
|---|---|---|
| **Shampoo** | arXiv:1802.09568; scaled: 2002.09018 | One preconditioner per tensor dimension: `L^{-1/4} G R^{-1/4}` |
| **SOAP** | arXiv:2409.11321 | "Shampoo with Adam in the Preconditioner's eigenbasis": run Adam in a slowly-rotating basis so second moments update **every** step even when eigenbases refresh rarely. That is what fixes Shampoo's degradation under infrequent preconditioner updates. |
| **KL-SOAP** | arXiv:2509.03378 | Recast preconditioner estimation as covariance estimation under KL divergence rather than Frobenius norm. KL-Shampoo needs no Adam grafting. |
| **PSGD (Kron)** | arXiv:1512.04202; Lie-group: 2402.04553 | **Whitening**, not curvature-fitting: find P such that `E[(Pg)(Pg)ᵀ] ≈ I`. Robust to noise and to indefinite Hessians. The **Lie-group parameterization** (triangular matrices under matmul) matters because P is fitted by online gradient descent *on the group manifold*. Staying in a connected Lie group guarantees P stays invertible and positive-definite by construction, with no damping or line search. |
| **Muon** | kellerjordan.github.io/posts/muon/; theory: 2409.20325 | Orthogonalized momentum. See Part 4. |
| **NorMuon** | arXiv:2510.05491 | Per-neuron second-moment normalization on top of Muon |
| **Muon²** | arXiv:2604.09967 | Adam-style second-moment preconditioning **before** orthogonalization, so Newton-Schulz converges in fewer iterations |
| **Newton-Muon** | arXiv:2604.01472 | Derives `W ← W − η·msgn(G(ZZᵀ)⁻¹)`; shows plain Muon is an implicit Newton method that **drops** the input-second-moment right preconditioner |
| **DynMuon** | arXiv:2605.17109 | Replace `UVᵀ` with `UΣᵖVᵀ`, scheduling p from +0.25 to −0.25: positive p early emphasizes high-curvature directions, negative p late reallocates toward low-curvature ones |
| **Muown** | arXiv:2605.10797 | Decompose spectral norm into row-magnitude × row-coherence; promote row-magnitude to an explicit optimizer variable under ℓ∞ geometry while Muon handles direction. (Weight normalization, arXiv:1602.07868, applied per-row inside Muon.) |
| **EMA-Nesterov** | arXiv:2605.25395 | Replace Nesterov's noisy one-step lookahead with an EMA of parameter updates, a low-pass filter on the trajectory |
| **SODA** | arXiv:2605.11172 | Shows Muon, Lion, AdEMAMix and NAdam are all *optimistic dual averaging* instances; proposes a wrapper with a 1/k decay that removes weight-decay tuning |
| **Hyperball** | arXiv:2606.16899 (critique: 2607.22444) | Fix the Frobenius norms of **both** the weight and its update to constants, making the **angular** learning rate the sole controlled quantity. Motivation: weight decay's real function is to set an equilibrium weight norm, which implicitly sets the angular rate; Hyperball sets it directly, removing wd tuning. |
| **MuLoCo** | arXiv:**2505.23725** ⚠️ | Muon as the inner optimizer in a DiLoCo-style inner/outer scheme. **The repo links arXiv:2502.07314, which is WRONG**: that ID is a cs.FL paper on monoids. |
| **Adafactor** | arXiv:1804.04235 | Factored second moments: store per-row and per-column sums, reconstruct `v̂_ij ≈ r_i c_j / Σr`. O(m+n) instead of O(mn). |
| **Lookahead** | arXiv:1907.08610 | Fast weights run k inner steps; slow weights move a fraction α toward the endpoint |
| **DiLoCo** | arXiv:2311.08105 | Same inner/outer pattern for distributed training: many inner AdamW steps, then an outer Nesterov step on the accumulated pseudogradient |
| **Cautious Optimizers** | arXiv:2411.16085 | Mask update coordinates whose sign disagrees with the gradient: `u ← u ⊙ 1[u·g > 0]`. "One line of code." |
| **Cautious Weight Decay** | arXiv:2510.12402 | Same masking applied to *decay*. Preserves the original loss's stationary points rather than optimizing a regularized objective. |
| **Grafting** | arXiv:2002.11803 | Direction from optimizer A, step magnitude from B: `u = ‖u_B‖ · (u_A/‖u_A‖)`. Introduced as a *diagnostic* showing many claimed optimizer wins were really implicit LR-schedule wins. Now load-bearing engineering. |
| **K-FAC** | arXiv:1503.05671 | Kronecker-factored Fisher approximation, the ancestor of Shampoo/SOAP/PSGD |
| **WSD schedule** | arXiv:2405.18392 | Warmup → long **constant** → short cooldown. Matches cosine's final loss but lets you fork at any point, making scaling studies far cheaper. |
| **Polyak/tail averaging** | Polyak & Juditsky 1992 (no arXiv); SWA 1803.05407; soups 2203.05482 | Averaging the last T iterates kills the variance term at rate 1/T while leaving bias untouched. See Part 9.8. |
| **RRE / Richardson extrapolation** | Sidi arXiv:1503.02552, 1807.03199 | Given iterates `x_n…x_{n+k}`, find weights `γ_i` summing to 1 minimizing the residual of `Σγ_i x_{n+i}`: an extrapolated estimate of the fixed point. **No prior work applies RRE to neural-net weight trajectories; this is a track-3 original.** |
| **Sinkhorn / matrix equilibration** | Sinkhorn & Knopp 1967 (no arXiv) | Alternately rescale rows and columns until both are uniform. A cheap diagonal preconditioner reducing condition number. |
| **SGLD** | Welling & Teh, ICML 2011 (no arXiv) | Add Gaussian noise of variance 2η to each SGD step; iterates then sample the Bayesian posterior |

**No paper exists; these are benchmark-native folklore.** Real, reproducible from the logfiles,
entirely unpublished. **Do not cite them as literature:**

`Contra-Muon` (github.com/nilin/contra-muon, code only) · `Circuit-Muon` (in-repo README: couples
attention V/O head pairs so the composed operator `W_O·W_V` has controlled per-step change, plus a
gauge rebalance along the invariance `(W_V, W_O) → (R W_V, W_O R⁻¹)`) · `Soft-Muon` · `PMuon`
(in-repo README: `polar(L^{−γ} M R^{−γ})` with streaming covariance, γ=0.3, essentially Shampoo
preconditioning inside Muon) · `Aurora` (github.com/tilde-research/aurora-release, code only:
replaces `polar(G)` with `polar(D·G)` where positive diagonal D is fitted by a damped fixed-point
iteration so the orthogonal factor has **uniform row norms**: Sinkhorn-style equilibration applied
to the polar factor rather than the Gram matrix; reduces to plain polar for square matrices) ·
`PowerCool` (`lr = min(initial_lr, power_c * (2900 - step)**1.2)`) · `SinkSOAP` · `RowFloor` /
`RowUpdateFloor` · `u/w-floor` · `radial brake` · `Tempered-Polar` · `BroadDelta` / `TrailDelta` ·
`CenterShrinkAdam` · `NorMuon-lite` · `SOAP-Muon` (note: nikhilvyas.github.io/SOAP_Muon.pdf, a note,
not a paper) · `One-Sided Shampoo` (folk variant) · `Tail-EMA`

**Not a separate algorithm:** `SpectralDescent`. Track-3 result #22 proves
**SpectralDescent(lr, wd) ≡ Muon(mu=0) ≡ Shampoo(betas=(0,0), power=−1/4)**, the cleanest statement
in the leaderboard about how these families relate.

## Architecture terms

| Term | Meaning |
|---|---|
| **RoPE** | Rotary Position Embedding: rotate coordinate pairs by an angle ∝ position, so dot products depend only on relative position |
| **YaRN** | NTK-by-parts RoPE rescaling: interpolate low-frequency dimensions, leave high-frequency untouched, plus an attention-temperature bump |
| **QK-norm** | RMS-normalize Q and K before the dot product; makes the logit a scaled cosine similarity, preventing self-reinforcing logit growth |
| **Softcap** | Bound logits smoothly. Gemma: `cap·tanh(z/cap)`. Here: `23·σ((z+5)/7.5)` = an affine-shifted, asymmetric tanh |
| **Value embeddings** | Learned per-token vectors injected directly into attention V, bypassing the residual stream. From value-residual learning (arXiv:2410.17897) |
| **U-Net skips** | Cross-layer skip connections from early to late layers, with learned gates |
| **MUDD** | Multiway Dynamic Dense connections (arXiv:2502.12170): per-token data-dependent skip weights, with *separate* weights for the Q, K, V, and residual input streams |
| **DCMHA / DCFormer** | Dynamically Composable MHA (arXiv:2405.08553). Heads borrow each other's attention matrices via an input-dependent H×H composition map |
| **XSA** | Exclusive Self Attention (arXiv:2603.09078): remove the component of the attention output parallel to the token's own value vector |
| **MTP** | Multi-token prediction (arXiv:2404.19737): auxiliary heads predicting t+2, t+3… forcing non-myopic trunk representations |
| **Paired-head attention** | Adjacent head pairs share a doubled head_dim with an interleaved sequence, so heads attend to each other's keys |
| **Attention sink** | A semantically empty always-visible token (usually BOS) that absorbs surplus softmax mass, because softmax rows must sum to 1 |
| **Sparse attention gate** | An explicit learned per-token per-head output gate that lets a head no-op, replacing the sink |
| **Bigram hash embedding** | A 377,280-row embedding indexed by hashed token bigrams, with a ±1 sign table to decorrelate collisions |
| **ReLU²** | `relu(x)²` activation, used instead of GELU in the MLP |
| **Smear** | Mix each token's embedding with a gated fraction of the previous token's |
| **Backout** | Subtract an early-layer context representation before the LM head |

## Systems terms

| Term | Meaning |
|---|---|
| **SM / warp / CTA / cluster** | Streaming Multiprocessor / 32-thread scheduling unit / thread block / Hopper's co-resident block group |
| **Coalescing** | A warp's 32 lanes touching 32 consecutive addresses ⇒ one 128-byte transaction. Otherwise up to 32× the traffic |
| **Bank conflict** | Two lanes hitting different addresses in the same one of shared memory's 32 banks ⇒ serialization |
| **Occupancy** | Resident warps per SM ÷ hardware max. Limited by registers, shared memory, or block count |
| **TMA** | Tensor Memory Accelerator, Hopper hardware doing descriptor-based async bulk tile copies with hardware bounds clamping and swizzling |
| **wgmma** | Warpgroup (4-warp, 128-thread) **asynchronous** matrix-multiply-accumulate; can read A directly from shared memory |
| **Warp specialization** | Splitting warpgroups into producers (issue TMA loads) and consumers (run wgmma), synchronized via a shared-memory circular buffer |
| **PTX / SASS** | NVIDIA's virtual ISA (stable, forward-compatible) / actual per-architecture machine code |
| **NVRTC** | NVIDIA Runtime Compilation: compiles CUDA C++ to PTX in-process, no nvcc subprocess |
| **E4M3 / E5M2** | FP8 formats. E4M3 max 448 (no infinities, one NaN encoding) for forward; E5M2 max 57344 (IEEE-like) for gradients |
| **amax / headroom** | Tensor absolute max, and the >1 fudge factor applied before computing a quantization scale |
| **Delayed scaling** | Derive this step's FP8 scale from previous steps' amax; no extra pass, but stale |
| **MFU** | Model FLOPs Utilization: model-required FLOPs ÷ hardware peak. Excludes recomputation (that's HFU) |
| **NCCL** | NVIDIA Collective Communications Library. Collectives run as **CUDA kernels on SMs**: they compete with your compute |
| **ZeRO-1** | Sharded optimizer state: reduce_scatter → per-shard update → all_gather of *parameters* |
| **Graph break** | A point where TorchDynamo cannot trace and falls back to eager, fragmenting the graph and killing fusion across the boundary |
| **Guard** | A runtime predicate that must hold for a compiled artifact to be reused. Tensor args guard on *metadata*; Python scalars guard on *value* |
| **Pinned memory** | Page-locked host memory whose physical address is fixed, enabling true async DMA |
| **varlen / cu_seqlens** | Packing variable-length documents into one flat sequence with a cumulative-length boundary index |
| **LSE** | Log-sum-exp per query row, saved by FlashAttention so the backward can reconstruct probabilities without storing the n² matrix |

---

# Part 12: Known Discrepancies and Traps

> Everything here is something that will waste your time if you don't know it.

## 12.1 The README describes an optimizer the code does not run

Covered in Part 4.0, repeated because it is the #1 trap:

| Source | Orthogonalizer | Coefficients | Iterations |
|---|---|---|---|
| `README.md` (prose) | `zeropower_via_newtonschulz5` | `(3.4445, -4.7750, 2.0315)` fixed | 5 |
| `train_gpt.py` (**what runs on track 1**) | `polar_express` | **five distinct triples** | 5 |
| `train_gpt_simple.py` (**track-3 baseline**) | `zeropower_via_newtonschulz5` | `(2, -1.5, 0.5)`: the *convergent* classical values | **12** |

All three are correct in context. Verified directly against source.

## 12.2 Torch version conflict between tracks

- Top-level `requirements.txt` pins **`torch==2.10`**.
- Track 3's README says: on **A100**, `torch==2.10` + `torch.compile` produces **NaNs**; use
  **`torch==2.11`**.

If you run track-3 experiments on A100 with the top-level requirements, you will get NaNs and waste a
day. Use separate environments per track.

## 12.3 The sparse gradient path is conditional and silent

`sparse_comms_share_gradients` is gated on `world_size == 8 and grad_accum_steps == 1`
(`train_gpt.py:259-261`) and **silently falls back** to the dense `sharded` path otherwise
(`:479-480`). Benchmarking on 4 GPUs measures a different communication path than the record used.

## 12.4 Training loss ≠ validation loss (different functions)

Training uses `FusedSoftcappedCrossEntropy` with MTP + prefix auxiliary terms (`:1687`). Validation
uses the plain LM head, an explicit softcap expression, and fp32 `F.cross_entropy` (`:1688-1692`).
Do not compare the two numbers directly.

## 12.5 Attention window changes trigger recompilation

`dynamic=False` + a schedule that changes batch size, sequence length, and window size means every
stage transition recompiles. `dynamo.config.recompile_limit = 64` (`:50`) is the budget. **Adding a
schedule stage can silently exceed it and drop you to eager**, which looks like a mysterious 10×
slowdown, not an error.

## 12.6 Hyperparameters must be 0-D CPU tensors, not floats

Pass a Python float that changes each step and you get ~1285 recompiles. Part 7.6 explains the guard
mechanism. If you add a new scheduled hyperparameter, follow the existing `_momentum_t` pattern at
`:459-463`.

## 12.7 Record times across eras are not comparable

Hardware was re-baselined multiple times. Several 2026 rows show *higher* absolute seconds than
earlier rows for this reason (documented in the FlattenForward, XSAGatedLayers, PairedHeadMuon and
RecursiveFromBest READMEs). **Compare deltas within a record, never absolute seconds across dates.**

## 12.8 Many accepted track-3 records are not statistically better than their predecessor

#11 (p=0.69), #16 (p=0.34), #34 (p=0.168), #13: all valid *records* (stat-sig below 3.28 at a lower
step count) but **not pairwise-significant improvements**. Step-count drops of ~25 should be treated
as noise until independently reproduced.

## 12.9 The SOTA scripts contain known barnacles

The current-record author explicitly lists components whose benefit is unproven (Part 9.9, item 8),
including one (the Muon momentum cooldown scheduled over steps 2700–2900) that **cannot possibly
matter, because the run is accepted at step 2690.** Ablate before you build on top.

## 12.10 There are no unit tests anywhere in the kernel files

Neither `triton_kernels.py` nor `dc_triton_kernels.py` contains a reference implementation or a
correctness test. The only validation is the end-to-end multi-seed loss. **If you modify a kernel,
you have no safety net**: write a reference check against the PyTorch equivalent before you trust it.

## 12.11 `dc_triton_kernels.py` has a stale path reference

Its docstring cites `machine2_exps/train_mudd.py`, which does not exist here. The sole importer is
`train_gpt.py:43-44`. Leftover from the kernel's origin repository.

## 12.12 Parameter count is never stated

The code does not print it, and the bigram embedding (377,280 × 768 ≈ 290M) dwarfs the ~124M
transformer trunk. **Do not assume 124M.** "GPT-2 small class" refers to the quality bar, not the
total parameter count.

## 12.13 `make_figures.py` parses the README

If you add a leaderboard row with the wrong column count, figure generation breaks. Match the existing
7-column format exactly.

## 12.14 Things I could not verify

Stated honestly, so you don't inherit unchecked claims:

- **HBM access latency ~450–600 ns**: from third-party microbenchmarks, not NVIDIA-published.
- **The `coordinate_descent_tuning` 25–30 minute cost**: the repo's own claim
  (`train_gpt.py:38` + README), not an independent benchmark.
- **8×H100 rental at $15–25/hr**: market observation, moves constantly.
- **FineWeb pipeline specifics** (14×8 MinHash bands, 0.65 fastText threshold) and **FineWeb-Edu's
  450k/1.3T figures**: from the papers, not re-verified here.
- **HellaSwag split sizes and reported model accuracies**: from literature.
- **The ~4.4 bytes/token estimate** used for the bits-per-byte conversion in Part 2.2: an
  approximation, not measured on this val shard.
- **Chinchilla's ~20:1 ratio**: from the paper.
- **Muon's "singular values land in [0.68, 1.13]"** (README) vs **"S'ᵢᵢ ~ Uniform(0.5, 1.5)"**
  (KellerJordan/Muon docstring): **the two official sources disagree.** The qualitative point is
  unaffected: κ ≈ 1.5–2.2 instead of 10³.
- **Why 0-D *CPU* tensors beat 0-D *CUDA* tensors** for the recompile trick: the guard mechanism is
  certain; the specific CPU-vs-CUDA reasoning is inferred from the kernel-launch path, not verified
  in Inductor source.
- **`torch.cuda._compile_kernel` is a private API** (leading underscore). It can change without
  notice.

## 12.15 arXiv ID corrections

| Cited in repo | Status |
|---|---|
| `arXiv:2502.07314` for MuLoCo | ❌ **WRONG**: resolves to *"Recognizing Numbers"*, a cs.FL paper. Correct ID: **arXiv:2505.23725** |
| `arXiv:2603.09078` for XSA | ✅ **Valid.** It looks like a future date but 2603 = March 2026, which is in the past. Title: *"Exclusive Self Attention"* (Shuangfei Zhai) |
| `arXiv:2410.17897` for value embeddings | ✅ Valid, but the repo cites the **v1–v4 title**. v5 (Jun 2025) renamed it to *"Value Residual Learning"*. Cite the short title. |
| Hyperball | Repo links only a Notion post. The paper exists: **arXiv:2606.16899**; there is also a critique, **arXiv:2607.22444** ("Hyperball May Not Be a Free Lunch"), arguing the advantage comes from an effective-step-size schedule rather than a better update direction. |
| SODA | Repo cites no ID. It is **arXiv:2605.11172** |
| All others checked | ✅ Resolve correctly |

---

# Appendix A: Fast-path answers

**"I want to run the thing."**
→ Part 1.8 (track 1) or Part 9.12 (track 3).

**"I want to beat the track-3 record."**
→ Part 9.16 (checklist), Part 9.9 (what's already failed), Part 9.11 (where gains remain), then fork
`results/20260609_soap_f1_auxb2_clean/train_gpt_clean_SOTA.py`.

**"What does 3.28 actually mean?"**
→ Part 2.2–2.4. Perplexity 26.58; 4.73 bits/token.

**"Explain Muon to me."**
→ Part 4.2–4.5. Short version: treat the weight matrix as a linear operator, snap all its singular
values to 1, which is steepest descent under the spectral norm, computed via matmul-only
Newton-Schulz so it survives bf16 on a GPU.

**"Why is there a hand-written CUDA kernel in a PyTorch repo?"**
→ Part 6.5. Five mechanisms: fusion, mathematical structure, recompute-vs-store, coalescing, FP8.

**"What is an H100 actually?"**
→ Part 3. The one number to remember: **295 FLOPs per byte** is the compute/bandwidth breakeven.
Below it you are bandwidth-bound, which is most things.

**"Why do the records need so many log files?"**
→ Part 7.11 and Part 2.14. Run-to-run σ ≈ 0.0013–0.005 is comparable to the improvements claimed.

**"Something is 10× slower than expected."**
→ Check `recompile_limit` (12.5), then check whether you passed a Python float where a 0-D tensor was
expected (12.6), then check `world_size` (12.3).

---

*End of knowledge base.*
