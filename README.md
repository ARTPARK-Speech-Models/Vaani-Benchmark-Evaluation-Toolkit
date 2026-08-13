# Vaani Benchmark Evaluation Toolkit

Evaluate an ASR (speech-to-text) model against
[`ARTPARK-IISc/Vaani-Benchmark-V1.0`](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Benchmark-V1.0)
— 5,343 Hindi speech segments from 1,103 speakers across 104 Indian
districts, each with 3 independent human transcriptions — and publish the
score to the model's HF card via the standard
[HF eval-results](https://huggingface.co/docs/hub/main/en/eval-results)
mechanism.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN (write access needed only for the final publish step)
```

**Already have `predictions.json`** (your own model, your own inference
script — see [1. Get predictions](#1-get-predictions))?
`main.py` chains transliteration + scoring + publishing in one command
(transliteration runs by default; pass `--no-transliterate` to skip it):

```bash
.venv/bin/python main.py --model org/model-name --predictions predictions.json --publish
# --dry-run previews the publish step without writing anything
```

**No predictions.json yet, and your model loads through a plain
`transformers.pipeline` call?** `run_eval.py --mode inference` runs
inference and scoring together — no separate transliteration step, since
there's no intermediate `predictions.json` to normalize:

```bash
.venv/bin/python run_eval.py --model org/model-name --mode inference
.venv/bin/python push_eval_results.py --model org/model-name --results eval_results.json
```

Either way, each underlying script (`transliterate_predictions.py`,
`run_eval.py`, `push_eval_results.py`) still runs standalone if you want
to inspect intermediate output or only run part of the pipeline — see
below.

### 1. Get predictions

Not included — every ASR model has a different loading/inference API.
Load `datasets.load_dataset("ARTPARK-IISc/Vaani-Benchmark-V1.0",
split="test")`, run your model over the `audio` column, and write
`predictions.json` in one of two shapes (`run_eval.py` and
`transliterate_predictions.py` auto-detect which one you gave them):

- **Dict, keyed by audio filename** (`dataset[i]["audio"]["path"]`, with or
  without the `.wav` extension) — matched by id against the dataset instead
  of row order, so it's safe even if your inference ran out of order. Any
  dataset row with no matching key scores as an empty prediction (a printed
  warning tells you how many). Recommended shape — see
  `predictions.example.json` (`{"predictions": {filename: text}}`).
- **List, index-aligned** — `predictions[i]` is your model's transcription
  of `dataset[i]["audio"]`, either a bare list or `{"predictions": [...]}`.
  Only correct if you scored the dataset in its default load order and
  didn't reshuffle/filter it — no example file, prefer the id-keyed shape
  above.

`predictions.example.json` is a 5-entry shape demo, not runnable as-is
— a real file has one entry per row in the test split (5,050 of them).

`run_eval.py --mode inference` does this automatically for models that
load through a generic `transformers.pipeline('automatic-speech-recognition', ...)`
call — useful flags: `--device cuda`, `--batch-size`, `--revision`. It runs
inference and scoring together, so if you use it you can skip straight to
step 3.

### 2. Normalize code-switched Latin script — optional, recommended

Vaani's Hindi speakers frequently code-switch into English mid-sentence.
Reference transcribers write these as Devanagari with the English gloss in
braces, e.g. `टाइम {time}`. Some ASR models transcribe these words in Latin
script instead of transliterating them like the references do — a
legitimate transcription that nonetheless scores as 100% wrong against
every reference under WER.

```bash
.venv/bin/python transliterate_predictions.py predictions.json
```

Replaces a Latin-script word with its Devanagari form **only when the exact
lowercase word is found** in `vaani_en_deva_lookup.json`; anything not in
the lookup is left untouched.

`vaani_en_deva_lookup.json` (3,050 entries) was mined entirely from the
dataset's own reference transcriptions — regex-extracted from the
`देवनागरी {english}` pattern shown above. It contains **only words the
corpus itself demonstrates a transliteration for**, deliberately not
supplemented with "obvious" or dictionary entries, since that would make
the normalization partially subjective rather than purely corpus-derived.
Evaluating against a different corpus? Regenerate the lookup from that
corpus's own annotations rather than reusing this one blindly.

If a model's output is **100% Latin script** even after this pass, that's a
sign it isn't actually attempting Hindi transcription (translating instead
of transcribing, or defaulting to English) — worth excluding from
comparison rather than scoring, since WER against Devanagari references
will be near-meaningless.

### 3. Score

```bash
.venv/bin/python run_eval.py --model org/model-name --mode score --predictions predictions.json
```

Computes **`Hindi_WER`**, chosen over a simpler single-reference or
best-of-3 WER because Vaani's 3 transcribers routinely disagree on
punctuation, disfluencies, and borderline word choices — a single noisy
reference shouldn't be able to tank (or inflate) a model's score:

1. Clean references and hypothesis the same way the reference
   transcriptions were authored — strip bracketed code-switch glosses
   (`टाइम {time}` → `टाइम`) and punctuation.
2. Align the hypothesis against each of the 3 references independently
   (word-level alignment via `jiwer`).
3. A hypothesis word counts as a **deletion** only if *all 3* references
   agree it's missing; substitutions/insertions are counted per-word based
   on agreement across the three alignments.
4. `Hindi_WER = (substitutions + insertions + deletions) / total_ref_len`,
   summed across the whole split.

Writes `eval_results.json` with the `Hindi_WER` field.

### 4. Publish

```bash
.venv/bin/python push_eval_results.py --model org/model-name --results eval_results.json
# --dry-run to preview the YAML without writing anything
```

Opens a **PR** on the **model** repo adding
`.eval_results/ARTPARK-IISc-Vaani-Benchmark-V1-0.yaml` — never commits
straight to main, so no write access needed, only a valid `HF_TOKEN`. Still
externally visible once opened: dry-run first, and don't run it against a
repo you don't intend to actually publish a score to.
