#!/usr/bin/env python3
"""Run the canonical Vaani-Benchmark-V1.0 WER evaluation ("approach 3").

Two modes:
  --mode inference   Load model, run transcription, score against all references
  --mode score       Score existing predictions file (skip inference)

Scoring method: for each sample, align the hypothesis against all 3
reference transcriptions independently (word-level alignment via jiwer),
then count a word as deleted only if ALL 3 references agree it's missing,
and substituted/inserted based on per-word agreement across the three
alignments. This is more robust than picking a single reference or taking
the min/best-of-3 WER, since Vaani's 3 transcribers routinely disagree on
punctuation, disfluencies, and borderline word choices — a single noisy
reference shouldn't be able to tank (or inflate) a model's score.

References and hypotheses are cleaned before alignment: bracketed
code-switch glosses (`टाइम {time}`) and punctuation are stripped, matching
how the reference transcriptions were authored.

--mode score accepts predictions.json as a flat list / {"predictions": [...]}
(index-aligned to the dataset split, in load order) or as a dict /
{"predictions": {filename: text}} keyed by each row's audio filename
(with or without the .wav extension) — matched by id instead of row order,
robust to split/revision reordering. Missing ids score as empty string.

Usage:
    .venv/bin/python run_eval.py --model org/model-name --mode inference
    .venv/bin/python run_eval.py --model org/model-name --mode score --predictions my_preds.json
"""

import argparse
import json
import os
import re
import sys

from dotenv import load_dotenv
from jiwer import process_words

load_dotenv()

DATASET_REPO = 'ARTPARK-IISc/Vaani-Benchmark-V1.0'
DATASET_CONFIG = 'Hindi'
AUDIO_COLUMN = 'audio'
REFERENCE_COLUMNS = {'v1': 'transcription_v1', 'v2': 'transcription_v2', 'v3': 'transcription_v3'}


def clean_text(text: str) -> str:
    """Strip bracketed glosses/annotations and punctuation, matching how
    Vaani's reference transcriptions are authored (e.g. 'टाइम {time}' -> 'टाइम')."""
    text = text or ''
    text = re.sub(r'\(.*?\)|\[.*?\]|\{.*?\}|<.*?>|❴.*?❵', '', text)
    text = re.sub(r'[,?.!;:"%\'‘’“”—–\-।]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def approach3_wer(refs1: list[str], refs2: list[str], refs3: list[str], hyps: list[str]) -> float:
    """3-way reference-aligned WER — see module docstring for the method."""
    total_subs = total_ins = total_dels = total_ref_len = 0

    for r1, r2, r3, hyp in zip(refs1, refs2, refs3, hyps):
        word_errors: dict[int, list[str]] = {}
        deleted_per_ref = []
        hyp_words = hyp.split()

        for ref in (r1, r2, r3):
            out = process_words(ref, hyp)
            hyp_words = out.hypotheses[0]
            ref_words = out.references[0]
            deleted_words = set()
            for chunk in out.alignments[0]:
                if chunk.type == 'equal':
                    for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                        word_errors.setdefault(i, []).append('correct')
                elif chunk.type == 'insert':
                    for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                        word_errors.setdefault(i, []).append('insert')
                elif chunk.type == 'substitute':
                    for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                        word_errors.setdefault(i, []).append('substitute')
                elif chunk.type == 'delete':
                    deleted_words.update(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
            deleted_per_ref.append(deleted_words)

        true_deletions = set.intersection(*deleted_per_ref) if deleted_per_ref else set()
        dels = len(true_deletions)
        subs = sum(1 for labels in word_errors.values() if 'correct' not in labels and 'substitute' in labels)
        ins = sum(1 for labels in word_errors.values() if 'correct' not in labels and 'substitute' not in labels and 'insert' in labels)

        total_subs += subs
        total_ins += ins
        total_dels += dels
        total_ref_len += dels - ins + len(hyp_words)

    if total_ref_len == 0:
        return float('nan')
    return (total_subs + total_ins + total_dels) / total_ref_len


def run_inference(args) -> tuple[dict[str, list[str]], list[str]]:
    """Returns ({ref_key: [str]}, predictions)."""
    try:
        from transformers import pipeline
    except ImportError:
        print('ERROR: transformers not installed. Run: pip install transformers torch', file=sys.stderr)
        sys.exit(1)

    from datasets import load_dataset

    token = os.environ.get('HF_TOKEN')

    print(f"Loading dataset: {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, DATASET_CONFIG, split=args.split, token=token)

    print(f"Loading model: {args.model}")
    asr = pipeline(
        'automatic-speech-recognition',
        model=args.model,
        revision=args.revision,
        device=args.device,
        token=token,
    )

    references = {key: ds[col] for key, col in REFERENCE_COLUMNS.items()}
    audio_arrays = [
        {'array': s['array'], 'sampling_rate': s['sampling_rate']}
        for s in ds[AUDIO_COLUMN]
    ]

    print(f"Running inference on {len(audio_arrays)} samples (batch_size={args.batch_size}) ...")
    outputs = asr(audio_arrays, batch_size=args.batch_size)
    predictions = [o['text'] for o in outputs]

    return references, predictions


def run_score_only(predictions_file: str, args) -> tuple[dict[str, list[str]], list[str]]:
    from datasets import Audio, load_dataset

    token = os.environ.get('HF_TOKEN')

    ds = load_dataset(args.dataset, DATASET_CONFIG, split=args.split, token=token)
    references = {key: ds[col] for key, col in REFERENCE_COLUMNS.items()}

    with open(predictions_file) as f:
        data = json.load(f)
    preds_field = data if not isinstance(data, dict) else data.get('predictions', data)

    if isinstance(preds_field, dict):
        # id-keyed: matched against each row's audio filename, not row order.
        paths = ds.cast_column(AUDIO_COLUMN, Audio(decode=False))[AUDIO_COLUMN]
        filenames = [os.path.basename(a['path']) for a in paths]
        predictions = []
        missing = []
        for fname in filenames:
            stem = os.path.splitext(fname)[0]
            text = preds_field.get(fname, preds_field.get(stem))
            if text is None:
                missing.append(fname)
                text = ''
            predictions.append(text)
        if missing:
            print(f"WARN: {len(missing)}/{len(filenames)} dataset rows had no matching "
                  f"prediction (scored as empty), e.g. {missing[:3]}", file=sys.stderr)
    else:
        predictions = preds_field
        n_ref = len(next(iter(references.values())))
        if len(predictions) != n_ref:
            print(f"ERROR: predictions ({len(predictions)}) != references ({n_ref})", file=sys.stderr)
            sys.exit(1)

    return references, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='HF model repo id to evaluate')
    parser.add_argument('--dataset', default=DATASET_REPO)
    parser.add_argument('--split', default='test')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--device', default='cpu', help='"cpu" or "cuda" (--mode inference only)')
    parser.add_argument('--batch-size', type=int, default=8, help='--mode inference only')
    parser.add_argument('--mode', choices=['inference', 'score'], default='inference')
    parser.add_argument('--predictions', help='Existing predictions JSON (--mode score)')
    parser.add_argument('--output', default='eval_results.json')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Embed predictions/references in the output JSON')
    args = parser.parse_args()

    if args.mode == 'score':
        if not args.predictions:
            print('ERROR: --predictions required for --mode score', file=sys.stderr)
            sys.exit(1)
        references, predictions = run_score_only(args.predictions, args)
    else:
        references, predictions = run_inference(args)

    print('Aligning against all 3 transcription references (approach 3) ...')
    refs1 = [clean_text(t) for t in references['v1']]
    refs2 = [clean_text(t) for t in references['v2']]
    refs3 = [clean_text(t) for t in references['v3']]
    hyps  = [clean_text(t) for t in predictions]

    wer = approach3_wer(refs1, refs2, refs3, hyps)

    output = {
        'model':     args.model,
        'dataset':   args.dataset,
        'split':     args.split,
        'n_samples': len(predictions),
        'Hindi_WER': round(wer * 100, 1),
    }
    print(f"\nHindi_WER: {output['Hindi_WER']}")

    if args.save_predictions:
        output['predictions'] = predictions
        output['references']  = {k: list(v) for k, v in references.items()}

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
