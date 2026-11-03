#!/usr/bin/env python3
"""Replace Latin-script English loanwords in a model's predictions with their
native-script transliteration, using a lookup table mined from Vaani's own
reference transcriptions (code-switched words appear there as 'देवनागरी {english}').

One table per language: Hindi uses vaani_en_deva_lookup.json, the other
languages lookups/vaani_<language>_lookup.json. lookups/vaani_<language>_lookup_extra.json
(where present) adds English loanwords seen only in model outputs, with
rule-based spellings; a mined reference spelling always wins. Mizo, Nyishi and
English are written in Latin script, so they have no table and are left unchanged.

This normalizes model output to the same script convention every other model
uses for code-switched words, avoiding an unfair WER penalty for correct
content transcribed in the "wrong" script.

Input/output shape matches run_eval.py's --mode score contract: a flat list
of strings (index-aligned to the dataset split) or a dict keyed by audio
filename, either bare or wrapped as {"predictions": [...] | {...}}.

Usage:
    .venv/bin/python transliterate_predictions.py <predictions.json> [--language telugu]

Writes back to the same file in place (after printing a substitution count),
unless --output is given.
"""

import argparse
import json
import re
from pathlib import Path

LOOKUP_PATH = Path(__file__).parent / 'vaani_en_deva_lookup.json'
LOOKUP_DIR = Path(__file__).parent / 'lookups'
WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


def load_lookup(language):
    """The language's table (mined spellings over the curated extras); None if it has none."""
    language = language.lower()
    if language == 'hindi':
        return json.loads(LOOKUP_PATH.read_text(encoding='utf-8'))
    path = LOOKUP_DIR / f'vaani_{language}_lookup.json'
    if not path.exists():
        return None
    lookup = json.loads(path.read_text(encoding='utf-8'))
    extra_path = LOOKUP_DIR / f'vaani_{language}_lookup_extra.json'
    if extra_path.exists():
        lookup = {**json.loads(extra_path.read_text(encoding='utf-8')), **lookup}
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('predictions_json')
    parser.add_argument('--output', default=None)
    parser.add_argument('--language', default='hindi',
                        help='dataset config the predictions are for (default: hindi)')
    args = parser.parse_args()

    lookup = load_lookup(args.language)
    if lookup is None:
        print(f"No lookup table for {args.language} (Latin-script language): predictions left unchanged")
        return

    in_path = Path(args.predictions_json)
    data = json.loads(in_path.read_text(encoding='utf-8'))
    wrapped = isinstance(data, dict) and 'predictions' in data
    preds = data['predictions'] if wrapped else data
    items = preds.items() if isinstance(preds, dict) else enumerate(preds)

    total_words = 0
    replaced_words = 0
    entries_changed = 0

    def sub_word(m):
        nonlocal total_words, replaced_words
        total_words += 1
        w = m.group(0).lower()
        if w in lookup:
            replaced_words += 1
            return lookup[w]
        return m.group(0)

    for key, text in items:
        if not text:
            continue
        new_text = WORD_RE.sub(sub_word, text)
        if new_text != text:
            entries_changed += 1
            preds[key] = new_text

    print(f"Latin words seen: {total_words}, transliterated: {replaced_words}, entries changed: {entries_changed}")

    out_path = Path(args.output) if args.output else in_path
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved -> {out_path}")


if __name__ == '__main__':
    main()
