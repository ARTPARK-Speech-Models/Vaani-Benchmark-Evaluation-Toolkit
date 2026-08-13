#!/usr/bin/env python3
"""One-command wrapper for the scoring pipeline:
predictions.json -> [transliterate] -> score -> [publish]

Chains transliterate_predictions.py, run_eval.py, and push_eval_results.py
with the right arguments — each step's script can still be run standalone
(see README.md) if you want to inspect intermediate output or only run
part of the pipeline.

For --mode inference (letting run_eval.py load your model itself via
transformers.pipeline), just call run_eval.py directly — it's already a
single command and produces no intermediate predictions.json to
transliterate.

Transliteration (Latin -> Devanagari code-switch normalization) runs by
default; pass --no-transliterate to skip it.

Usage:
    .venv/bin/python main.py --model org/model-name --predictions predictions.json
    .venv/bin/python main.py --model org/model-name --predictions predictions.json --publish
    .venv/bin/python main.py --model org/model-name --predictions predictions.json --no-transliterate --publish --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def run(script_args: list[str]):
    cmd = [sys.executable, str(HERE / script_args[0])] + script_args[1:]
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True, help='HF model repo id')
    parser.add_argument('--predictions', required=True, help='predictions.json to score')
    parser.add_argument('--no-transliterate', dest='transliterate', action='store_false',
                        help='Skip Latin-script normalization before scoring (on by default)')
    parser.add_argument('--publish', action='store_true',
                        help='Push the resulting Hindi_WER to the model repo on HF')
    parser.add_argument('--dry-run', action='store_true', help='With --publish, preview without writing')
    parser.add_argument('--output', default='eval_results.json')
    parser.set_defaults(transliterate=True)
    args = parser.parse_args()

    if args.transliterate:
        run(['transliterate_predictions.py', args.predictions])

    run(['run_eval.py', '--model', args.model, '--mode', 'score',
         '--predictions', args.predictions, '--output', args.output])

    if args.publish:
        cmd = ['push_eval_results.py', '--model', args.model, '--results', args.output]
        if args.dry_run:
            cmd.append('--dry-run')
        run(cmd)


if __name__ == '__main__':
    main()
