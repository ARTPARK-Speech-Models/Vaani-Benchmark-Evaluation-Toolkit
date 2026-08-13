#!/usr/bin/env python3
"""Push Vaani evaluation results to a model repo's .eval_results/ folder on HF Hub.

Reads eval_results.json (output of run_eval.py) and writes
.eval_results/<dataset-slug>.yaml to the model repo.

Results appear on the model card and feed the HF benchmark leaderboard.
Docs: https://huggingface.co/docs/hub/main/en/eval-results

Usage:
    .venv/bin/python push_eval_results.py --model org/model-name --results eval_results.json
"""

import argparse
import json
import os
import re
import sys
from datetime import date

import yaml
from dotenv import load_dotenv

load_dotenv()

DATASET_REPO = 'ARTPARK-IISc/Vaani-Benchmark-V1.0'
ALL_TASKS = ['Hindi_WER']


def slug(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '-', text)


def build_entries(results: dict, dataset_repo: str, tasks: list[str]) -> list[dict]:
    today = date.today().isoformat()
    entries = []
    for task_id in tasks:
        value = results.get(task_id)
        if value is None:
            print(f'WARN: metric "{task_id}" not in results, skipping', file=sys.stderr)
            continue
        entries.append({
            'dataset': {'id': dataset_repo, 'task_id': task_id},
            'value':   round(float(value), 6),
            'date':    today,
        })
    return entries


def push_to_model_repo(entries: list[dict], model_repo: str, dataset_repo: str, dry_run: bool):
    from huggingface_hub import HfApi

    token = os.environ.get('HF_TOKEN')
    yaml_content = yaml.dump(entries, default_flow_style=False, allow_unicode=True)
    filename = f'.eval_results/{slug(dataset_repo)}.yaml'

    print(f'Writing {filename} to {model_repo}')
    print('---')
    print(yaml_content)
    print('---')

    if dry_run:
        print('DRY RUN — not pushing')
        return

    api = HfApi(token=token)
    result = api.upload_file(
        path_or_fileobj=yaml_content.encode(),
        path_in_repo=filename,
        repo_id=model_repo,
        repo_type='model',
        commit_message=f'Add Vaani eval results for {dataset_repo}',
        create_pr=True,
    )
    print(f'Opened PR: {result.pr_url}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   required=True, help='HF model repo id to publish results to')
    parser.add_argument('--dataset', default=DATASET_REPO)
    parser.add_argument('--results', default='eval_results.json')
    parser.add_argument('--tasks',   nargs='+', default=ALL_TASKS,
                        help='Task IDs to push (must match the dataset\'s published eval-results task ids)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)

    entries = build_entries(results, args.dataset, args.tasks)
    if not entries:
        print('ERROR: no valid metric entries to push', file=sys.stderr)
        sys.exit(1)

    push_to_model_repo(entries, args.model, args.dataset, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
