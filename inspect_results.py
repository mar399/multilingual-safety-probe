#!/usr/bin/env python3
"""
Result inspection utility.

Prints a readable view of saved result files without exposing prompt
text (which is never stored in result files in the first place).

Usage:
    python inspect_results.py                    # list all results
    python inspect_results.py --lang ha           # filter by language
    python inspect_results.py --show id_ha_001    # show one result in full
"""

import json
import argparse
from pathlib import Path

RAW_DIR = Path("results/raw")


def load_all():
    records = []
    for path in sorted(RAW_DIR.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception:
            continue
    return records


def main():
    ap = argparse.ArgumentParser(description="Inspect FIVB result files")
    ap.add_argument("--lang", type=str)
    ap.add_argument("--category", type=str)
    ap.add_argument("--model", type=str)
    ap.add_argument("--show", type=str, help="prompt_id to show in full")
    args = ap.parse_args()

    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.json")):
        print("No results found. Run run_battery.py first.")
        return

    records = load_all()

    if args.lang:
        records = [r for r in records if r.get("prompt_language") == args.lang]
    if args.category:
        records = [r for r in records if r.get("prompt_category") == args.category]
    if args.model:
        records = [r for r in records if r.get("model") == args.model]

    if args.show:
        match = [r for r in records if r.get("prompt_id") == args.show]
        if not match:
            print(f"No result found for prompt_id={args.show}")
            return
        r = match[0]
        print(f"\n{'='*60}")
        print(f"Prompt ID:  {r['prompt_id']}")
        print(f"Category:   {r.get('prompt_category')}")
        print(f"Language:   {r.get('prompt_language')}")
        print(f"Model:      {r.get('model')}")
        print(f"Orientation score: {r.get('score_orientation')}")
        print(f"Technique count:   {r.get('score_technique_count')}")
        print(f"{'='*60}")
        print("\nResponse:\n")
        print(r.get("response", "")[:2000])
        return

    print(f"\n{len(records)} result(s)\n")
    print(f"{'PROMPT ID':<20} {'LANG':<6} {'CATEGORY':<20} {'MODEL':<10} {'OR':<4} {'TC':<4}")
    print("-" * 70)
    for r in records:
        print(f"{r.get('prompt_id','?'):<20} "
              f"{r.get('prompt_language','?'):<6} "
              f"{r.get('prompt_category','?'):<20} "
              f"{r.get('model','?'):<10} "
              f"{str(r.get('score_orientation','-')):<4} "
              f"{str(r.get('score_technique_count','-')):<4}")


if __name__ == "__main__":
    main()
