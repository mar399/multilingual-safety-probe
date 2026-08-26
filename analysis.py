#!/usr/bin/env python3
"""
Aggregate statistics across all scored results.

Reads results/scored/*.json and prints summary statistics: mean
orientation score by language and category, gap counts, and overall
distribution. Complements the detailed report from score_results.py.

Usage:
    python analysis.py
"""

import json
from pathlib import Path
from collections import defaultdict

SCORED_DIR = Path("results/scored")


def main():
    if not SCORED_DIR.exists() or not any(SCORED_DIR.glob("*.json")):
        print("No scored results found. Run score_results.py first.")
        return

    records = []
    for path in SCORED_DIR.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception:
            continue

    print(f"\n{len(records)} scored result(s)\n")

    by_lang = defaultdict(list)
    by_cat = defaultdict(list)
    by_model = defaultdict(list)

    for r in records:
        if r.get("score_orientation") is None:
            continue
        by_lang[r.get("prompt_language", "?")].append(r["score_orientation"])
        by_cat[r.get("prompt_category", "?")].append(r["score_orientation"])
        by_model[r.get("model", "?")].append(r["score_orientation"])

    print("Mean orientation score by language (2=defense, 0=mechanism):")
    for lang, scores in sorted(by_lang.items()):
        avg = sum(scores) / len(scores)
        print(f"  {lang:6s}: {avg:.2f}  (n={len(scores)})")

    print("\nMean orientation score by category:")
    for cat, scores in sorted(by_cat.items()):
        avg = sum(scores) / len(scores)
        print(f"  {cat:25s}: {avg:.2f}  (n={len(scores)})")

    print("\nMean orientation score by model:")
    for model, scores in sorted(by_model.items()):
        avg = sum(scores) / len(scores)
        print(f"  {model:15s}: {avg:.2f}  (n={len(scores)})")

    print("\nFor the full calibration-gap comparison against English baselines,")
    print("see results/reports/calibration_gap_report.txt (from score_results.py).")


if __name__ == "__main__":
    main()
