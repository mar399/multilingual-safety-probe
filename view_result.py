#!/usr/bin/env python3
"""
FIVB Result Viewer — human-readable inspection.

Reads raw result JSON files (which store responses as a single escaped
string with literal \\n characters) and prints them with real line
breaks, clearly separating the reasoning chain from the final answer,
so you can judge orientation yourself without relying on any scorer.

Usage:
    python view_result.py id_ha_002                    # all trials for this prompt_id
    python view_result.py id_ha_002 --model model_a     # filter by model
    python view_result.py id_ha_002 --final-only        # skip the <think> block
    python view_result.py --pair id_002                 # show EN and target-language side by side
    python view_result.py --list                        # list all available prompt_ids
"""

import json
import argparse
from pathlib import Path

RAW_DIR = Path("results/raw")


def load_matching(prompt_id: str, model: str = None) -> list:
    matches = []
    for path in sorted(RAW_DIR.glob(f"{prompt_id}__*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except Exception:
            continue
        if record.get("prompt_id") != prompt_id:
            continue
        if model and record.get("model") != model:
            continue
        record["_file"] = path.name
        matches.append(record)
    return matches


def split_thinking(response: str):
    if "<think>" in response and "</think>" in response:
        start = response.find("<think>") + len("<think>")
        end = response.find("</think>")
        thinking = response[start:end].strip()
        final = response[end + len("</think>"):].strip()
        return thinking, final
    return None, response.strip()


def print_record(record: dict, final_only: bool = False, width: int = 100):
    print("\n" + "=" * width)
    print(f"FILE:     {record.get('_file')}")
    print(f"PROMPT ID: {record.get('prompt_id')}   LANGUAGE: {record.get('prompt_language')}   "
          f"MODEL: {record.get('model')}   TRIAL: {record.get('trial', 1)}")
    print(f"CATEGORY: {record.get('prompt_category')}")
    print("=" * width)

    response = record.get("response", "")
    thinking, final = split_thinking(response)

    if thinking and not final_only:
        print("\n--- REASONING (model deciding how to respond — not the answer) ---\n")
        print(thinking)

    print("\n--- FINAL ANSWER (what the model actually said) ---\n")
    print(final if final else "(empty)")
    print()


def list_prompt_ids():
    ids = set()
    for path in RAW_DIR.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
            ids.add((record.get("prompt_id"), record.get("prompt_language")))
        except Exception:
            continue
    for pid, lang in sorted(ids):
        print(f"  {pid}  ({lang})")


def show_pair(pair_id: str, model: str = None, final_only: bool = False):
    """
    Find and print the English and target-language versions of a pair
    side by side (one after another), for the given pair_id pattern.
    Tries both known ID conventions: '<pair>_en' and 'id_en_<num>' style.
    """
    all_files = list(RAW_DIR.glob("*.json"))
    en_candidates, other_candidates = [], []

    for path in all_files:
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
        except Exception:
            continue
        pid = record.get("prompt_id", "")
        lang = record.get("prompt_language", "")
        parts = pid.split("_")
        stripped = "_".join(p for p in parts if p not in {"en", "ha", "yo", "ig", "pcm"})
        if stripped != pair_id:
            continue
        if model and record.get("model") != model:
            continue
        record["_file"] = path.name
        if lang == "en":
            en_candidates.append(record)
        else:
            other_candidates.append(record)

    if not en_candidates and not other_candidates:
        print(f"No results found for pair_id '{pair_id}'. Try --list to see available prompt_ids.")
        return

    print(f"\n########## PAIR: {pair_id} ##########")
    for r in en_candidates:
        print_record(r, final_only=final_only)
    for r in other_candidates:
        print_record(r, final_only=final_only)


def main():
    ap = argparse.ArgumentParser(description="FIVB Result Viewer")
    ap.add_argument("prompt_id", nargs="?", help="prompt_id to view (e.g. id_ha_002)")
    ap.add_argument("--model", type=str, help="filter to a specific model alias")
    ap.add_argument("--final-only", action="store_true", help="skip the reasoning block, show only the final answer")
    ap.add_argument("--pair", type=str, help="show EN + target-language versions of a pair_id together")
    ap.add_argument("--list", action="store_true", help="list all available prompt_ids")
    args = ap.parse_args()

    if not RAW_DIR.exists() or not any(RAW_DIR.glob("*.json")):
        print("No results found. Run run_battery.py first.")
        return

    if args.list:
        list_prompt_ids()
        return

    if args.pair:
        show_pair(args.pair, model=args.model, final_only=args.final_only)
        return

    if not args.prompt_id:
        print("Provide a prompt_id, or use --pair <id> or --list. See --help.")
        return

    records = load_matching(args.prompt_id, args.model)
    if not records:
        print(f"No results found for prompt_id '{args.prompt_id}'. Try --list.")
        return

    for r in records:
        print_record(r, final_only=args.final_only)


if __name__ == "__main__":
    main()