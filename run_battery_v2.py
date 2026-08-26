#!/usr/bin/env python3
"""
Runner for the Finding B Verification Battery v2.

Each line in the battery is a prompt_pair_v2 record containing both the
English and Hausa prompt together (not separate files). This script runs
both sides of each pair, saves results in the same results/raw/ format
run_battery.py and view_result.py already use, so you can read them with:

    python view_result.py --pair v2_sim_protective

Usage:
    python run_battery_v2.py --models model_a --trials 3
    python run_battery_v2.py --models model_a --trials 3 --pair v2_sim_protective
    python run_battery_v2.py --dry-run
"""

import json
import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BATTERY_FILE = Path("battery/finding_b_main_v3.jsonl")
RESULTS_DIR = Path("results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AVAILABLE_MODELS = {
    "model_a": os.environ.get("MODEL_A", "SET_ME_IN_ENV"),
    "model_b": os.environ.get("MODEL_B", "SET_ME_IN_ENV"),
    "model_c": os.environ.get("MODEL_C", "SET_ME_IN_ENV"),
    "model_d": os.environ.get("MODEL_D", "SET_ME_IN_ENV"),
}

MODEL_PROVIDER = {
    "model_a": "groq",
    "model_b": "groq",
    "model_c": "anthropic",
    "model_d": "openai",
}

SYSTEM_PROMPT = "You are a helpful assistant. Answer questions clearly and completely."
MAX_TOKENS = 2048
DELAY = 1.5


def load_pairs(path: Path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") in ("prompt_pair_v2", "prompt_pair_v3"):
                    pairs.append(obj)
            except json.JSONDecodeError:
                continue
    return pairs


def result_path(prompt_id: str, model_key: str, trial: int = 1) -> Path:
    suffix = "" if trial == 1 else f"__trial{trial}"
    return RESULTS_DIR / f"{prompt_id}__{model_key}{suffix}.json"


def already_done(prompt_id: str, model_key: str, trial: int = 1) -> bool:
    return result_path(prompt_id, model_key, trial).exists()


def get_client(provider: str):
    if provider == "groq":
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise EnvironmentError("GROQ_API_KEY not set")
        return Groq(api_key=key)
    elif provider == "anthropic":
        import anthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        return anthropic.Anthropic(api_key=key)
    elif provider == "openai":
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError("OPENAI_API_KEY not set")
        return OpenAI(api_key=key)
    raise ValueError(f"Unknown provider: {provider}")


def call_model(client, provider: str, model_name: str, text: str):
    if provider == "anthropic":
        r = client.messages.create(
            model=model_name, max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": text}])
        return r.content[0].text, {
            "input_tokens": r.usage.input_tokens,
            "output_tokens": r.usage.output_tokens}
    else:
        r = client.chat.completions.create(
            model=model_name, max_tokens=MAX_TOKENS, temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": text}])
        return r.choices[0].message.content, {
            "input_tokens": r.usage.prompt_tokens,
            "output_tokens": r.usage.completion_tokens}


def run_one_side(pair_record, lang, prompt_text, model_key, provider, model_name,
                  client, trial, dry_run):
    prompt_id = f"{pair_record['pair_id']}_{lang}"

    if dry_run:
        print(f"  [{lang}] {prompt_id} — {len(prompt_text)} chars")
        return

    if already_done(prompt_id, model_key, trial):
        print(f"  [{lang}] {prompt_id} trial {trial} — cached, skipping")
        return

    start = time.time()
    try:
        response_text, usage = call_model(client, provider, model_name, prompt_text)
        error = None
    except Exception as e:
        response_text = ""
        usage = {}
        error = str(e)
    elapsed = round(time.time() - start, 2)

    result = {
        "prompt_id": prompt_id,
        "pair_id": pair_record["pair_id"],
        "prompt_category": pair_record["category"],
        "prompt_language": lang,
        "prompt_framing": pair_record["framing"],
        "is_control": pair_record.get("is_control", False),
        "model": model_key,
        "model_name": model_name,
        "trial": trial,
        "response": response_text,
        "response_length": len(response_text),
        "usage": usage,
        "elapsed_seconds": elapsed,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score_orientation": None,
        "score_specificity": None,
        "prompt_text_access": "Available through responsible disclosure. See repository owner's profile.",
    }

    for k in ("prompt", "prompt_en", "prompt_ha", "prompt_text"):
        assert k not in result, f"SECURITY: '{k}' must never be stored in results"

    path = result_path(prompt_id, model_key, trial)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    status = f"ERROR: {error}" if error else f"saved ({elapsed}s, {usage.get('output_tokens','?')} tok)"
    print(f"  [{lang}] {prompt_id} trial {trial} — {status}")

    if not dry_run:
        time.sleep(DELAY)


def main():
    ap = argparse.ArgumentParser(description="Finding B Verification Battery v2 Runner")
    ap.add_argument("--models", nargs="+", default=["model_a"], choices=list(AVAILABLE_MODELS.keys()))
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--pair", type=str, help="Run only this pair_id")
    ap.add_argument("--battery", type=str, default=None, help="Override the default battery file path")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    battery_path = Path(args.battery) if args.battery else BATTERY_FILE

    if not battery_path.exists():
        print(f"ERROR: {battery_path} not found. Save the battery there first.")
        sys.exit(1)

    pairs = load_pairs(battery_path)
    if args.pair:
        pairs = [p for p in pairs if p["pair_id"] == args.pair]
    if not pairs:
        print("No pairs match.")
        sys.exit(0)

    print(f"\n{'='*60}")
    print("  Finding B Verification Battery v2")
    print(f"{'='*60}")
    print(f"  Pairs   : {len(pairs)}")
    print(f"  Models  : {args.models}")
    print(f"  Trials  : {args.trials}")
    if args.dry_run:
        print("  MODE    : DRY RUN")
    print(f"{'='*60}\n")

    for model_key in args.models:
        provider = MODEL_PROVIDER.get(model_key, "groq")
        model_name = AVAILABLE_MODELS[model_key]
        client = None if args.dry_run else get_client(provider)

        for pair_record in pairs:
            print(f"\nPAIR: {pair_record['pair_id']} ({pair_record['category']}, {pair_record['framing']})")
            for trial in range(1, args.trials + 1):
                run_one_side(pair_record, "en", pair_record["prompt_en"],
                             model_key, provider, model_name, client, trial, args.dry_run)
                run_one_side(pair_record, "ha", pair_record["prompt_ha"],
                             model_key, provider, model_name, client, trial, args.dry_run)

    print(f"\n{'='*60}")
    print("  Done. Read results with:")
    print("    python view_result.py --pair <pair_id>")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()