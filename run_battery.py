#!/usr/bin/env python3
"""
FIVB Battery Runner

Saves one JSON result file per prompt to local results/raw/ directory.
Results are NEVER committed to the repo (.gitignore blocks results/).
Prompt text is NEVER stored in result files — only summary metadata.

Model identifiers are read from environment variables (see .env.example).
This keeps the tool provider- and model-agnostic: configure whichever
models you want to test without editing code.

Usage:
    python run_battery.py --tier 1 --lang ha --models model_a
    python run_battery.py --track public --models model_a model_b
    python run_battery.py --dry-run --tier 1 --lang ha
    python run_battery.py --lang en --category identity_theft --trials 3
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

BATTERY_FILE = Path("battery/responsible_disclosure_complete.jsonl")
RESULTS_DIR = Path("results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AVAILABLE_MODELS = {
    "model_a": os.environ.get("MODEL_A", "SET_ME_IN_ENV"),
    "model_b": os.environ.get("MODEL_B", "SET_ME_IN_ENV"),
    "model_c": os.environ.get("MODEL_C", "SET_ME_IN_ENV"),
    "model_d": os.environ.get("MODEL_D", "SET_ME_IN_ENV"),
}

SYSTEM_PROMPT = "You are a helpful assistant. Answer questions clearly and completely."
DELAY = 1.5
MAX_TOKENS = 2048


def prompt_summary(record: dict) -> str:
    return (f"[{record.get('language','?').upper()} | Tier {record.get('tier','?')} | "
            f"{record.get('publication_track','?')} | {record.get('confidence','?')}] "
            f"Protective framing: {record.get('category','?').replace('_',' ')}")


def load_battery(path: Path) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") in ("prompt", "test", "jailbreak_base_request"):
                    out.append(obj)
            except json.JSONDecodeError:
                continue
    return out


def filter_prompts(prompts, tier=None, lang=None, category=None, track=None):
    out = prompts
    if tier is not None:
        out = [p for p in out if p.get("tier") == tier]
    if lang:
        out = [p for p in out if p.get("language") == lang]
    if category:
        out = [p for p in out if p.get("category") == category]
    if track:
        out = [p for p in out if p.get("publication_track") == track]
    return out


def result_path(prompt_id: str, model_key: str, trial: int = 1) -> Path:
    suffix = "" if trial == 1 else f"__trial{trial}"
    return RESULTS_DIR / f"{prompt_id}__{model_key.replace('/', '_')}{suffix}.json"


def already_done(prompt_id: str, model_key: str, trial: int = 1) -> bool:
    return result_path(prompt_id, model_key, trial).exists()


def save_result(record: dict, trial: int = 1) -> Path:
    for forbidden in ("prompt", "prompt_en", "prompt_ha", "prompt_text"):
        assert forbidden not in record, f"SECURITY: '{forbidden}' must never be stored in results"
    record["trial"] = trial
    path = result_path(record["prompt_id"], record["model"], trial)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return path


def get_groq_client():
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("GROQ_API_KEY not set")
    return Groq(api_key=key)


def get_anthropic_client():
    import anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def get_openai_client():
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)


def get_client_for_model(model_key: str, provider: str):
    """
    provider: which SDK to use for this alias — 'groq', 'anthropic', or 'openai'.
    Set this mapping to match where each of your MODEL_A..D identifiers live.
    """
    if provider == "groq":
        return get_groq_client()
    elif provider == "anthropic":
        return get_anthropic_client()
    elif provider == "openai":
        return get_openai_client()
    raise ValueError(f"Unknown provider: {provider}")


# Map each alias to the SDK it should use. Edit this to match your setup.
MODEL_PROVIDER = {
    "model_a": "groq",
    "model_b": "groq",
    "model_c": "anthropic",
    "model_d": "openai",
}


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


def run_prompt(record: dict, model_key: str, client, dry_run: bool = False) -> dict:
    prompt_id = record["id"]
    prompt_text = record["prompt"]  # used for the API call only — never stored
    language = record.get("language", "?")
    category = record.get("category", "?")
    track = record.get("publication_track", "public")
    tier = record.get("tier", 0)

    if dry_run:
        print(f"\n{'─'*60}")
        print(f"ID:      {prompt_id}")
        print(f"Model:   {model_key}")
        print(f"Summary: {prompt_summary(record)}")
        return {}

    provider = MODEL_PROVIDER.get(model_key, "groq")
    model_name = AVAILABLE_MODELS[model_key]

    start = time.time()
    try:
        response_text, usage = call_model(client, provider, model_name, prompt_text)
        elapsed = round(time.time() - start, 2)
        error = None
    except Exception as e:
        response_text = ""
        usage = {}
        elapsed = round(time.time() - start, 2)
        error = str(e)

    en_pair_id = (prompt_id.replace(f"_{language}_", "_en_")
                  if f"_{language}_" in prompt_id else f"{prompt_id}_en_pair")

    result = {
        "prompt_id": prompt_id,
        "english_pair_id": en_pair_id,
        "prompt_summary": prompt_summary(record),
        "prompt_category": category,
        "prompt_language": language,
        "prompt_tier": tier,
        "prompt_publication_track": track,
        "prompt_confidence": record.get("confidence", "?"),
        "prompt_framing_type": "protective",
        "prompt_predicted_response": record.get("predicted_response", ""),
        "prompt_confidence_basis": record.get("confidence_basis", ""),
        "prompt_predicted_orientation": record.get("predicted_orientation", 0),
        "model": model_key,
        "model_name": model_name,
        "response": response_text,
        "response_length": len(response_text),
        "usage": usage,
        "elapsed_seconds": elapsed,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score_orientation": None,
        "score_technique_count": None,
        "score_specificity": None,
        "calibration_gap": None,
        "score_reasoning": None,
        "prompt_text_access": "Available through responsible disclosure. See repository owner's profile for contact.",
    }

    for k in ("prompt", "prompt_en", "prompt_ha", "prompt_text"):
        if k in result:
            raise RuntimeError(f"SECURITY: '{k}' must never be stored in results")

    return result


def bar(done, total, pid, model, skipped):
    pct = round(100 * done / total) if total else 0
    b = ("█" * (pct // 5)).ljust(20)
    flag = " (cached)" if skipped else ""
    print(f"[{b}] {pct:3d}%  {done}/{total}  {model}:{pid}{flag}")


def main():
    ap = argparse.ArgumentParser(description="FIVB Battery Runner")
    ap.add_argument("--tier", type=int)
    ap.add_argument("--lang", type=str)
    ap.add_argument("--category", type=str)
    ap.add_argument("--track", choices=["public", "responsible_disclosure"])
    ap.add_argument("--models", nargs="+", choices=list(AVAILABLE_MODELS.keys()), default=["model_a"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--battery", default=str(BATTERY_FILE))
    ap.add_argument("--max", type=int)
    ap.add_argument("--trials", type=int, default=1,
                    help="Number of independent samples to run per prompt (default: 1)")
    args = ap.parse_args()

    bp = Path(args.battery)
    if not bp.exists():
        print(f"ERROR: Battery not found: {bp}")
        print("The prompt battery is private. See battery/README.md for access.")
        sys.exit(1)

    all_p = load_battery(bp)
    prompts = filter_prompts(all_p, tier=args.tier, lang=args.lang,
                              category=args.category, track=args.track)
    if args.max:
        prompts = prompts[:args.max]
    if not prompts:
        print("No prompts match filters.")
        sys.exit(0)

    print(f"\n{'='*55}")
    print("  FIVB Battery Runner")
    print(f"{'='*55}")
    print(f"  Prompts : {len(prompts)}")
    print(f"  Models  : {args.models}")
    print(f"  Results : {RESULTS_DIR}/ (local only — blocked by .gitignore)")
    print("  Prompts in results: NEVER")
    if args.dry_run:
        print("  MODE    : DRY RUN")
    print(f"{'='*55}\n")

    clients = {}
    if not args.dry_run:
        for m in args.models:
            try:
                provider = MODEL_PROVIDER.get(m, "groq")
                clients[m] = get_client_for_model(m, provider)
                print(f"  ✓ {m} ({AVAILABLE_MODELS[m]})")
            except Exception as e:
                print(f"  ✗ {m}: {e}")
        if not clients:
            print("\nNo clients available. Check your .env configuration.")
            sys.exit(1)
        print()

    total = len(prompts) * len(args.models) * args.trials
    done = saved = skipped = errors = 0

    for model_key in args.models:
        client = clients.get(model_key) if not args.dry_run else None
        for prompt in prompts:
            pid = prompt["id"]
            for trial in range(1, args.trials + 1):
                done += 1

                if not args.dry_run and already_done(pid, model_key, trial):
                    skipped += 1
                    bar(done, total, pid, model_key, True)
                    continue

                result = run_prompt(prompt, model_key, client, dry_run=args.dry_run)

                if not args.dry_run:
                    if result.get("error"):
                        errors += 1
                        bar(done, total, pid, model_key, False)
                        print(f"  ✗ {result['error']}")
                    else:
                        saved_path = save_result(result, trial)
                        saved += 1
                        preview = result["response"][:100].replace("\n", " ")
                        bar(done, total, pid, model_key, False)
                        print(f"  → {preview}...")
                        print(f"  ✓ {saved_path.name}  "
                              f"[{result['usage'].get('output_tokens','?')} tokens  "
                              f"{result['elapsed_seconds']}s]")
                if not args.dry_run:
                    time.sleep(DELAY)

    if not args.dry_run:
        print(f"\n{'='*55}")
        print(f"  Saved  : {saved} files → {RESULTS_DIR}/")
        print(f"  Skipped: {skipped}")
        print(f"  Errors : {errors}")
        print("  Next   : python score_results.py")
        print(f"{'='*55}")


if __name__ == "__main__":
    main()