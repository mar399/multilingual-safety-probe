#!/usr/bin/env python3
"""
FIVB Layer 1 — Probe Runner

Runs a set of prompts against configured models and records raw responses
to a CSV file for downstream analysis (see analyze_reasoning.py, score.py).

Prompts are never hardcoded in this file. They are loaded from a private,
gitignored battery file so that no research prompt text is ever committed
to the repository.

Usage:
    python probe.py --battery battery/live_probes.json
    python probe.py --battery battery/live_probes.json --mock
"""

import os
import csv
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

AVAILABLE_MODELS = {
    "model_a": os.environ.get("MODEL_A", "SET_ME_IN_ENV"),
    "model_b": os.environ.get("MODEL_B", "SET_ME_IN_ENV"),
}


def get_client():
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Run with --mock to test without an API key.")
    return Groq(api_key=api_key)


def load_probes(path: Path) -> list:
    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("This file is private and gitignored. Create it locally with your")
        print("own probe set (JSON list of {id, label, language, attack_type, messages}).")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_probes(probes: list, model_key: str, mock: bool = False) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"raw_responses_{timestamp}.csv"

    client = None if mock else get_client()
    model_name = AVAILABLE_MODELS.get(model_key, model_key)

    fieldnames = ["prompt_id", "label", "language_name", "attack_type",
                  "prompt_text", "response", "model", "timestamp"]

    print(f"\nFIVB Layer 1 — Probe Runner")
    print(f"Model: {model_key} ({model_name})")
    print(f"Mode: {'MOCK' if mock else 'REAL'}")
    print(f"Probes: {len(probes)}")
    print(f"Output: {output_path}\n")

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for probe in probes:
            print(f"Running: {probe.get('label', probe.get('id'))}")

            if mock:
                response_text = "<think>\n[mock reasoning]\n</think>\n[mock response]"
            else:
                r = client.chat.completions.create(
                    model=model_name,
                    messages=probe["messages"],
                    max_tokens=4096,
                    temperature=0.0,
                )
                response_text = r.choices[0].message.content
                time.sleep(1.5)

            writer.writerow({
                "prompt_id": probe.get("id", ""),
                "label": probe.get("label", ""),
                "language_name": probe.get("language", ""),
                "attack_type": probe.get("attack_type", ""),
                "prompt_text": probe["messages"][0]["content"] if probe.get("messages") else "",
                "response": response_text,
                "model": model_key,
                "timestamp": datetime.now().isoformat(),
            })
            f.flush()

    print(f"\nDone. Results: {output_path}")
    return output_path


def main():
    ap = argparse.ArgumentParser(description="FIVB Layer 1 Probe Runner")
    ap.add_argument("--battery", type=str, default="battery/live_probes.json",
                     help="Path to a private, gitignored JSON probe file")
    ap.add_argument("--model", type=str, default="model_a", choices=list(AVAILABLE_MODELS.keys()))
    ap.add_argument("--mock", action="store_true", help="Run without API calls, for testing")
    args = ap.parse_args()

    probes = load_probes(Path(args.battery))
    run_probes(probes, args.model, mock=args.mock)


if __name__ == "__main__":
    main()
