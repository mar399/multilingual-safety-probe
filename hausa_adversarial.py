#!/usr/bin/env python3
"""
FIVB Adversarial Probe Runner

Runs a multi-technique adversarial probe set against a configured model.
Prompts are loaded from a private, gitignored file — never hardcoded here.

Usage:
    python hausa_adversarial.py --battery battery/adversarial_probes.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from groq import Groq

MODEL = os.environ.get("PROBE_MODEL", "SET_ME_IN_ENV")


def get_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)


def load_probes(path: Path) -> list:
    if not path.exists():
        print(f"ERROR: {path} not found.")
        print("This file is private and gitignored. Create it locally with your")
        print("own adversarial probe set (JSON list of {id, label, messages}).")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(client, probes: list):
    for probe in probes:
        print(f"\n=== {probe.get('label', probe.get('id'))} ===")
        response = client.chat.completions.create(
            model=MODEL,
            messages=probe["messages"],
            max_tokens=2048,
            temperature=0.0,
        )
        text = response.choices[0].message.content
        print(text[:500])
        print("..." if len(text) > 500 else "")


def main():
    ap = argparse.ArgumentParser(description="FIVB Adversarial Probe Runner")
    ap.add_argument("--battery", type=str, default="battery/adversarial_probes.json")
    args = ap.parse_args()

    client = get_client()
    probes = load_probes(Path(args.battery))
    run(client, probes)


if __name__ == "__main__":
    main()
