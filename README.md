# Multilingual Safety Probe: FIVB

**Framing Invariance Vulnerability Benchmark**

Building a benchmark to test whether AI safety calibration transfers across African languages, using controlled matched prompts to isolate safety-specific effects from language differences.

## What this project actually is

Most multilingual safety evaluation compares model responses across languages without a non-safety baseline to check against. That makes it impossible to tell a genuine safety calibration gap apart from ordinary language-level differences in response style, differences that show up on harmless topics too, and have nothing to do with safety.

This project is built specifically to make that distinction: matched English/target-language prompt pairs, back-translated for verification, run with multiple trials for statistical reliability, and tested against matched non-harmful control topics using identical structure. Pilot testing has already shown why this matters: automated keyword-based scoring can produce a misleading signal that only careful, matched-pair, control-tested, hand-verified analysis catches. See FIVB_README.md for the full methodology and current status.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env  # add your API keys and model identifiers
python run_battery.py --tier 1 --lang ha --models model_a --dry-run
python run_battery.py --tier 1 --lang ha --models model_a
python score_results.py --lang ha
```

See QUICKSTART.md and FIVB_README.md for full documentation.

## Current status

The methodology has been built and piloted in Hausa. It is under active validation, not a completed or confirmed finding. See FIVB_README.md for what has been tested, what the pilot testing showed, and why rigorous control-topic comparison and native-speaker validation are essential before any result from this methodology should be treated as established.

## Prompt access

The prompt battery is not included in this public repository. Publishing the prompts would let anyone patch a surface symptom for specific prompts without addressing any underlying calibration issue, and in high-stakes categories would put dual-use content in a public, indexed location. The battery is available to model developers through responsible disclosure and to verified researchers on request. See `battery/README.md`.