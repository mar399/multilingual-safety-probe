# Battery Access Policy

This directory holds the FIVB prompt battery locally. The battery files themselves (`*.jsonl`) are not included in the public repository.

## Why the prompts are not public

FIVB tests whether protective-framing requests receive different safety calibration across languages. Publishing the exact prompts would let anyone patch the surface symptom for those specific prompts without addressing the underlying calibration gap — and in the highest-stakes categories, it would put dual-use content into a public, indexed location.

## Who can get access

- **Model developers**, through the responsible disclosure process, receive the specific prompts relevant to findings that affect their models.
- **Verified researchers** working on multilingual AI safety evaluation can request access for replication or extension.
- **Grant reviewers and funders** evaluating this work can request a sample sufficient to verify the methodology.

## How to request access

See the repository owner's profile for contact details.

## What you'll receive

- The full prompt battery in JSONL format, with paired English and target-language prompts, back-translations, scoring anchors, and confidence tiers.
- Instructions for running the battery with `run_battery.py`.
- The responsible disclosure protocol for any results in `publication_track: responsible_disclosure` categories.

## Local file structure (after you have access)

```
battery/
├── README.md                              ← this file (public)
├── responsible_disclosure_complete.jsonl   ← main battery (private)
├── responsible_disclosure_gaps.jsonl       ← extended categories (private)
├── FIVB_complete.jsonl                     ← paired EN+target benchmark (private)
├── framing_invariance_v6_definitive.jsonl  ← English baseline set (private)
├── jailbreak_battery.jsonl                 ← jailbreak effectiveness base requests (private)
├── technique_configs.jsonl                 ← technique-specific configurations (private)
├── live_probes.json                        ← probes for analyze_reasoning.py --live (private)
└── adversarial_probes.json                 ← probes for hausa_adversarial.py (private)
```

All of the above except this README are gitignored and will never be committed.
