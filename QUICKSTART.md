# Quick Start

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Configure

```bash
cp .env.example .env
```

Edit `.env` and add your API key(s) and the model identifiers you want to test. This tool is provider-agnostic — it works with any inference provider whose SDK you configure in the model-loading functions.

## 3. Get the prompt battery

The prompt battery is not included in this repository (see `battery/README.md` for why). Request access, then place the battery files in `battery/`.

## 4. Dry run

Confirm everything loads without making API calls:

```bash
python run_battery.py --tier 1 --lang ha --models model_a --dry-run
```

## 5. Run live

```bash
python run_battery.py --tier 1 --lang ha --models model_a
```

## 6. Score results

```bash
python score_results.py --lang ha
```

This produces `results/reports/calibration_gap_report.txt` with orientation, technique-count, and specificity scores compared between English and the target language.

## 7. Responsible disclosure

If your results include `publication_track: responsible_disclosure` categories, do not publish full response content. See the protocol in `FIVB_README.md`.

## File safety

- Prompt text is never written into any result file — only category, language, and score metadata.
- `results/` and `battery/*.jsonl` are gitignored by default. Do not remove these entries.
- Before pushing any changes, run a search for API keys, prompt text, and result files to confirm nothing sensitive is staged.
