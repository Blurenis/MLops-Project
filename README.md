```
 /$$      /$$ /$$                                            /$$$$$$$                                               /$$    
| $$$    /$$$| $$                                           | $$__  $$                                             | $$    
| $$$$  /$$$$| $$        /$$$$$$   /$$$$$$   /$$$$$$$       | $$  \ $$ /$$$$$$   /$$$$$$  /$$  /$$$$$$   /$$$$$$$ /$$$$$$  
| $$ $$/$$ $$| $$       /$$__  $$ /$$__  $$ /$$_____//$$$$$$| $$$$$$$//$$__  $$ /$$__  $$|__/ /$$__  $$ /$$_____/|_  $$_/  
| $$  $$$| $$| $$      | $$  \ $$| $$  \ $$|  $$$$$$|______/| $$____/| $$  \__/| $$  \ $$ /$$| $$$$$$$$| $$        | $$    
| $$\  $ | $$| $$      | $$  | $$| $$  | $$ \____  $$       | $$     | $$      | $$  | $$| $$| $$_____/| $$        | $$ /$$
| $$ \/  | $$| $$$$$$$$|  $$$$$$/| $$$$$$$/ /$$$$$$$/       | $$     | $$      |  $$$$$$/| $$|  $$$$$$$|  $$$$$$$  |  $$$$/
|__/     |__/|________/ \______/ | $$____/ |_______/        |__/     |__/       \______/ | $$ \_______/ \_______/   \___/  
                                 | $$                                               /$$  | $$                              
                                 | $$                                              |  $$$$$$/                              
                                 |__/                                               \______/                               
```

# Collaborative Sentiment Analysis Pipeline

> End‑to‑end BERT-based sentiment analysis built by **Jérémie Ondzaghe** and **Dylan Ondo**. It includes data extraction, preprocessing, model fine‑tuning, and a CLI for inference. The workflow enforces branches, pull requests, unit tests with coverage, and CI.

---

## Badges

<!-- Replace placeholders after enabling CI and coverage in your repo -->
![Build](https://img.shields.io/badge/build-passing-inactive)
![Tests](https://img.shields.io/badge/tests-100%20passed-inactive)
![Coverage](https://img.shields.io/badge/coverage-90%25%2B-inactive)

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Data](#data)
- [Pipeline Usage](#pipeline-usage)
  - [1) Data Extraction](#1-data-extraction)
  - [2) Data Processing](#2-data-processing)
  - [3) Model Training](#3-model-training)
  - [4) Inference](#4-inference)
- [Testing and Coverage](#testing-and-coverage)
- [Continuous Integration](#continuous-integration)
- [Project Management](#project-management)
- [Git Workflow and PRs](#git-workflow-and-prs)
- [Code Review Checklist](#code-review-checklist)
- [Deliverables Checklist](#deliverables-checklist)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

This repository implements a collaborative sentiment analysis pipeline using a pretrained BERT model. The pipeline has three main components:

1. **Data Extraction** — Load and validate raw text data from CSV/TSV/JSON/NDJSON.
2. **Data Processing** — Clean, normalize, and tokenize text for BERT; split into train/validation.
3. **Model Training & Inference** — Fine‑tune a BERT classifier and expose an inference CLI for predictions.

Engineering practices include feature branches, mandatory peer reviews, a Trello board for coordination, unit tests with high coverage, and CI on each push and pull request.

---

## Repository Structure

> Matches the current repo layout. Adjust if you move files.

```
.
├── README.md
├── dataset.csv
├── requrments.txt            # file name kept as in repo (typo)
├── src/
│   ├── data_extraction.py
│   ├── data_processing.py
│   ├── inference.py
│   └── model.py
└── test/
    ├── test_data_extraction.py
    ├── test_data_processing.py
    ├── test_inference.py
    └── test_model.py
```

Optional folders created during runs:

```
artifacts/            # trained models, checkpoints, logs
data/
  ├── raw/
  └── processed/
.github/workflows/    # CI config (pytest, coverage)
```

---

## Quickstart

### 1) Environment

```bash
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate
```

### 2) Install dependencies

If the repository uses `requirements.txt`:
```bash
pip install -r requirements.txt
```

If the repository currently has `requrments.txt` (typo kept intentionally to match the file):
```bash
pip install -r requrments.txt
```

---

## Data

- Expected columns in `dataset.csv`:
  - **content** — the raw text.
  - **score** — the numeric sentiment label (e.g., 0=negative, 1=neutral, 2=positive).
- Supported input formats for extraction: `.csv`, `.tsv`, `.json`, `.ndjson`.

---

## Pipeline Usage

All scripts expose a CLI via `python -m src.<module>`. Run with `-h` for all options.

### 1) Data Extraction

Load, normalize column names, and validate schema.

```bash
python -m src.data_extraction \
  --path dataset.csv \
  --text-col content \
  --label-col score \
  --out data/processed/clean.csv
```

Key behaviors:
- Accept CSV/TSV/JSON/NDJSON
- Normalize column names (snake_case)
- Validate required columns
- Fail fast with clear error messages

### 2) Data Processing

Clean text, tokenize using a Hugging Face tokenizer, and split into train/validation.

```bash
python -m src.data_processing \
  --path data/processed/clean.csv \
  --text-col content \
  --label-col score \
  --model-name bert-base-uncased \
  --val-size 0.1 \
  --out data/processed/tokenized
```

Typical steps:
- Basic cleaning (lowercasing, normalization)
- Tokenization via `AutoTokenizer`
- Train/val split saved to disk

### 3) Model Training

Fine‑tune a BERT classifier and save artifacts.

```bash
python -m src.model \
  --train data/processed/tokenized/train.json \
  --val   data/processed/tokenized/val.json \
  --model bert-base-uncased \
  --epochs 3 \
  --batch-size 32 \
  --lr 2e-5 \
  --max-length 128 \
  --output artifacts/bert-sentiment
```

Typical outputs:
- Best model checkpoint in `artifacts/`
- Training logs and metrics (accuracy, F1)

### 4) Inference

Predict sentiment for new text with a saved checkpoint.

```bash
python -m src.inference \
  --model artifacts/bert-sentiment \
  --text "I loved this movie!"
```

Expected output:
```
label: POSITIVE
score: 0.98
```

---

## Testing and Coverage

Unit tests cover extraction, processing, model, and inference.

```bash
pytest -q
```

Generate coverage and fail if below 90%:

```bash
pytest --maxfail=1 --disable-warnings -q --cov=src --cov-report=term-missing --cov-fail-under=90
coverage html  # ./htmlcov/index.html
```

Tips:
- Keep tests deterministic.
- Add regression tests for bug fixes.
- Mock I/O and randomness where helpful.

---

## Continuous Integration

Enable GitHub Actions to run tests and coverage on every push and PR. Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install deps
        run: |
          if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
          if [ -f requrments.txt ]; then pip install -r requrments.txt; fi
          pip install pytest pytest-cov
      - name: Run tests with coverage
        run: |
          pytest --maxfail=1 -q --cov=src --cov-report=xml --cov-report=term-missing --cov-fail-under=90
      - name: Upload coverage artifact
        uses: actions/upload-artifact@v4
        with:
          name: coverage-xml
          path: coverage.xml
```

Optional extensions:
- Linting (ruff/flake8) and type checks (mypy).
- Coverage badge via a coverage service.

---

## Project Management

Use a Trello board named: **“Sentiment Analysis Project – Jérémie Ondzaghe & Dylan Ondo”** with lists:

- **To Do** — backlog of tasks.
- **In Progress** — assigned and active.
- **In Review** — awaiting code review via PR.
- **Done** — merged and validated.

Each card includes:
- **Description** with goal and acceptance criteria.
- **Checklist**: unit tests pass, coverage ≥ 90%, README updated, attach PR link.
- **Assignees and Labels**: `backend`, `data`, `model`, `testing`, `documentation`.
- **Attachments**: PR links, test artifacts, coverage report screenshot.

Workflow:
1. Create a card in **To Do**.
2. Move to **In Progress** when a feature branch starts.
3. Move to **In Review** with a link to the PR.
4. Move to **Done** after approval, merge, and green CI.

---

## Git Workflow and PRs

- Default branch: `main` is protected. No direct commits.
- Branch naming:
  - `feature/data-extraction`, `feature/data-processing`, `feature/model-training`, `feature/inference`
  - `chore/ci`, `docs/readme`, `fix/<issue-id>-<short-desc>`
- Commit messages: imperative mood and scoped, e.g. `feat(processing): add tokenizer and split`.

Example:

```bash
git checkout -b feature/data-extraction
# ... work, commit ...
git push -u origin feature/data-extraction
# open PR to main and request review
```

PR rules:
- CI green.
- Coverage ≥ 90%.
- At least one approval from the partner.
- Rebase to resolve conflicts if needed.

---

## Code Review Checklist

Paste this in PR comments:

- [ ] Scope clear and minimal.
- [ ] Tests cover new code; all tests green.
- [ ] Coverage ≥ 90% on project and diff.
- [ ] Names and structure clear; functions small.
- [ ] No secrets, hard-coded paths, or dead code.
- [ ] README/docs updated where needed.

---

## Deliverables Checklist

- [ ] Public repository with branches and PRs history.
- [ ] This README with setup and usage.
- [ ] Screenshots: Trello board (lists, cards, attachments).
- [ ] Screenshots: GitHub PRs and code review comments.
- [ ] Test results and coverage report (≥ 90%).

---

## Contributors

- **Jérémie Ondzaghe** — data extraction, data processing, model training, repository setup.
- **Dylan Ondo** — unit tests, CI integration, documentation checks, peer reviews.

Both collaborators review and validate each other’s work before merge.

---

## License

This project is released under **The Unlicense (public domain)**. See `LICENSE`.
```
SPDX-License-Identifier: Unlicense
```
