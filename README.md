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

End-to-end BERT-based sentiment analysis built by **Jérémie Ondzaghe** and **Dylan Ondo**.  
The project covers data ingestion, preprocessing, model fine-tuning, and a CLI for inference—now delivered as a **containerized application** with **Docker Compose** orchestration and **GitHub Actions CI/CD**.

---

## Badges

[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-dockerjeremieo1-blue?logo=docker)](https://hub.docker.com/u/dockerjeremieo1)

---

## Architecture (Part 2: Containerized + CI/CD)

This repository has migrated from a local Python/venv workflow (Part 1) to a production-style, container-first workflow (Part 2):

- **Docker**: Packages the sentiment application into a reproducible image (runtime, dependencies, entrypoints).
- **Docker Compose**: Orchestrates the stack:
  - `sentiment-app` (the Python application container)
  - `redis` (service container, used for caching/queueing depending on the pipeline configuration)
- **GitHub Actions (CI/CD)**: Automates testing, evaluation gates, and image delivery to Docker Hub.

At a high level:

1. Developers run training/inference via `docker-compose run ...`
2. CI executes tests and linting on every push
3. If tests pass, CI evaluates model quality
4. If evaluation passes, CI builds and pushes the Docker image to Docker Hub

---

## Repository Structure

> Adjust paths below if your repo layout differs.

```text
.
├── README.md
├── docker-compose.yml
├── Dockerfile
├── data/
│   └── dataset.csv
├── artifacts/                  # created at runtime (models, metrics, outputs)
├── src/
│   ├── data_extraction.py
│   ├── data_processing.py
│   ├── inference.py
│   └── model.py
├── test/
│   ├── test_data_extraction.py
│   ├── test_data_processing.py
│   ├── test_inference.py
│   └── test_model.py
└── .github/
    └── workflows/
        ├── test.yml
        ├── evaluate.yml
        └── build-and-push.yml
```

---

## Quickstart (Docker Compose)

### Prerequisites

- Docker Engine + Docker Compose v2
- (Optional) NVIDIA Container Toolkit if you plan to run training on GPU

### Build the stack

```bash
docker-compose build
```

This builds the `sentiment-app` image locally using the repo’s `Dockerfile`.

---

## Data

- Default dataset location (example): `data/dataset.csv`
- Expected columns:
  - `content` — raw text input
  - `score` — numeric sentiment label (e.g., 0/1/2)

> If your dataset uses different column names, pass them explicitly via CLI flags (see training example below).

---

## Pipeline Usage (CLI via Docker Compose)

All pipeline modules are executed inside the application container using `docker-compose run --rm sentiment-app ...`.

### Model Training (example)

```bash
docker-compose run --rm sentiment-app -m src.model \
  --path /app/data/dataset.csv \
  --text content \
  --score score \
  --output /app/artifacts \
  --epochs 1
```

Notes:
- `/app/data/...` and `/app/artifacts` are container paths.
- Ensure Compose mounts local folders as volumes if you want outputs persisted on the host (commonly `./data:/app/data` and `./artifacts:/app/artifacts`).

### Inference (example)

```bash
docker-compose run --rm sentiment-app -m src.inference \
  --text "I love this product" \
  --model-dir /app/artifacts
```

Expected behavior:
- Loads the model from `--model-dir`
- Prints a predicted label/score (exact format depends on the implementation)

---

## CI/CD Automation (GitHub Actions)

The CI/CD system is split into three workflows, forming a gated pipeline:

### 1) **Test** workflow
**Trigger:** on every push (and typically on pull requests)  
**What it does:**
- Runs `pytest`
- Runs `flake8`
- Fails fast if unit tests or linting fail

### 2) **Evaluate** workflow
**Trigger:** runs after **Test** succeeds  
**What it does:**
- Executes model evaluation logic (e.g., validation metrics)
- Enforces a quality gate: **metric threshold must be > 0.70**
- Blocks delivery if the model does not meet the threshold

### 3) **Build & Push** workflow
**Trigger:** runs after **Evaluate** succeeds  
**What it does:**
- Builds the Docker image
- Pushes it to Docker Hub:

`dockerjeremieo1/sentiment-app`

> Ensure repository secrets are configured for Docker Hub authentication (e.g., `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`) as required by your workflow definitions.

---

## Deliverables / Skills Validated

This project explicitly validates the following competencies:

- **C01 — Docker**: containerized application build and runtime
- **C02 — Volumes**: persistent data/artifacts via mounted volumes
- **C03 — Compose**: multi-service orchestration (App + Redis)
- **C04 — CI/CD**: automated test → evaluate → build & push delivery pipeline

---

## Contributors

- **Jérémie Ondzaghe**
- **Dylan Ondo**

---

## License

Released under **The Unlicense (public domain)**. See `LICENSE`.

```text
SPDX-License-Identifier: Unlicense
```