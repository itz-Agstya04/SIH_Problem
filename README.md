# OIL SIF Precursor Detection

A local prototype for turning safety-report narratives into traceable screening results, Life-Saving Rule suggestions, precursor groups, and an HSE review workflow.

## Status and limitations

This is a working prototype, not an operational safety system. It seeds clearly marked synthetic examples. The current extractor uses transparent keyword rules, and the screening score is illustrative, untrained, and uncalibrated. The Life-Saving Rule catalogue is provisional. Do not use these outputs to make safety decisions.

Production use requires OIL-approved SIF definitions, an approved rule catalogue and taxonomy, expert-labelled reports, model validation, security review, and operational governance.

## Requirements

- Windows, macOS, or Linux
- Python 3.12
- A virtual environment (the current workspace uses `.venv`)
- Packages listed in `requirements.txt`

Install packages from the project folder:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you have not created the environment yet:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run locally

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. The interactive API docs are at <http://127.0.0.1:8000/docs>.

The local SQLite database is created at `data/sif_reports.sqlite3`. Set `DATABASE_URL` to a SQLAlchemy-compatible URL to use another database. The first run inserts synthetic `DEMO-*` records. Imported CSV reports remain separate from those seed records.

## Supervised SIF model

The current trainable baseline is a TF-IDF + Logistic Regression classifier. It trains only on non-demo reports explicitly reviewed as either `sif_potential` or `not_sif_potential`; uncertain labels and synthetic demo rows are excluded. For a training request, the project requires at least 30 eligible reports total and at least 10 examples from each class. Training evaluates the model with stratified cross-validation and saves a versioned artifact under `data/models/`.

The model score is not calibrated and must not be interpreted as a probability. Cross-validation uses random folds; it does not replace temporal holdout, site holdout, independent HSE validation, or calibration. The current local database has only synthetic demo reports, so it cannot train this model yet.

- `GET /api/ml/status`: check labelled-data readiness and saved model metrics.
- `POST /api/ml/train`: train when the reviewed-label thresholds are satisfied.
- `GET /api/ml/predict/{report_id}`: run the trained model separately from the demo rules score.

The interface exposes these actions in the overview and report detail. Existing model artifacts are preserved; retraining replaces them only after the data thresholds are met.

## CSV import format

Upload a UTF-8 CSV file from the dashboard. A narrative column is required: `raw_description`, `description`, or `narrative`. These metadata columns are recognized when present: `report_id`, `timestamp` (or `date`), `site`, `location`, `activity`, and `report_type`. Duplicate IDs and rows without narratives are skipped.

Example:

```csv
report_id,timestamp,site,location,activity,report_type,raw_description
OIL-001,2026-06-20,Duliajan,Unit 2,Maintenance,Near miss,"The panel remained energized and LOTO was not verified."
```

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Project structure

- `app/main.py`: FastAPI routes, CSV ingestion, review workflow, demo seed data
- `app/engine.py`: evidence extraction, provisional rule mapping, transparent demo scoring, pattern grouping
- `app/database.py`: SQLAlchemy models and local SQLite setup
- `app/static/`: responsive dashboard, explorer, review dialog, and styles
- `tests/`: focused tests for extraction and precursor grouping
