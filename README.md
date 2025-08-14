# Sexual Violence Against Women — Predictive ML (India NCRB)

**Purpose:** Build a transparent ML pipeline that estimates *aggregate risk levels* using NCRB data.  
**Important:** This project works at **State/UT level** only. It does **not** predict individual crimes.

## Quickstart
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python -m src.data.make_dataset
python -m src.features.build_features
python -m src.visualization.eda
python -m src.models.train_classification
python -m src.models.train_regression
python -m src.models.evaluate
```

## Repo layout
```
data/
models/
notebooks/
reports/
src/
.github/
```