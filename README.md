# Cancer Prediction

A Flask web app that predicts whether a breast tumor is benign or malignant
using a Logistic Regression model trained on the scikit-learn breast cancer
dataset.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## Deploy

This app needs a Python host (not a static host like GitHub Pages), since it
runs a Flask server. It's ready to deploy on Render, Railway, or similar:
start command is `gunicorn app:app --bind 0.0.0.0:$PORT` (see `Procfile`).
