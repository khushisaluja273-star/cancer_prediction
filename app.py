from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

data = load_breast_cancer()
X = data.data
y = data.target
FEATURE_NAMES = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = LogisticRegression(max_iter=5000)
model.fit(X_train_scaled, y_train)

FEATURE_RANGES = {
    "mean radius":              {"min": 6.98,   "max": 28.11,   "mean": 14.13},
    "mean texture":             {"min": 9.71,   "max": 39.28,   "mean": 19.29},
    "mean perimeter":           {"min": 43.79,  "max": 188.50,  "mean": 91.97},
    "mean area":                {"min": 143.50, "max": 2501.00, "mean": 654.89},
    "mean smoothness":          {"min": 0.0526, "max": 0.1634,  "mean": 0.0964},
    "mean compactness":         {"min": 0.0194, "max": 0.3454,  "mean": 0.1043},
    "mean concavity":           {"min": 0.0,    "max": 0.4268,  "mean": 0.0888},
    "mean concave points":      {"min": 0.0,    "max": 0.2012,  "mean": 0.0489},
    "mean symmetry":            {"min": 0.1060, "max": 0.3040,  "mean": 0.1812},
    "mean fractal dimension":   {"min": 0.0500, "max": 0.0974,  "mean": 0.0628},
    "radius error":             {"min": 0.1115, "max": 2.873,   "mean": 0.4052},
    "texture error":            {"min": 0.3602, "max": 4.885,   "mean": 1.2169},
    "perimeter error":          {"min": 0.757,  "max": 21.98,   "mean": 2.8661},
    "area error":               {"min": 6.802,  "max": 542.20,  "mean": 40.337},
    "smoothness error":         {"min": 0.0017, "max": 0.0311,  "mean": 0.0070},
    "compactness error":        {"min": 0.0023, "max": 0.1354,  "mean": 0.0255},
    "concavity error":          {"min": 0.0,    "max": 0.3960,  "mean": 0.0319},
    "concave points error":     {"min": 0.0,    "max": 0.0528,  "mean": 0.0118},
    "symmetry error":           {"min": 0.0079, "max": 0.0790,  "mean": 0.0205},
    "fractal dimension error":  {"min": 0.0009, "max": 0.0298,  "mean": 0.0038},
    "worst radius":             {"min": 7.93,   "max": 36.04,   "mean": 16.27},
    "worst texture":            {"min": 12.02,  "max": 49.54,   "mean": 25.68},
    "worst perimeter":          {"min": 50.41,  "max": 251.20,  "mean": 107.26},
    "worst area":               {"min": 185.20, "max": 4254.00, "mean": 880.58},
    "worst smoothness":         {"min": 0.0712, "max": 0.2226,  "mean": 0.1324},
    "worst compactness":        {"min": 0.0273, "max": 1.0580,  "mean": 0.2543},
    "worst concavity":          {"min": 0.0,    "max": 1.2520,  "mean": 0.2722},
    "worst concave points":     {"min": 0.0,    "max": 0.2910,  "mean": 0.1146},
    "worst symmetry":           {"min": 0.1565, "max": 0.6638,  "mean": 0.2901},
    "worst fractal dimension":  {"min": 0.0550, "max": 0.2075,  "mean": 0.0839},
}


@app.route("/")
def index():
    groups = {
        "Mean Values": FEATURE_NAMES[:10],
        "Error Values": FEATURE_NAMES[10:20],
        "Worst Values": FEATURE_NAMES[20:],
    }
    return render_template("index.html", groups=groups, ranges=FEATURE_RANGES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form.get(name, 0)) for name in FEATURE_NAMES]
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        result = {
            "prediction": int(prediction),
            "label": "Benign" if prediction == 1 else "Malignant",
            "benign_prob": round(float(probability[1]) * 100, 2),
            "malignant_prob": round(float(probability[0]) * 100, 2),
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
