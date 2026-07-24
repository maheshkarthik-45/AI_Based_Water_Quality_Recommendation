from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path

# ==========================================
# APP SETUP
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

# ==========================================
# PATHS
# ==========================================

DATASET_PATH = BASE_DIR / "dataset" / "water_quality_india.csv"
MODEL_DIR = BASE_DIR / "model"

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

total_samples = len(df)
states_count = df["State"].nunique()
avg_wqi = round(df["WQI"].mean(), 2)

state_list = sorted(df["State"].dropna().unique())

# ==========================================
# LOAD MODELS
# ==========================================

rf_model = joblib.load(MODEL_DIR / "random_forest_model.pkl")

dl_model = tf.keras.models.load_model(
    MODEL_DIR / "water_quality_nn.keras"
)

scaler = joblib.load(MODEL_DIR / "scaler.pkl")

encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        total_samples=total_samples,
        states=states_count,
        avg_wqi=avg_wqi
    )

# ==========================================
# PREDICT PAGE
# ==========================================

@app.route("/predict")
def predict():

    return render_template(
        "predict.html",
        states=state_list
    )

# ==========================================
# RESULT
# ==========================================

@app.route("/result", methods=["POST"])
def result():

    state = request.form["state"]

    ph = float(request.form["ph"])
    do = float(request.form["do"])
    bod = float(request.form["bod"])
    cod = float(request.form["cod"])
    tds = float(request.form["tds"])
    conductivity = float(request.form["conductivity"])
    turbidity = float(request.form["turbidity"])

    sample = np.array([[
        ph,
        do,
        bod,
        cod,
        tds,
        conductivity,
        turbidity
    ]])

    # Machine Learning Prediction

    rf_prediction = rf_model.predict(sample)

    quality = encoder.inverse_transform(rf_prediction)[0]

    # Deep Learning Confidence

    scaled = scaler.transform(sample)

    dl_prediction = dl_model.predict(
        scaled,
        verbose=0
    )

    confidence = round(
        float(np.max(dl_prediction)) * 100,
        2
    )

    return render_template(
        "result.html",
        state=state,
        quality=quality,
        confidence=confidence
    )

# ==========================================
# DASHBOARD
# ==========================================

@app.route("/dashboard")
def dashboard():

    quality_counts = df["Quality"].value_counts().to_dict()

    return render_template(
        "dashboard.html",
        total_samples=total_samples,
        states=states_count,
        avg_wqi=avg_wqi,
        quality_counts=quality_counts
    )

# ==========================================
# ABOUT
# ==========================================

@app.route("/about")
def about():

    return render_template("about.html")

# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)