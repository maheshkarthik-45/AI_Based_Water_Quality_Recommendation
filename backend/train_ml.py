import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================
# PATHS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

DATASET = BASE_DIR / "dataset" / "water_quality_india.csv"
MODEL_DIR = BASE_DIR / "model"

MODEL_DIR.mkdir(exist_ok=True)

# =====================================
# LOAD DATASET
# =====================================

print("Loading Dataset...")

df = pd.read_csv(DATASET)

print("Dataset Loaded Successfully")
print(df.head())

# =====================================
# FEATURES
# =====================================

X = df[
    [
        "pH",
        "Dissolved_Oxygen",
        "BOD",
        "COD",
        "TDS",
        "Conductivity",
        "Turbidity"
    ]
]

y = df["Quality"]

# =====================================
# LABEL ENCODING
# =====================================

encoder = LabelEncoder()

y = encoder.fit_transform(y)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================
# RANDOM FOREST
# =====================================

print("\nTraining Random Forest...\n")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================
# EVALUATION
# =====================================

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("\n===================================")
print("Random Forest Accuracy")
print("===================================")
print(f"{accuracy*100:.2f}%")

print("\nClassification Report\n")
print(classification_report(y_test, prediction))

# =====================================
# SAVE MODEL
# =====================================

joblib.dump(model, MODEL_DIR / "random_forest_model.pkl")
joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

print("\nRandom Forest Model Saved!")

print("random_forest_model.pkl")
print("label_encoder.pkl")