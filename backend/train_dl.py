import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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

print("Dataset Loaded Successfully!")

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
# FEATURE SCALING
# =====================================

scaler = StandardScaler()

X = scaler.fit_transform(X)

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
# BUILD MODEL
# =====================================

model = Sequential()

model.add(Dense(128, activation="relu", input_shape=(7,)))
model.add(Dropout(0.3))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(32, activation="relu"))

model.add(Dense(len(encoder.classes_), activation="softmax"))

# =====================================
# COMPILE
# =====================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================
# EARLY STOPPING
# =====================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

print("\nTraining Deep Learning Model...\n")

model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =====================================
# TEST
# =====================================

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n================================")
print(f"Deep Learning Accuracy : {accuracy*100:.2f}%")
print("================================")

# =====================================
# SAVE
# =====================================

model.save(MODEL_DIR / "water_quality_nn.keras")

joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")

print("\nDeep Learning Model Saved!")

print("water_quality_nn.keras")
print("scaler.pkl")
print("label_encoder.pkl")