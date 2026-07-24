# AI-Based Water Quality Prediction System

## Overview

The AI-Based Water Quality Prediction System is a web application that predicts the quality of water using Artificial Intelligence. The project uses Machine Learning (Random Forest) and Deep Learning (Artificial Neural Network) models to analyze water quality parameters and classify water into different quality categories.

The application is developed using Python, Flask, HTML, CSS, JavaScript, Scikit-learn, TensorFlow, and Keras.

---

## Features

- AI-based water quality prediction
- Machine Learning and Deep Learning models
- User-friendly web interface
- Real-time prediction
- State-wise water quality prediction
- Dashboard and result visualization

---

## Technologies Used

- Python 3.10
- Flask
- HTML5
- CSS3
- JavaScript
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- NumPy
- Joblib

---

## Dataset

Dataset Used:
Water Quality India Dataset

Input Features:

- State
- pH
- Dissolved Oxygen (DO)
- Biological Oxygen Demand (BOD)
- Chemical Oxygen Demand (COD)
- Total Dissolved Solids (TDS)
- Conductivity
- Turbidity

Output:

- Predicted Water Quality

---

## Machine Learning Model

Algorithm:
Random Forest Classifier

Saved Files:

- random_forest_model.pkl
- scaler.pkl
- label_encoder.pkl

---

## Deep Learning Model

Algorithm:
Artificial Neural Network (ANN)

Saved File:

- water_quality_nn.keras

---

## Project Structure

water/
│
├── backend/
│   ├── app.py
│   ├── train_ml.py
│   ├── train_dl.py
│   ├── dataset/
│   ├── model/
│   ├── static/
│   ├── templates/
│
└── README.md

---

## Installation

Create Virtual Environment

python -m venv venv

Activate Virtual Environment

Windows

venv\Scripts\activate

Install Required Libraries

pip install flask pandas numpy scikit-learn tensorflow joblib

---

## Train Machine Learning Model

python train_ml.py

---

## Train Deep Learning Model

python train_dl.py

---

## Run the Application

python app.py

Open the application in your browser:

http://127.0.0.1:5000

---

## Application Workflow

1. User enters water quality parameters.
2. Flask receives the input.
3. Input data is preprocessed.
4. Random Forest and ANN models perform prediction.
5. Predicted water quality is displayed to the user.

---

## Future Scope

- IoT-based real-time water monitoring
- Mobile application
- Cloud deployment
- Live dashboard
- Improved prediction accuracy using larger datasets

---

## Developed By

Project Title:
AI-Based Water Quality Prediction System Using Machine Learning and Deep Learning

Author :
Mahesh Karthik L

---
