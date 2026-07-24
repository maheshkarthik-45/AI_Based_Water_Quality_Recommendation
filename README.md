# 💧 AI-Based Water Quality Prediction System

## Overview

The **AI-Based Water Quality Prediction System** is an intelligent web application developed using **Machine Learning** and **Deep Learning** techniques to predict water quality based on various physicochemical parameters.

The system uses a **Random Forest Classifier** and an **Artificial Neural Network (ANN)** to analyze water quality parameters and classify the water quality. A user-friendly web interface built with **Flask** allows users to enter water quality parameters and obtain instant prediction results.

---

## Features

- AI-based Water Quality Prediction
- Machine Learning (Random Forest)
- Deep Learning (Artificial Neural Network)
- Flask Web Application
- Responsive User Interface
- Real-time Prediction
- Easy-to-use Dashboard
- State-wise Water Quality Selection

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Flask | Backend Framework |
| HTML5 | Web Page Structure |
| CSS3 | Styling |
| JavaScript | Frontend Interactivity |
| Scikit-learn | Machine Learning |
| TensorFlow | Deep Learning |
| Keras | Neural Network Development |
| Pandas | Data Processing |
| NumPy | Numerical Computation |
| Joblib | Model Serialization |
| Visual Studio Code | Development Environment |
| Google Colab | Model Training |

---

## Dataset

The project uses the **Water Quality India Dataset** in CSV format.

### Input Parameters

- State
- pH
- Dissolved Oxygen (DO)
- Biological Oxygen Demand (BOD)
- Chemical Oxygen Demand (COD)
- Total Dissolved Solids (TDS)
- Conductivity
- Turbidity

### Output

- Predicted Water Quality Category

---

## Project Structure

```text
AI-Based-Water-Quality-Prediction/
│
├── backend/
│   ├── app.py
│   ├── train_ml.py
│   ├── train_dl.py
│   ├── requirements.txt
│   │
│   ├── dataset/
│   │   └── water_quality_india.csv
│   │
│   ├── models/
│   │   ├── random_forest_model.pkl
│   │   ├── water_quality_nn.keras
│   │   ├── scaler.pkl
│   │   └── label_encoder.pkl
│   │
│   ├── static/
│   │   ├── style.css
│   │   ├── script.js
│   │   └── images/
│   │
│   └── templates/
│       ├── index.html
│       ├── predict.html
│       ├── result.html
│       ├── dashboard.html
│       └── about.html
│
├── README.md
└── LICENSE
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/AI-Based-Water-Quality-Prediction.git
```

### Navigate to Project Folder

```bash
cd AI-Based-Water-Quality-Prediction/backend
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## Training the Models

### Machine Learning Model

```bash
python train_ml.py
```

### Deep Learning Model

```bash
python train_dl.py
```

---

## Running the Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Application Workflow

1. Load the Water Quality Dataset.
2. Preprocess the dataset.
3. Train Random Forest and ANN models.
4. Save trained models.
5. Launch the Flask web application.
6. User enters water quality parameters.
7. The trained AI model predicts the water quality.
8. Display the prediction result on the web interface.

---

## Screenshots

- Home Page
- Prediction Page
- Prediction Result
- Dashboard

---

## Future Scope

- Real-time monitoring using IoT sensors
- Cloud deployment
- Mobile application support
- Interactive analytics dashboard
- Improved prediction accuracy using larger datasets
- Integration with smart water management systems

---

## Author

**Project Title:** AI-Based Water Quality Prediction System Using Machine Learning and Deep Learning

**Author:** Mahesh karthik L
