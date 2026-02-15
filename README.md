🚚 Shipping Cost Prediction System

An end-to-end Machine Learning application that predicts shipping costs dynamically using advanced regression models like XGBoost and CatBoost.

This project replaces traditional static pricing formulas with a data-driven system capable of handling complex, non-linear relationships between shipment features and cost.  


📌 Project Overview

Shipping cost calculation in many systems relies on fixed formulas. These often fail to capture real-world complexity such as material type, artist category, transport type, and feature interactions.

This system uses machine learning to provide more accurate and scalable pricing.  



🎯 Objective

To build a production-style ML pipeline that:

Automates data ingestion and validation

Trains multiple models

Selects the best-performing model

Serves predictions via a FastAPI backend

Provides a simple frontend for users

🧠 Key Features

Automated Data Pipeline (MongoDB → Model)

Data validation and drift detection

Feature transformation and preprocessing

Model training with XGBoost & CatBoost

Model comparison and automatic best-model selection

FastAPI backend for serving predictions

Simple HTML/JS frontend for end-users

Docker and AWS S3 ready deployment

🏗️ System Architecture
🔹 High-Level Architecture Diagram
flowchart TD

%% ================= TRAINING PIPELINE =================
subgraph Training_Pipeline

A[MongoDB Database] --> B[Data Ingestion]
B --> C[Data Validation<br/>Schema Check + Data Drift]
C --> D[Data Transformation<br/>Encoding + Scaling]
D --> E[Model Trainer<br/>XGBoost / CatBoost]
E --> F[Model Evaluation<br/>RMSE / MAE]
F --> G[Model Pusher<br/>Save Best Model<br/>Local / AWS S3]

end

%% ================= DEPLOYMENT FLOW =================
subgraph Deployment_Pipeline

U[User - Web UI] --> H[HTML / CSS / JavaScript]
H --> I[FastAPI Backend<br/>app.py]
I --> J[Model Predictor]
J --> K[Load Preprocessor + Model]
K --> L[Predict Shipping Cost]
L --> U

end

🔍 Architecture Explanation
🧠 Training Flow

Data is pulled from MongoDB.

Data validation ensures schema consistency and detects data drift.

Data transformation handles:

Missing values

Categorical encoding

Feature scaling

Models (XGBoost & CatBoost) are trained.

Performance is evaluated using RMSE and MAE.

The best-performing model is saved and pushed for deployment.

🚀 Prediction Flow

User enters shipment details in the web interface.

FastAPI receives the request.

The saved preprocessor transforms the input.

The trained model predicts the shipping cost.

The prediction is returned instantly to the user.

🛠️ Tech Stack
Language

Python 3.8+

Backend

FastAPI

Uvicorn

Database

MongoDB (via PyMongo)

Machine Learning

XGBoost

CatBoost

Scikit-learn

Data Processing

Pandas

NumPy

Monitoring

Evidently (Data Drift Detection)

Cloud & Infrastructure

Docker

AWS S3 (via Boto3)

Frontend

HTML

CSS

JavaScript

📂 Project Structure
```
Shipping_cost_prediction/
│
├── shipment/
│   ├── pipeline/
│   │   └── training_pipeline.py
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   ├── model_pusher.py
│   │   └── model_predictor.py
│
├── app.py
├── requirements.txt
└── README.md
```

🔎 Core Components Explained
training_pipeline.py

Orchestrates the complete ML workflow:
Ingestion → Validation → Transformation → Training → Evaluation → Deployment.

data_ingestion.py

Connects to MongoDB and splits data into train/test datasets.

data_validation.py

Checks schema consistency and monitors data drift.

data_transformation.py

Cleans data

Encodes categorical features

Scales numerical features

Saves preprocessor.pkl

model_trainer.py

Trains XGBoost and CatBoost models and selects the best performer.

model_evaluation.py

Compares newly trained model with currently deployed model.

model_pusher.py

Deploys the better-performing model to local storage or AWS S3.

model_predictor.py

Loads trained model and preprocessor during inference.

⚙️ Installation
1️⃣ Clone the repository
```
git clone https://github.com/Rmangal37/Shipping_cost_prediction.git
cd Shipping_cost_prediction
```

2️⃣ Create Virtual Environment

Windows:
```
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:
```
source venv/bin/activate
```

3️⃣ Install Dependencies
```
pip install -r requirements.txt
```

▶️ Running the Application
Start FastAPI Server
```
uvicorn app:app --reload
```

Open in browser:
```
http://127.0.0.1:8000
```
Trigger Training
GET /train

Make Prediction
POST /predict


Submit form data to receive the predicted shipping cost.



📊 Machine Learning Approach

This is a Regression Problem because shipping cost is a continuous variable.

Why XGBoost & CatBoost?

Strong performance on structured/tabular data

Handles non-linear relationships well

Built-in regularization

High accuracy and efficiency

Models are evaluated using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

The best-performing model is automatically selected and deployed.

🚀 Future Improvements

Automated retraining pipeline

CI/CD integration

Model explainability using SHAP

Real-time monitoring dashboard

Deployment on AWS EC2 or Kubernetes

🎯 Use Cases

Logistics companies

E-commerce platforms

Dynamic pricing engines

Shipping marketplaces

👨‍💻 Author

Ruchir Mangal
Computer Engineering Student | Machine Learning Enthusiast

GitHub:
https://github.com/Rmangal37
