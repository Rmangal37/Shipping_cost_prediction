🚚 Shipping Cost Prediction System

An end-to-end Machine Learning application designed to predict shipping costs dynamically using advanced regression models like XGBoost and CatBoost.

This project replaces traditional static pricing formulas with a data-driven system capable of handling complex, non-linear relationships between shipment features and cost.

📌 Project Overview

Shipping pricing is often calculated using fixed formulas that fail to capture real-world complexity.

This system solves that problem using machine learning.

It takes input features such as:

Weight

Height

Width

Artist

Material

Transport Type

And predicts the final shipping cost.

🎯 Goal

To improve pricing accuracy using powerful gradient boosting models trained on structured data.

🧠 Key Features

Automated Data Pipeline (MongoDB → Model)

Data validation and drift detection

Feature transformation and preprocessing

Model training with XGBoost & CatBoost

Model comparison and best-model selection

FastAPI backend for serving predictions

Simple HTML/JS frontend for user interaction

Cloud-ready deployment using Docker & AWS S3

🏗️ System Architecture
1️⃣ Training Workflow

Data pulled from MongoDB

Schema validation and drift detection

Data cleaning and transformation

Model training (XGBoost / CatBoost)

Model evaluation using RMSE / MAE

Best model saved and deployed

Final artifacts:

shipping_preprocessor.pkl

shipping_price_model.pkl

2️⃣ Prediction Workflow

User fills form in UI

JavaScript sends JSON request to FastAPI

FastAPI loads preprocessor and trained model

Input data is transformed

Model predicts shipping cost

Prediction returned to user

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

Model Monitoring

Evidently (Data Drift Detection)

Cloud & Deployment

Docker

AWS S3 (via Boto3)

Frontend

HTML

CSS

JavaScript

📂 Project Structure
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

🔍 Core Components Explained
🔹 training_pipeline.py

Orchestrates the entire ML workflow:
Ingestion → Validation → Transformation → Training → Evaluation → Deployment.

🔹 data_ingestion.py

Connects to MongoDB and splits data into train/test sets.

🔹 data_validation.py

Ensures data schema consistency and checks for data drift.

🔹 data_transformation.py

Handles:

Missing values

Categorical encoding

Feature scaling

Saves preprocessor.pkl

🔹 model_trainer.py

Trains XGBoost and CatBoost models and selects the best performer.

🔹 model_evaluation.py

Compares the new model with the currently deployed model.

🔹 model_pusher.py

Pushes the better-performing model to deployment directory or S3.

🔹 model_predictor.py

Used during inference to load model and generate predictions.

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/Rmangal37/Shipping_cost_prediction.git
cd Shipping_cost_prediction

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate


Mac/Linux:

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

▶️ Running the Application
Start FastAPI Server
uvicorn app:app --reload


Visit:

http://127.0.0.1:8000

Trigger Training Manually
GET /train

Make Prediction
POST /predict


Submit form data to receive shipping cost prediction.

📊 Machine Learning Approach

This is a Regression Problem because the target variable (Shipping Cost) is continuous.

Why XGBoost / CatBoost?

Excellent performance on tabular data

Handles non-linear feature interactions

Strong regularization

Efficient training

Models are evaluated using:

RMSE

MAE

Best model is automatically selected.

🚀 Future Improvements

CI/CD pipeline integration

Automated retraining schedule

Real-time data monitoring dashboard

Deployment on AWS EC2 or Kubernetes

Add pricing explainability (SHAP values)

🎯 Use Cases

Logistics companies

E-commerce platforms

Shipping marketplaces

Dynamic pricing systems

👨‍💻 Author

Ruchir Mangal
Computer Engineering Student | Machine Learning Enthusiast

GitHub:
https://github.com/Rmangal37
