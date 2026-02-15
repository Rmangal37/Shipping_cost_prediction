# Shipping Cost Prediction  
**By Akshat Garg**  

Accurately forecasting logistics expenses is a cornerstone of financial planning for any business involved in shipping goods. This project addresses the critical task of predicting shipping costs by leveraging machine learning to analyze various factors that influence freight charges. By creating a reliable predictive model, businesses can optimize their supply chain, offer competitive pricing, and improve budget accuracy.

Python libraries like **Pandas**, **NumPy**, **Scikit-learn**, and **XGBoost** are employed to process the data, train the model, and evaluate its performance in predicting shipping costs.

---

## Dataset Description  

The dataset for this project contains transactional data for various shipments, detailing the key attributes associated with each freight movement. The primary data is stored in `shipment.csv`.  

| Column Name           | Description |
|-----------------------|-------------|
| **Country**           | Destination country of the shipment. |
| **Shipment Mode**     | The method of transportation (e.g., "Air", "Ocean"). |
| **Manufacturing Site**| The location where the product was made. |
| **Weight (Kilograms)**| The gross weight of the shipment. |
| **Line Item Value**   | The total value of the items in the shipment. |
| **Pack Price**        | The price per pack of the item. |
| **Unit Price**        | The price per unit of the item. |
| **Freight Cost (USD)**| **Target variable** – total cost of the shipment in USD. |

---

## Model Selection  

To achieve the highest accuracy in predicting shipping costs, the **XGBoost Regressor** was selected as the final model.  
XGBoost (Extreme Gradient Boosting) is a powerful and efficient implementation of the gradient boosting framework. It excels at handling complex, non-linear relationships in tabular data and consistently delivers high performance through its optimized tree-based learning algorithm.

**Other models tested:**  
- Linear Regression  
- Random Forest Regressor  
- Support Vector Regressor (SVR)  
- Neural Network  

---

## Model Performance (R² Score)  

| Model                        | R² Score |
|------------------------------|----------|
| Linear Regression            | 0.78     |
| Support Vector Regressor     | 0.84     |
| Random Forest Regressor      | 0.89     |
| Neural Network (Keras)       | 0.87     |
| **XGBoost Regressor**        | **0.94** |

---

## Why the Model is Not Overfitting  

- **Cross-Validated R²:** The model achieved a consistent cross-validated R² score of **0.92**, indicating stable performance across different subsets of the data.  
- **Residuals Plot:** The residuals (differences between predicted and actual values) show a random pattern centered around zero. This confirms that the model's errors are random and not systematic — a key indicator of a well-fitted model.
