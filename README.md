# 🏨 Delhi Smart Hotel Price Prediction System

An end-to-end Machine Learning project that scrapes hotel listing data, builds a regression model to predict hotel prices, and deploys the model using a Streamlit web application.
## Link
Visit the Site : " https://delhi-smart-hotel-price-prediction-system-by-nabanit.streamlit.app/  "
---

## 📌 Project Overview

This project predicts the **estimated hotel price for a week** based on multiple features such as:
- District
- Transportation facilities
- Room category
- Customer score
- Number of reviews
- Bed configuration

The solution follows a complete **data science lifecycle**:
**Web Scraping → Data Preprocessing → Regression Modeling → Web Deployment**

---

## 🕸️ Data Collection (Web Scraping)

- Hotel data was collected from online hotel listing platforms using web scraping techniques.
- Extracted attributes include:
  - Location (District)
  - Transportation facilities
  - Room types
  - Customer ratings
  - Review counts
  - Bed information
- The scraped data was structured into a tabular format for analysis.

**Notebook:** `Web Scrapping1.ipynb`

---

## 🧹 Data Preprocessing & Feature Engineering

- Handled missing values and inconsistent entries
- Log-transformed skewed features (e.g., number of beds)
- Encoded categorical variables using:
  - Label Encoding
  - One-Hot Encoding
- Prepared the dataset using an **ETL pipeline**

---

## 🤖 Model Building

- Algorithm used: **Gradient Boosting Regressor**
- Reason for selection:
  - Handles non-linear relationships
  - Works well with tabular data
  - Does not require feature scaling
- Model evaluated using regression metrics (R², error-based measures)

**Notebook:** `Regression Model.ipynb`

---

## 🌐 Model Deployment (Streamlit App)

- Built an interactive web app using **Streamlit**
- Users can input hotel characteristics via a sidebar UI
- Model artifacts loaded:
  - Trained regression model
  - Feature list
  - Label encoders
- Predictions are generated in real time without retraining

**App file:** `app.py`

---

## 🚀 How to Run the Project Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
