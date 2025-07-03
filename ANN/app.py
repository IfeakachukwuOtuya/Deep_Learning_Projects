import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ========================
# LOAD MODELS & PIPELINE
# ========================
models = {
    "Logistic Regression": joblib.load("LogisticRegression_model.pkl"),
    "Random Forest": joblib.load("RandomForest_model.pkl"),
    "Decision Tree": joblib.load("DecisionTree_model.pkl"),
    "KNN": joblib.load("KNN_model.pkl"),
    "Naive Bayes": joblib.load("NaiveBayes_model.pkl"),
    "LightGBM": joblib.load("LightGBM_model.pkl"),
    "SVM": joblib.load("SVM_model.pkl"),
    "XGBoost": joblib.load("XGBoost_model.pkl"),
    "ANN (Keras)": load_model("ann_model.h5")
}

# ✅ Load your pre-fitted encoders & scaler
le = joblib.load("label_encoder.pkl")
ct = joblib.load("column_transformer.pkl")
sc = joblib.load("scaler.pkl")

# ========================
# STREAMLIT INTERFACE
# ========================
st.title("Churn Prediction App")
st.sidebar.title("Choose Model")
model_choice = st.sidebar.selectbox("Select a model to use", list(models.keys()))

st.write("### Enter Customer Information:")
st.sidebar.header("Input Features")

credit_score = st.sidebar.number_input("Credit Score", min_value=0)
geography = st.sidebar.selectbox("Geography", ("France", "Germany", "Spain"))
gender = st.sidebar.selectbox("Gender", ("Female", "Male"))
age = st.sidebar.number_input("Age", min_value=0)
tenure = st.sidebar.number_input("Tenure", min_value=0)
balance = st.sidebar.number_input("Balance", min_value=0.0, format="%.2f")
num_of_products = st.sidebar.number_input("Number of Products", min_value=1, max_value=4)
has_cr_card = st.sidebar.selectbox("Has Credit Card", (0, 1))
is_active_member = st.sidebar.selectbox("Is Active Member", (0, 1))
estimated_salary = st.sidebar.number_input("Estimated Salary", min_value=0.0, format="%.2f")

# Transform user input
user_data = np.array([[credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]])

# ✅ Apply the loaded transformers (NO fitting!)
user_data[:, 2] = le.transform(user_data[:, 2])  # Gender
user_data = ct.transform(user_data)              # Geography One-Hot Encoding
user_data = sc.transform(user_data)              # Scaling

# ========================
# PREDICT CHURN
# ========================
if st.button("Predict"):
    model = models[model_choice]

    if model_choice == "ANN (Keras)":
        prediction = model.predict(user_data)
        result = "Churn" if prediction[0][0] > 0.5 else "No Churn"
    else:
        prediction = model.predict(user_data)
        result = "Churn" if prediction[0] == 1 else "No Churn"

    st.success(f"Prediction is: **{result}**")

st.markdown('<h5 class="#ffff00">Made by Ifeakachukwu Otuya</h5>', unsafe_allow_html=True)
