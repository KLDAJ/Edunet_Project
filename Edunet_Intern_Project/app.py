import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# ===== Label Encodings used during training =====
workclass_map = {
    'Private': 3, 'Self-emp-not-inc': 5, 'Self-emp-inc': 4,
    'Federal-gov': 0, 'Local-gov': 1, 'State-gov': 6, 'Without-pay': 7, 'Never-worked': 2
}
marital_status_map = {
    'Married-civ-spouse': 2, 'Divorced': 1, 'Never-married': 3,
    'Separated': 4, 'Widowed': 5, 'Married-spouse-absent': 0
}
occupation_map = {
    'Tech-support': 12, 'Craft-repair': 0, 'Other-service': 8,
    'Sales': 10, 'Exec-managerial': 3, 'Prof-specialty': 9,
    'Handlers-cleaners': 4, 'Machine-op-inspct': 6, 'Adm-clerical': 1,
    'Farming-fishing': 2, 'Transport-moving': 13, 'Priv-house-serv': 11,
    'Protective-serv': 7, 'Armed-Forces': 14
}
relationship_map = {
    'Wife': 5, 'Own-child': 2, 'Husband': 0,
    'Not-in-family': 3, 'Other-relative': 4, 'Unmarried': 1
}
race_map = {
    'White': 4, 'Black': 0, 'Asian-Pac-Islander': 1,
    'Amer-Indian-Eskimo': 2, 'Other': 3
}
gender_map = {'Male': 1, 'Female': 0}
country_map = {
    'United-States': 37, 'India': 13, 'Mexico': 24,
    'Philippines': 29, 'Germany': 9, 'Canada': 4,
    'Others': 1  # fallback/default
}

# ===== Sidebar Inputs =====
st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 90, 30)
workclass = st.sidebar.selectbox("Workclass", list(workclass_map.keys()))
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=200000)
marital_status = st.sidebar.selectbox("Marital Status", list(marital_status_map.keys()))
occupation = st.sidebar.selectbox("Occupation", list(occupation_map.keys()))
relationship = st.sidebar.selectbox("Relationship", list(relationship_map.keys()))
race = st.sidebar.selectbox("Race", list(race_map.keys()))
gender = st.sidebar.selectbox("Gender", list(gender_map.keys()))
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 100000, 0)
education_num = st.sidebar.slider("Education Level (Num)", 1, 16, 10)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 80, 40)
country = st.sidebar.selectbox("Native Country", list(country_map.keys()))

# ===== Build Input DataFrame =====
input_df = pd.DataFrame([{
    'age': age,
    'workclass': workclass_map.get(workclass, 1),
    'fnlwgt': fnlwgt,
    'marital-status': marital_status_map.get(marital_status, 3),
    'occupation': occupation_map.get(occupation, 0),
    'relationship': relationship_map.get(relationship, 3),
    'race': race_map.get(race, 4),
    'gender': gender_map.get(gender, 1),
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'educational-num': education_num,
    'hours-per-week': hours_per_week,
    'native-country': country_map.get(country, 1)
}])

st.write("### 🔎 Input Data")
st.write(input_df)

# ===== Predict Button =====
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    label = ">50K" if prediction[0] == 1 else "≤50K"
    st.success(f"✅ Prediction: The employee earns {label}")

# ===== Batch Prediction =====
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    # Apply same encodings to batch data
    for col, mapping in {
        'workclass': workclass_map,
        'marital-status': marital_status_map,
        'occupation': occupation_map,
        'relationship': relationship_map,
        'race': race_map,
        'gender': gender_map,
        'native-country': country_map
    }.items():
        batch_data[col] = batch_data[col].map(mapping).fillna(1).astype(int)

    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = ['>50K' if p == 1 else '≤50K' for p in batch_preds]

    st.write("✅ Predictions:")
    st.write(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')

