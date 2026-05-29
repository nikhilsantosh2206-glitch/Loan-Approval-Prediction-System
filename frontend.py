import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    with open("ab_best_model.pkl", "rb") as file:
        return pickle.load(file)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# Title
st.title("🏦 Loan Approval Prediction System")
st.markdown("---")

# Input Form
with st.form("loan_form"):

    credit_history = st.selectbox(
        "Credit History",
        [0, 1],
        format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)"
    )

    property_area = st.selectbox(
        "Property Area",
        [0, 1, 2],
        format_func=lambda x: {
            0: "Rural",
            1: "Semi Urban",
            2: "Urban"
        }[x]
    )

    income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    submit = st.form_submit_button("Predict Loan Status")

# Prediction
if submit:
    try:
        features = np.array([
            [credit_history, property_area, income]
        ])

        prediction = model.predict(features)

        if int(prediction[0]) == 1:
            st.success("✅ Loan Approved")
            st.balloons()
        else:
            st.error("❌ Loan Rejected")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("Machine Learning Based Loan Approval Prediction")