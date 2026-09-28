import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ABC Ltd. | Customer Decision Tool", layout="centered")

@st.cache_resource
def load_models():
    churn = joblib.load("churn_model.joblib")
    charge = joblib.load("monthly_charge_model.joblib")
    return churn, charge

try:
    churn_model, charge_model = load_models()
except Exception as exc:
    st.error("Model files could not be loaded. Check that both .joblib files are in the repository.")
    st.exception(exc)
    st.stop()

st.title("ABC Ltd. Customer Decision Tool")
st.write(
    "An educational decision-support prototype. "
    "It estimates churn risk and, separately, a monthly charge from service choices."
)

tab1, tab2 = st.tabs(["Customer churn risk", "Monthly charge estimate"])

with tab1:
    st.subheader("Estimate customer churn risk")
    st.caption("Enter the customer's current account details.")

    with st.form("churn_form"):
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input(
            "Monthly charges (dataset units)", min_value=0.0, max_value=200.0,
            value=70.0, step=1.0
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        internet_service = st.selectbox(
            "Internet service", ["DSL", "Fiber optic", "No"]
        )
        tech_support = st.selectbox(
            "Tech support", ["No", "Yes", "No internet service"]
        )
        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        churn_submit = st.form_submit_button("Estimate churn risk")

    if churn_submit:
        customer = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "Contract": contract,
            "InternetService": internet_service,
            "TechSupport": tech_support,
            "PaymentMethod": payment_method,
        }])

        probability = float(churn_model.predict_proba(customer)[0, 1])
        st.metric("Estimated churn probability", f"{probability:.1%}")

        if probability >= 0.50:
            st.warning("Flagged for a manager's review (0.50 threshold).")
        else:
            st.success("Not flagged at the 0.50 threshold.")

        st.info(
            "This is a statistical estimate, not a certain outcome or an explanation "
            "of why this individual might leave. Confirm details with the customer "
            "before taking action."
        )

with tab2:
    st.subheader("Estimate monthly charge")
    st.caption(
        "A separate linear-regression demonstration. It does not predict churn."
    )

    with st.form("charge_form"):
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple lines", ["No", "Yes", "No phone service"]
        )
        security = st.selectbox(
            "Online security", ["No", "Yes", "No internet service"]
        )
        backup = st.selectbox(
            "Online backup", ["No", "Yes", "No internet service"]
        )
        protection = st.selectbox(
            "Device protection", ["No", "Yes", "No internet service"]
        )
        support = st.selectbox(
            "Tech support", ["No", "Yes", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming movies", ["No", "Yes", "No internet service"]
        )
        charge_submit = st.form_submit_button("Estimate monthly charge")

    if charge_submit:
        services = pd.DataFrame([{
            "InternetService": internet,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": protection,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
        }])

        estimate = max(0.0, float(charge_model.predict(services)[0]))
        st.metric("Estimated monthly charge (dataset units)", f"{estimate:,.2f}")
        st.caption(
            "Illustrative estimate only. Select logically consistent service options; "
            "this prototype does not validate every combination."
        )

st.divider()
st.caption(
    "Source: public IBM Sample Data Sets / Kaggle Telco Customer Churn dataset. "
    "ABC Ltd. is a fictional case name. Do not enter real customer personal data."
)
