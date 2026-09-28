# ============================================================
# app.py
# ABC Ltd. - Predictive Intelligence Dashboard
#
# Streamlit application for:
#   1. Employee Attrition Prediction
#   2. Monthly Income Prediction
#   3. What-If Analysis
#   4. Model Validation
#   5. Gemini Managerial AI Assistant
# ============================================================

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ABC Ltd. | Predictive Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #263244;
    }

    /* Headers */
    h1, h2, h3 {
        color: #f9fafb;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 12px;
        padding: 15px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-weight: 600;
    }

    /* Info boxes */
    .manager-card {
        background-color: #111827;
        border: 1px solid #263244;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
    }

    .risk-high {
        color: #f87171;
        font-weight: 800;
        font-size: 1.5rem;
    }

    .risk-medium {
        color: #fbbf24;
        font-weight: 800;
        font-size: 1.5rem;
    }

    .risk-low {
        color: #34d399;
        font-weight: 800;
        font-size: 1.5rem;
    }

    .small-text {
        color: #9ca3af;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

LOGISTIC_MODEL_PATH = MODEL_DIR / "logistic_model.joblib"
LINEAR_MODEL_PATH = MODEL_DIR / "linear_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
CONFIG_PATH = MODEL_DIR / "model_config.json"


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    """
    Load trained models once and cache them.
    """

    if not LOGISTIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            "logistic_model.joblib not found. "
            "Run model_builder.py first."
        )

    if not LINEAR_MODEL_PATH.exists():
        raise FileNotFoundError(
            "linear_model.joblib not found. "
            "Run model_builder.py first."
        )

    logistic_model = joblib.load(
        LOGISTIC_MODEL_PATH
    )

    linear_model = joblib.load(
        LINEAR_MODEL_PATH
    )

    return logistic_model, linear_model


@st.cache_data
def load_metrics():
    """
    Load model metrics.
    """

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "metrics.json not found. "
            "Run model_builder.py first."
        )

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


@st.cache_data
def load_config():
    """
    Load model configuration.
    """

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "model_config.json not found."
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# SAFE MODEL LOADING
# ============================================================

try:

    logistic_model, linear_model = load_models()
    metrics = load_metrics()
    config = load_config()

except Exception as error:

    st.error(
        "The application could not load the trained models."
    )

    st.info(
        "Please run `python model_builder.py` first."
    )

    st.code(
        str(error)
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📊 ABC Ltd. Predictive Intelligence Dashboard")

st.markdown(
    """
    **Managerial AI Decision-Support Platform**

    Use predictive analytics to estimate employee attrition risk,
    expected monthly income, and explore what-if scenarios.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Employee Profile")

st.sidebar.caption(
    "Adjust the inputs below to simulate different managerial scenarios."
)


# ----------------------------
# Basic information
# ----------------------------

st.sidebar.subheader("Employee Profile")

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=65,
    value=35
)

business_travel = st.sidebar.selectbox(
    "Business Travel",
    [
        "Travel_Rarely",
        "Travel_Frequently",
        "Non-Travel"
    ]
)

department = st.sidebar.selectbox(
    "Department",
    [
        "Sales",
        "Research & Development",
        "Human Resources"
    ]
)

distance_from_home = st.sidebar.slider(
    "Distance From Home",
    min_value=1,
    max_value=50,
    value=10
)

job_level = st.sidebar.slider(
    "Job Level",
    min_value=1,
    max_value=5,
    value=2
)

job_role = st.sidebar.selectbox(
    "Job Role",
    [
        "Sales Executive",
        "Research Scientist",
        "Laboratory Technician",
        "Manager",
        "Human Resources",
        "Healthcare Representative"
    ]
)


# ----------------------------
# Satisfaction / work
# ----------------------------

st.sidebar.subheader("Work Experience")

job_satisfaction = st.sidebar.slider(
    "Job Satisfaction",
    min_value=1,
    max_value=4,
    value=3
)

job_involvement = st.sidebar.slider(
    "Job Involvement",
    min_value=1,
    max_value=4,
    value=3
)

work_life_balance = st.sidebar.slider(
    "Work-Life Balance",
    min_value=1,
    max_value=4,
    value=3
)

overtime = st.sidebar.selectbox(
    "Overtime",
    ["No", "Yes"]
)

total_working_years = st.sidebar.slider(
    "Total Working Years",
    min_value=0,
    max_value=40,
    value=10
)

years_at_company = st.sidebar.slider(
    "Years at Company",
    min_value=0,
    max_value=30,
    value=5
)

years_current_role = st.sidebar.slider(
    "Years in Current Role",
    min_value=0,
    max_value=20,
    value=3
)

years_since_promotion = st.sidebar.slider(
    "Years Since Last Promotion",
    min_value=0,
    max_value=15,
    value=2
)

years_manager = st.sidebar.slider(
    "Years With Current Manager",
    min_value=0,
    max_value=20,
    value=3
)

education = st.sidebar.slider(
    "Education Level",
    min_value=1,
    max_value=5,
    value=3
)


# ============================================================
# BUILD INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame(
    {
        "Age": [age],
        "BusinessTravel": [business_travel],
        "Department": [department],
        "DistanceFromHome": [distance_from_home],
        "JobLevel": [job_level],
        "JobRole": [job_role],
        "JobSatisfaction": [job_satisfaction],
        "OverTime": [overtime],
        "TotalWorkingYears": [total_working_years],
        "YearsAtCompany": [years_at_company],
        "YearsInCurrentRole": [years_current_role],
        "YearsSinceLastPromotion": [years_since_promotion],
        "YearsWithCurrManager": [years_manager],
        "JobInvolvement": [job_involvement],
        "WorkLifeBalance": [work_life_balance],
        "Education": [education],
    }
)


# ============================================================
# MODEL PREDICTIONS
# ============================================================

try:

    attrition_probability = float(
        logistic_model.predict_proba(input_data)[0][1]
    )

    attrition_prediction = int(
        logistic_model.predict(input_data)[0]
    )

    predicted_income = float(
        linear_model.predict(input_data)[0]
    )

except Exception as error:

    st.error(
        "Prediction could not be generated."
    )

    st.exception(error)

    st.stop()


# ============================================================
# RISK CATEGORIZATION
# ============================================================

def get_risk_category(probability):

    if probability >= 0.70:
        return "HIGH", "risk-high"

    if probability >= 0.40:
        return "MEDIUM", "risk-medium"

    return "LOW", "risk-low"


risk_category, risk_css = get_risk_category(
    attrition_probability
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🎯 Predictions & What-If",
        "📈 Model Validation",
        "🤖 AI Manager Assistant",
        "ℹ️ Managerial Interpretation",
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.header("Predictive Decision Dashboard")

    st.caption(
        "Predictions are statistical estimates and should be used "
        "as decision-support rather than automatic decisions."
    )

    col1, col2, col3 = st.columns(3)

    # ----------------------------
    # Attrition probability
    # ----------------------------

    with col1:

        st.metric(
            "Attrition Probability",
            f"{attrition_probability:.1%}"
        )

        st.markdown(
            f"""
            <div class="{risk_css}">
                {risk_category} RISK
            </div>
            """,
            unsafe_allow_html=True
        )

    # ----------------------------
    # Prediction
    # ----------------------------

    with col2:

        status = (
            "Potential Attrition"
            if attrition_prediction == 1
            else "Likely Retention"
        )

        st.metric(
            "Model Status",
            status
        )

    # ----------------------------
    # Income
    # ----------------------------

    with col3:

        st.metric(
            "Predicted Monthly Income",
            f"₹{predicted_income:,.0f}"
        )

    st.divider()

    # ----------------------------
    # Probability chart
    # ----------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("Attrition Risk")

        fig, ax = plt.subplots(
            figsize=(6, 3)
        )

        ax.barh(
            ["Attrition"],
            [attrition_probability]
        )

        ax.set_xlim(0, 1)

        ax.set_xlabel(
            "Probability"
        )

        ax.set_title(
            "Predicted Attrition Probability"
        )

        ax.grid(
            axis="x",
            alpha=0.2
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with right:

        st.subheader("Current Employee Profile")

        display_df = input_data.T.reset_index()

        display_df.columns = [
            "Variable",
            "Value"
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    st.divider()

    # ========================================================
    # WHAT-IF ANALYSIS
    # ========================================================

    st.subheader("🔄 What-If Scenario Simulator")

    st.write(
        """
        Change selected variables below to see how the model's
        predicted attrition probability changes.
        """
    )

    scenario_col1, scenario_col2 = st.columns(2)

    with scenario_col1:

        scenario_overtime = st.selectbox(
            "Scenario: Overtime",
            ["No", "Yes"],
            index=0 if overtime == "No" else 1
        )

        scenario_satisfaction = st.slider(
            "Scenario: Job Satisfaction",
            1,
            4,
            job_satisfaction
        )

    with scenario_col2:

        scenario_distance = st.slider(
            "Scenario: Distance From Home",
            1,
            50,
            distance_from_home
        )

        scenario_job_involvement = st.slider(
            "Scenario: Job Involvement",
            1,
            4,
            job_involvement
        )

    scenario_data = input_data.copy()

    scenario_data["OverTime"] = scenario_overtime
    scenario_data["JobSatisfaction"] = scenario_satisfaction
    scenario_data["DistanceFromHome"] = scenario_distance
    scenario_data["JobInvolvement"] = scenario_job_involvement

    scenario_probability = float(
        logistic_model.predict_proba(
            scenario_data
        )[0][1]
    )

    difference = (
        scenario_probability
        - attrition_probability
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Current Risk",
            f"{attrition_probability:.1%}"
        )

    with c2:

        st.metric(
            "Scenario Risk",
            f"{scenario_probability:.1%}"
        )

    with c3:

        st.metric(
            "Risk Change",
            f"{difference:+.1%}"
        )

    if difference < 0:

        st.success(
            "The simulated scenario reduces predicted attrition risk."
        )

    elif difference > 0:

        st.warning(
            "The simulated scenario increases predicted attrition risk."
        )

    else:

        st.info(
            "The simulated scenario produces the same predicted risk."
        )


# ============================================================
# TAB 2 — VALIDATION
# ============================================================

with tab2:

    st.header("Model Validation & Performance")

    st.caption(
        "Validation metrics are calculated on the held-out test set."
    )

    logistic_metrics = metrics[
        "logistic_regression"
    ]

    linear_metrics = metrics[
        "linear_regression"
    ]

    # ========================================================
    # LOGISTIC METRICS
    # ========================================================

    st.subheader("Logistic Regression — Attrition")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Accuracy",
            f"{logistic_metrics['accuracy']:.3f}"
        )

    with c2:
        st.metric(
            "Precision",
            f"{logistic_metrics['precision']:.3f}"
        )

    with c3:
        st.metric(
            "Recall",
            f"{logistic_metrics['recall']:.3f}"
        )

    with c4:
        st.metric(
            "F1 Score",
            f"{logistic_metrics['f1_score']:.3f}"
        )

    # Confusion matrix
    cm = np.array(
        [
            [
                logistic_metrics["true_negative"],
                logistic_metrics["false_positive"]
            ],
            [
                logistic_metrics["false_negative"],
                logistic_metrics["true_positive"]
            ]
        ]
    )

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Predicted No Attrition",
            "Predicted Attrition"
        ],
        yticklabels=[
            "Actual No Attrition",
            "Actual Attrition"
        ],
        ax=ax
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

    plt.close(fig)

    st.info(
        """
        **Managerial interpretation:** Recall measures how many actual
        attrition cases the model successfully identifies. A false
        negative represents an employee who may leave but is not
        identified by the model.
        """
    )

    st.divider()

    # ========================================================
    # LINEAR METRICS
    # ========================================================

    st.subheader(
        "Linear Regression — Monthly Income"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "R² Score",
            f"{linear_metrics['r2_score']:.3f}"
        )

    with c2:

        st.metric(
            "RMSE",
            f"₹{linear_metrics['rmse']:,.0f}"
        )

    with c3:

        st.metric(
            "MAE",
            f"₹{linear_metrics['mae']:,.0f}"
        )

    st.markdown(
        """
        **Interpretation**

        - **R²:** proportion of variation explained by the model.
        - **RMSE:** average prediction error with greater penalty for large errors.
        - **MAE:** average absolute prediction error in the same unit as income.
        """
    )


# ============================================================
# TAB 3 — GEMINI AI ASSISTANT
# ============================================================

with tab3:

    st.header("🤖 AI Manager Assistant")

    st.markdown(
        """
        Ask questions about the current prediction, employee profile,
        model output, or managerial implications.

        Example:

        > Why is this employee at risk and what factors should I investigate?
        """
    )

    st.divider()

    prompt = st.text_area(
        "Enter your managerial question",
        placeholder=(
            "Example: Why is this employee at risk and "
            "what should a manager investigate before taking action?"
        ),
        height=130
    )

    ask_button = st.button(
        "Ask AI Assistant",
        type="primary",
        use_container_width=True
    )

    if ask_button:

        if not prompt.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            # ------------------------------------------------
            # Try to obtain Gemini key
            # ------------------------------------------------

            api_key = None

            try:

                api_key = st.secrets.get(
                    "GEMINI_API_KEY",
                    None
                )

            except Exception:

                api_key = None

            # Also allow environment variable
            if not api_key:

                import os

                api_key = os.getenv(
                    "GEMINI_API_KEY"
                )

            # ------------------------------------------------
            # No API key
            # ------------------------------------------------

            if not api_key:

                st.warning(
                    "Gemini API key is not configured."
                )

                st.info(
                    """
                    The predictive model is working, but the generative
                    AI assistant is currently unavailable.

                    Add your API key using Streamlit Secrets:

                    GEMINI_API_KEY = "your-api-key"
                    """
                )

                st.subheader(
                    "Built-in analytical response"
                )

                st.write(
                    f"""
                    The current model estimates an attrition probability
                    of **{attrition_probability:.1%}**.

                    The current risk category is **{risk_category}**.

                    The predicted monthly income is approximately
                    **₹{predicted_income:,.0f}**.

                    Before taking action, a manager should investigate
                    job satisfaction, overtime, workload, distance from
                    home, career progression, and other contextual factors.
                    """
                )

            # ------------------------------------------------
            # Gemini
            # ------------------------------------------------

            else:

                try:

                    from google import genai

                    client = genai.Client(
                        api_key=api_key
                    )

                    context = {
                        "employee_profile":
                            input_data.to_dict(
                                orient="records"
                            )[0],

                        "attrition_probability":
                            round(
                                attrition_probability,
                                4
                            ),

                        "risk_category":
                            risk_category,

                        "predicted_monthly_income":
                            round(
                                predicted_income,
                                2
                            ),

                        "logistic_metrics":
                            logistic_metrics,

                        "linear_metrics":
                            linear_metrics,
                    }

                    system_instruction = """
You are a managerial decision-support assistant
for ABC Ltd.

Your role is to help managers interpret predictive
analytics without making automatic employment decisions.

Important principles:
1. Do not claim that a prediction proves an employee will leave.
2. Do not recommend firing, promotion, salary reduction, or disciplinary action solely from the model.
3. Explain predictions as probabilities and model estimates.
4. Encourage managers to combine model output with human judgement,
   organizational policy, and direct employee context.
5. Distinguish model evidence from managerial judgement.
6. If the user asks for a managerial action, provide reasonable
   investigation or discussion steps rather than an automatic decision.
7. Keep the answer concise and practical.
"""

                    full_prompt = f"""
{system_instruction}

CURRENT ANALYTICS CONTEXT:

{json.dumps(context, indent=2, default=str)}

MANAGER'S QUESTION:

{prompt}
"""

                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=full_prompt
                    )

                    answer = getattr(
                        response,
                        "text",
                        None
                    )

                    if answer:

                        st.success(
                            "AI response generated."
                        )

                        st.markdown(
                            answer
                        )

                    else:

                        st.warning(
                            "Gemini returned an empty response."
                        )

                except Exception as error:

                    st.error(
                        "The AI assistant could not be reached."
                    )

                    st.caption(
                        "The predictive analytics components remain available."
                    )

                    with st.expander(
                        "Technical error"
                    ):

                        st.code(
                            str(error)
                        )


# ============================================================
# TAB 4 — MANAGERIAL INTERPRETATION
# ============================================================

with tab4:

    st.header(
        "Managerial Interpretation"
    )

    st.subheader(
        "How should managers use this application?"
    )

    st.markdown(
        """
        ### 1. Predictive analytics

        The logistic regression model estimates the probability
        of employee attrition.

        The linear regression model estimates expected monthly income.

        ### 2. What-if analysis

        Managers can modify selected employee characteristics to
        explore how the model prediction changes.

        This does **not** prove that changing one variable will
        necessarily cause the observed change in a real employee.

        ### 3. Human judgement

        Model predictions should support rather than automatically
        replace managerial judgement.

        A prediction can identify an employee or situation that
        deserves further investigation.

        ### 4. Explainability

        Managers should be able to understand:

        - what the model predicts;
        - how confident/probabilistic the output is;
        - what information was provided to the model;
        - what the model cannot determine;
        - where human judgement remains necessary.

        ### 5. Responsible AI

        Predictive models can contain errors and reflect limitations
        in their training data.

        Therefore, managers should avoid treating the output as a
        definitive statement about an individual employee.
        """
    )

    st.divider()

    st.subheader(
        "Assignment Connection"
    )

    st.markdown(
        """
        **Descriptive Analytics**
        → What has happened?

        **Diagnostic Analytics**
        → Why might it have happened?

        **Predictive Analytics**
        → What may happen?

        **Prescriptive / Decision Support**
        → What options should the manager investigate?

        **Generative AI**
        → Provides a natural-language interface for interpreting
        predictive outputs.
        """
    )

    st.divider()

    st.caption(
        "ABC Ltd. Predictive Intelligence Dashboard | Academic Project"
    )
