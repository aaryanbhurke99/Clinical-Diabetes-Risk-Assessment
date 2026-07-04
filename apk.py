"""
Meridian — Clinical Diabetes Risk Assessment
==============================================
A Streamlit front end for the Random Forest diabetes classifier trained
in diabetes_classification.py. Loads model.pkl and renders an
enterprise-style clinical decision-support interface.
 
Run with:
    streamlit run app.py
"""
 
import joblib
from datetime import datetime
from pathlib import Path
 
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
 
# ----------------------------------------------------------------------
# PAGE CONFIG (must be the first Streamlit call)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Meridian | Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ----------------------------------------------------------------------
# DESIGN TOKENS
# ----------------------------------------------------------------------
NAVY = "#0E2A47"
NAVY_2 = "#163A5F"
TEAL = "#0EA894"
AMBER = "#C98A1B"
RED = "#B4453F"
BG = "#F6F8FA"
SURFACE = "#FFFFFF"
INK = "#10182B"
MUTED = "#5C6579"
BORDER = "#E3E8EF"
 
# ----------------------------------------------------------------------
# GLOBAL CSS — re-theme Streamlit's default chrome entirely
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
 
    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
        color: {INK};
    }}
 
    .stApp {{
        background: {BG};
    }}
 
    /* Hide default Streamlit chrome for a shipped-product feel */
    #MainMenu, footer, header {{ visibility: hidden; }}
    div[data-testid="stToolbar"] {{ visibility: hidden; }}
    div[data-testid="stDecoration"] {{ visibility: hidden; }}
 
    /* ---- Top bar ---- */
    .m-topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(135deg, {NAVY} 0%, {NAVY_2} 100%);
        padding: 22px 32px;
        border-radius: 14px;
        margin-bottom: 28px;
        box-shadow: 0 6px 20px rgba(14, 42, 71, 0.18);
    }}
    .m-wordmark {{
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        font-size: 22px;
        letter-spacing: 0.08em;
        color: #FFFFFF;
    }}
    .m-wordmark span {{ color: {TEAL}; }}
    .m-subtitle {{
        font-size: 12.5px;
        color: rgba(255,255,255,0.6);
        margin-top: 2px;
        letter-spacing: 0.02em;
    }}
    .m-status {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: rgba(255,255,255,0.75);
    }}
    .m-dot {{
        width: 8px; height: 8px; border-radius: 50%;
        background: {TEAL};
        box-shadow: 0 0 0 0 rgba(14,168,148, 0.6);
        animation: pulse 2.2s infinite;
    }}
    @keyframes pulse {{
        0%   {{ box-shadow: 0 0 0 0 rgba(14,168,148, 0.55); }}
        70%  {{ box-shadow: 0 0 0 7px rgba(14,168,148, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(14,168,148, 0); }}
    }}
 
    /* ---- Cards ---- */
    .m-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 20px 22px;
        height: 100%;
    }}
    .m-card-title {{
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {MUTED};
        margin-bottom: 14px;
    }}
 
    /* ---- Lab result chips ---- */
    .lab-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 3px solid {BORDER};
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }}
    .lab-card.normal {{ border-left-color: {TEAL}; }}
    .lab-card.borderline {{ border-left-color: {AMBER}; }}
    .lab-card.high {{ border-left-color: {RED}; }}
    .lab-label {{
        font-size: 11px;
        color: {MUTED};
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-weight: 600;
    }}
    .lab-value {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 20px;
        font-weight: 600;
        color: {INK};
        margin-top: 2px;
    }}
    .lab-unit {{
        font-size: 12px;
        color: {MUTED};
        font-weight: 400;
        margin-left: 4px;
    }}
    .flag {{
        display: inline-block;
        margin-top: 6px;
        font-size: 10.5px;
        font-weight: 600;
        letter-spacing: 0.04em;
        padding: 2px 8px;
        border-radius: 20px;
        text-transform: uppercase;
    }}
    .flag.normal {{ background: rgba(14,168,148,0.12); color: {TEAL}; }}
    .flag.borderline {{ background: rgba(201,138,27,0.14); color: {AMBER}; }}
    .flag.high {{ background: rgba(180,69,63,0.12); color: {RED}; }}
 
    /* ---- Verdict banner ---- */
    .verdict {{
        border-radius: 12px;
        padding: 18px 22px;
        font-family: 'IBM Plex Mono', monospace;
    }}
    .verdict-label {{
        font-size: 13px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        opacity: 0.85;
    }}
    .verdict-value {{
        font-size: 26px;
        font-weight: 600;
        margin-top: 4px;
    }}
 
    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background: {SURFACE};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 22px;
    }}
    .sidebar-heading {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {NAVY};
        border-bottom: 1px solid {BORDER};
        padding-bottom: 8px;
        margin: 18px 0 12px 0;
    }}
 
    /* ---- Buttons ---- */
    .stButton>button, .stFormSubmitButton>button {{
        background: {NAVY};
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 14px;
        letter-spacing: 0.02em;
        width: 100%;
        transition: background 0.15s ease;
    }}
    .stButton>button:hover, .stFormSubmitButton>button:hover {{
        background: {TEAL};
        color: white;
    }}
 
    /* ---- Footer / model card ---- */
    .m-footer {{
        font-size: 12px;
        color: {MUTED};
        line-height: 1.6;
        border-top: 1px solid {BORDER};
        padding-top: 16px;
        margin-top: 30px;
    }}
 
    .empty-state {{
        text-align: center;
        padding: 90px 20px;
        color: {MUTED};
    }}
    .empty-state .icon {{ font-size: 40px; margin-bottom: 14px; }}
    .empty-state h3 {{
        color: {INK};
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        margin-bottom: 6px;
    }}
 
    /* ---- Defensive label-color backstop ----
       Belt-and-braces on top of .streamlit/config.toml: some browsers/OS
       dark-mode settings can otherwise make native widget labels render
       light-on-light against our custom white sidebar. */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] span {{
        color: {INK} !important;
    }}
    /* Radio option text (e.g. Yes/No) sits in nested elements that the
       rules above don't always reach — force every descendant. */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] * {{
        color: {INK} !important;
    }}
    section[data-testid="stSidebar"] .stButton>button,
    section[data-testid="stSidebar"] .stFormSubmitButton>button,
    section[data-testid="stSidebar"] button[data-testid^="baseButton-"],
    section[data-testid="stSidebar"] button,
    section[data-testid="stSidebar"] button * {{
        color: #FFFFFF !important;
    }}
 
    /* Native bordered containers used for chart cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 12px !important;
        border-color: {BORDER} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ----------------------------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "model.pkl"
 
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["scaler"], bundle["columns"]
 
FEATURE_LABELS = {
    "age": "Age",
    "hypertension": "Hypertension",
    "heart_disease": "Heart Disease",
    "bmi": "BMI",
    "HbA1c_level": "HbA1c Level",
    "blood_glucose_level": "Blood Glucose",
    "gender_Male": "Gender: Male",
    "gender_Other": "Gender: Other",
    "smoking_history_current": "Smoking: Current",
    "smoking_history_ever": "Smoking: Ever",
    "smoking_history_former": "Smoking: Former",
    "smoking_history_never": "Smoking: Never",
    "smoking_history_not current": "Smoking: Not Current",
}
 
# ----------------------------------------------------------------------
# CLINICAL REFERENCE RANGES (for the lab-report style flags)
# ----------------------------------------------------------------------
def classify_bmi(v):
    if v < 18.5:
        return "Underweight", "borderline"
    if v < 25:
        return "Normal", "normal"
    if v < 30:
        return "Overweight", "borderline"
    return "Obese", "high"
 
def classify_hba1c(v):
    if v < 5.7:
        return "Normal", "normal"
    if v < 6.5:
        return "Prediabetes range", "borderline"
    return "Diabetes range", "high"
 
def classify_glucose(v):
    if v < 100:
        return "Normal", "normal"
    if v < 126:
        return "Prediabetes range", "borderline"
    return "Diabetes range", "high"
 
def classify_binary(v, name):
    return ("Present", "high") if v == 1 else ("None reported", "normal")
 
# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def lab_card(label, value_str, unit, flag_label, flag_class):
    st.markdown(
        f"""
        <div class="lab-card {flag_class}">
            <div class="lab-label">{label}</div>
            <div class="lab-value">{value_str}<span class="lab-unit">{unit}</span></div>
            <div class="flag {flag_class}">{flag_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
 
def build_feature_row(inputs: dict, columns: list) -> pd.DataFrame:
    row = {c: 0 for c in columns}
    row["age"] = inputs["age"]
    row["hypertension"] = inputs["hypertension"]
    row["heart_disease"] = inputs["heart_disease"]
    row["bmi"] = inputs["bmi"]
    row["HbA1c_level"] = inputs["hba1c"]
    row["blood_glucose_level"] = inputs["glucose"]
 
    if inputs["gender"] == "Male":
        row["gender_Male"] = 1
    elif inputs["gender"] == "Other":
        row["gender_Other"] = 1
    # Female -> reference category, both stay 0
 
    smoke_col = f"smoking_history_{inputs['smoking']}"
    if smoke_col in row:
        row[smoke_col] = 1
    # "No Info" -> reference category, all smoking dummies stay 0
 
    return pd.DataFrame([row])[columns]
 
def risk_gauge(probability: float):
    if probability < 0.20:
        bar_color = TEAL
    elif probability < 0.50:
        bar_color = AMBER
    else:
        bar_color = RED
 
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"family": "IBM Plex Mono", "size": 44, "color": INK}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED, "tickfont": {"size": 10}},
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20], "color": "rgba(14,168,148,0.10)"},
                    {"range": [20, 50], "color": "rgba(201,138,27,0.10)"},
                    {"range": [50, 100], "color": "rgba(180,69,63,0.10)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=240,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "IBM Plex Sans"},
    )
    return fig
 
def importance_chart(model, columns):
    importances = model.feature_importances_
    labels = [FEATURE_LABELS.get(c, c) for c in columns]
    order = np.argsort(importances)[-6:]  # top 6
    fig = go.Figure(
        go.Bar(
            x=[importances[i] for i in order],
            y=[labels[i] for i in order],
            orientation="h",
            marker=dict(color=NAVY_2),
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=20, t=10, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "IBM Plex Sans", "size": 12, "color": INK},
        xaxis=dict(showgrid=True, gridcolor=BORDER, title="Relative importance", automargin=True),
        yaxis=dict(showgrid=False, automargin=True),
    )
    return fig
 
# ----------------------------------------------------------------------
# TOP BAR
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <div class="m-topbar">
        <div>
            <div class="m-wordmark">MERIDI<span>AN</span></div>
            <div class="m-subtitle">Clinical Decision Support &nbsp;&middot;&nbsp; Diabetes Risk Model</div>
        </div>
        <div class="m-status">
            <div class="m-dot"></div>
            MODEL ONLINE &middot; RF-v1.4.0
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
 
# ----------------------------------------------------------------------
# LOAD MODEL (with graceful failure state)
# ----------------------------------------------------------------------
model, scaler, columns = None, None, None
load_error = None
try:
    model, scaler, columns = load_model()
except Exception as e:
    load_error = str(e)
 
# ----------------------------------------------------------------------
# SIDEBAR — PATIENT INTAKE FORM
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(f'<h3 style="color:#0EA894;">Patient Intake</h3>', unsafe_allow_html=True)
    st.caption("Enter values from the patient chart to generate a risk estimate.")
 
    with st.form("intake_form"):
        st.markdown('<div class="sidebar-heading">Demographics</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.slider("Age", 0, 100, 45)
 
        st.markdown('<div class="sidebar-heading">Clinical History</div>', unsafe_allow_html=True)
        hypertension = st.toggle("Hypertension", value=False)
        heart_disease = st.toggle("Heart Disease", value=False)
        smoking = st.selectbox(
            "Smoking History",
            ["never", "No Info", "current", "former", "ever", "not current"],
            format_func=lambda x: "Unknown / not specified" if x == "No Info" else x.capitalize(),
        )
 
        st.markdown('<div class="sidebar-heading">Lab Values</div>', unsafe_allow_html=True)
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0, value=24.5, step=0.1)
        hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.6, step=0.1)
        glucose = st.number_input("Blood Glucose (mg/dL)", min_value=50, max_value=400, value=110)
 
        submitted = st.form_submit_button("Run Risk Assessment")
 
    if submitted:
        st.session_state["result_inputs"] = {
            "gender": gender,
            "age": age,
            "hypertension": 1 if hypertension else 0,
            "heart_disease": 1 if heart_disease else 0,
            "smoking": smoking,
            "bmi": bmi,
            "hba1c": hba1c,
            "glucose": glucose,
        }
        st.session_state["assessed_at"] = datetime.now().strftime("%H:%M:%S")
 
# ----------------------------------------------------------------------
# MAIN PANEL
# ----------------------------------------------------------------------
if load_error:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="icon">⚠️</div>
            <h3>Model unavailable</h3>
            <p>Could not load <code>model.pkl</code> ({load_error}).<br>
            Run <code>diabetes_classification.py</code> first to train and save the model,
            then restart this app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif "result_inputs" not in st.session_state:
    st.markdown(
        """
        <div class="empty-state">
            <div class="icon">🩺</div>
            <h3>No assessment yet</h3>
            <p>Enter patient information in the panel on the left, then select<br>
            <b>Run Risk Assessment</b> to generate a diabetes risk profile.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    inputs = st.session_state["result_inputs"]
    row = build_feature_row(inputs, columns)
    proba = model.predict_proba(row)[0][1]
 
    if proba < 0.20:
        verdict_text, verdict_bg, verdict_fg = "Low Risk", "rgba(14,168,148,0.10)", TEAL
    elif proba < 0.50:
        verdict_text, verdict_bg, verdict_fg = "Elevated Risk", "rgba(201,138,27,0.12)", AMBER
    else:
        verdict_text, verdict_bg, verdict_fg = "High Risk", "rgba(180,69,63,0.12)", RED
 
    # ---- Hero row: gauge + verdict + summary ----
    col1, col2 = st.columns([1.1, 1])
    with col1:
        with st.container(border=True):
            st.markdown('<div class="m-card-title">Predicted Risk Probability</div>', unsafe_allow_html=True)
            st.plotly_chart(risk_gauge(proba), use_container_width=True, theme=None, config={"displayModeBar": False})
 
    with col2:
        st.markdown(
            f"""
            <div class="verdict" style="background:{verdict_bg}; color:{verdict_fg};">
                <div class="verdict-label">Assessment Result</div>
                <div class="verdict-value">{verdict_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="m-card">
                <div class="m-card-title">Patient Summary</div>
                <p style="font-size:14px; line-height:1.9; color:{INK};">
                    <b>{inputs['age']}</b>-year-old &middot; {inputs['gender']}<br>
                    Assessed at {st.session_state.get('assessed_at','—')}<br>
                    Model: Random Forest Classifier (ROC-AUC 0.963)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ---- Lab report grid ----
    st.markdown('<div class="m-card-title" style="margin-top:6px;">Clinical Values &amp; Reference Ranges</div>', unsafe_allow_html=True)
    lc1, lc2, lc3, lc4 = st.columns(4)
 
    bmi_label, bmi_flag = classify_bmi(inputs["bmi"])
    hba1c_label, hba1c_flag = classify_hba1c(inputs["hba1c"])
    glucose_label, glucose_flag = classify_glucose(inputs["glucose"])
    htn_label, htn_flag = classify_binary(inputs["hypertension"], "hypertension")
    hd_label, hd_flag = classify_binary(inputs["heart_disease"], "heart disease")
 
    with lc1:
        lab_card("BMI", f"{inputs['bmi']:.1f}", "kg/m²", bmi_label, bmi_flag)
    with lc2:
        lab_card("HbA1c", f"{inputs['hba1c']:.1f}", "%", hba1c_label, hba1c_flag)
    with lc3:
        lab_card("Blood Glucose", f"{inputs['glucose']}", "mg/dL", glucose_label, glucose_flag)
    with lc4:
        lab_card("Age", f"{inputs['age']}", "yrs", "Reference only", "normal")
 
    lc5, lc6, lc7, lc8 = st.columns(4)
    with lc5:
        lab_card("Hypertension", "Yes" if inputs["hypertension"] else "No", "", htn_label, htn_flag)
    with lc6:
        lab_card("Heart Disease", "Yes" if inputs["heart_disease"] else "No", "", hd_label, hd_flag)
    with lc7:
        lab_card("Smoking History", inputs["smoking"].capitalize(), "", "Self-reported", "normal")
    with lc8:
        lab_card("Gender", inputs["gender"], "", "—", "normal")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ---- Feature importance ----
    with st.container(border=True):
        st.markdown('<div class="m-card-title">Key Risk Drivers (Model Feature Importance)</div>', unsafe_allow_html=True)
        st.plotly_chart(importance_chart(model, columns), use_container_width=True, theme=None, config={"displayModeBar": False})
 
# ----------------------------------------------------------------------
# FOOTER — model card / disclaimer
# ----------------------------------------------------------------------
with st.expander("Model & data details"):
    st.markdown(
        f"""
        **Algorithm:** Random Forest Classifier (200 estimators, class-balanced)
        **Training data:** 96,146 de-identified patient records, 8 clinical features
        **Validation ROC-AUC:** 0.963 · **Accuracy:** 96.9%
        **Last trained:** see `diabetes_classification.py`
 
        This tool produces a statistical risk estimate based on a machine learning
        model trained on historical clinical data. It is intended for educational
        and demonstration purposes only, does not constitute medical advice or a
        diagnosis, and should not replace evaluation by a licensed healthcare
        professional.
        """
    )
 
st.markdown(
    f"""
    <div class="m-footer">
        Meridian Clinical Decision Support &middot; Diabetes Risk Model &middot; Internal demo build
    </div>
    """,
    unsafe_allow_html=True,
)
 