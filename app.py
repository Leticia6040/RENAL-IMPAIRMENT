# ===============================
# CKD STREAMLIT APP – PRODUCTION READY WITH FULL FEATURES
# ===============================



# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
import base64
from pathlib import Path

BASE_DIR = Path(__file__).parent
logo_path = BASE_DIR / "logo.png"

# Create base64 for HTML
logo_base64 = ""
if logo_path.exists():
    with open(logo_path, "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
else:
    st.warning("Logo file not found")

st.set_page_config(page_title="CKD Dashboard", layout="wide")
st.write("🔥 THIS IS THE NEW VERSION 🔥")

# Now your HTML will work
st.markdown(f"""
<div class="title-bar">
    <img src="data:image/png;base64,{logo_base64}" width="90">
    <div>
        <h4>Chronic Kidney Disease (CKD)</h4>
    </div>
</div>
""", unsafe_allow_html=True)


import joblib
import shap
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# =========================
st.set_page_config(page_title="CKD Dashboard", layout="wide")

# =========================
# NEW VERSION MESSAGE (optional)
# =========================
st.write("🔥 THIS IS THE NEW VERSION 🔥")

# =========================
# LOGO BASE64
# =========================


# =========================
# UI HEADER WITH LOGO
# =========================
st.markdown(f"""
<style>
.title-bar {{
    display: flex;
    align-items: center;
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    color: white;
    padding: 1.2rem;
    border-radius: 12px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}}
.title-bar img {{
    border-radius: 10px;
    margin-right: 20px;
}}
.title-bar h1 {{
    font-size: 1.8rem;
    margin-bottom: 0;
}}
.title-bar h4 {{
    font-size: 1.1rem;
    font-weight: 400;
    color: #dcdcdc;
}}
</style>

<div class="title-bar">
    <img src="data:image/png;base64,{logo_base64}" width="90">
    <div>
        <h4>Chronic Kidney Disease (CKD)</h4>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
st.sidebar.image("/Users/test/Desktop/KIDNEY APP/logo.png", width=100)
st.sidebar.header("📘 About the Model")
st.sidebar.info("""
Predicts **CKD stage** in **PLHIV** using a **stacking ensemble** (RF, GB, Logistic Regression).
Developed for early kidney risk detection in clinical settings.
""")
st.sidebar.markdown("""
---
**Developer:** [LETICIA MINTAH AGYEI]  
**Supervisor:** [DR JOSEPH DADZIE]  
**Institution:** [ACCRA TECHNICAL UNIVERSITY]  
**Year:** 2025  
""")



# -------------------------
# FUNCTIONS
# -------------------------

def compute_eGFR(creatinine, age, gender):
    """Simplified CKD-EPI Equation for eGFR (mg/dL)."""
    if gender.lower() == "female":
        k = 0.7
        a = -0.329
    else:
        k = 0.9
        a = -0.411
    scr = creatinine / k
    egfr = 141 * min(scr, 1)**a * max(scr, 1)**(-1.209) * (0.993**age)
    return round(egfr, 2)


def ckd_stage_from_egfr(egfr):
    """Return CKD stage based on eGFR (KDIGO 2012)."""
    if egfr is None:
        return "Unknown"
    if egfr >= 90:
        return "Stage 1 (Normal / High)"
    elif 60 <= egfr < 90:
        return "Stage 2 (Mildly decreased)"
    elif 45 <= egfr < 60:
        return "Stage 3a (Mild to moderate)"
    elif 30 <= egfr < 45:
        return "Stage 3b (Moderate to severe)"
    elif 15 <= egfr < 30:
        return "Stage 4 (Severe)"
    else:
        return "Stage 5 (Kidney Failure)"



def tdf_action_by_egfr(eGFR, hbv_positive=False):
    """Return TDF recommendation according to Ghana STG."""
    if eGFR is None:
        return "Unknown — no eGFR value"
    if eGFR >= 50:
        return "Continue TDF (normal dosing)"
    if 30 <= eGFR < 50:
        if hbv_positive:
            return "Renal dose adjustment + Specialist referral (HBV+)"
        return "Increase dose interval (renal adjustment)"
    if eGFR < 30:
        if hbv_positive:
            return "STOP TDF + Specialist referral (HBV+ high risk)"
        return "STOP TDF — severe renal dysfunction"


# -------------------------
# HELPER MAPPINGS
# -------------------------
def yesno_to_float(x):
    if isinstance(x, str):
        x = x.strip().lower()
        if x == "yes": return 1.0
        if x == "no": return 0.0
    return 0.0


def gender_map(x):
    if isinstance(x, str) and x.lower() == "male":
        return 1.0
    return 0.0


def bmimap(x):
    if isinstance(x, str):
        s = x.lower()
        if "under" in s: return 0.0
        if "norm" in s: return 1.0
        return 2.0
    return 1.0


def hivstage_map(x):
    try:
        if isinstance(x, str) and "stage" in x.lower():
            return float(int(x.split()[-1]))
    except:
        return 1.0
    return 1.0


def build_input_df():
    """Build a DataFrame aligned to model features."""
    df_in = pd.DataFrame(columns=expected_features, dtype=float)
    df_in.loc[0] = 0.0

    numeric_map = {
        "Age": Age,
        "SerumCreatinineMgPerdL": SerumCreatinineMgPerdL,
        "CurrentCD4": CurrentCD4,
        "TDFDurYrs": TDFDurYrs,
        "RetroDurMths": RetroDurMths,
        "TotalDurOnARVyrs": TotalDurOnARVyrs,
        "SMK": SMK,
        "DRK": DRK
    }

    for key, val in numeric_map.items():
        for c in [key, key.lower(), key.upper(), key.capitalize()]:
            if c in df_in.columns:
                df_in.at[0, c] = float(val)
                break

    cat_map = {
        "Gender": gender_map(Gender),
        "HTNPresence": yesno_to_float(HTNPresence),
        "Alcohol": yesno_to_float(Alcohol),
        "Smoking": yesno_to_float(Smoking),
        "Comorbidity": yesno_to_float(Comorbidity),
        "HyperDyslipidemia": yesno_to_float(HyperDyslipidemia),
        "HepBC": yesno_to_float(HepBC)
    }

    for key, val in cat_map.items():
        for c in [key, key.lower(), key.upper(), key.capitalize()]:
            if c in df_in.columns:
                df_in.at[0, c] = float(val)
                break

    # BMICat mapping
    bmi_val = bmimap(BMICat)
    for c in ["BMICat", "bmicat", "BMICAT"]:
        if c in df_in.columns:
            df_in.at[0, c] = float(bmi_val)
            break

    # HIV Stage mapping
    hiv_val = hivstage_map(HIVStage)
    for c in ["HIVStage", "HIVstage", "HIV_STG", "HIVSTAGE"]:
        if c in df_in.columns:
            df_in.at[0, c] = float(hiv_val)
            break

    # Educational, MaritalStatus mapping
    encoders = {
        "Educational": {"NONE":0.0, "JHS/ELEMENTARY":1.0, "SEC/VOC/TECH":2.0, "TERTIARY":3.0},
        "MaritalStatus": {"Single":0.0, "Married":1.0, "Divorced":2.0, "Widowed":3.0}
    }

    for feat, mapping in encoders.items():
        sel = globals().get(feat, None)
        val = mapping.get(sel, 0.0)
        for c in [feat, feat.lower(), feat.upper()]:
            if c in df_in.columns:
                df_in.at[0, c] = float(val)
                break

    return df_in


# Load Logo
logo_path = Path("/Users/test/Desktop/KIDNEY APP/logo.png")
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()
logo_base64 = get_base64_image(logo_path)

# -------------------------
# Load Model & Scaler
model = joblib.load("ckd_stage_stacking_model.pkl")
scaler = joblib.load("ckd_scaler.pkl")
expected_features = []
if hasattr(scaler, "feature_names_in_"):
    expected_features = list(scaler.feature_names_in_)
elif hasattr(model, "feature_names_in_"):
    expected_features = list(model.feature_names_in_)
else:
    st.error("Saved scaler/model does not expose feature names. Retrain with pipeline.")
    st.stop()




tabs = st.tabs([
    "Patient Info",
    "CKD Prediction",
    "Visual Trends",
    "Feature Explanation",
    "Download Report"
])


# -------------------------
# FORM INPUTS
# -------------------------
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input("Age (years)", min_value=0, max_value=120, step=1, value=0)
        Gender = st.selectbox("Gender", ["Select", "Male", "Female"])
        Educational = st.selectbox("Educational level", ["Select", "NONE", "JHS/ELEMENTARY", "SEC/VOC/TECH", "TERTIARY"])
        MaritalStatus = st.selectbox("Marital status", ["Select", "Single", "Married", "Divorced", "Widowed"])

    with col2:
        SerumCreatinineMgPerdL = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, max_value=50.0, step=0.1, value=0.0)
        CurrentCD4 = st.number_input("Current CD4 Count", min_value=0.0, max_value=5000.0, step=1.0, value=0.0)
        TDFDurYrs = st.number_input("TDF Duration (years)", min_value=0.0, max_value=50.0, step=0.1, value=0.0)
        TDFCategory = "Short (<2 yrs)" if TDFDurYrs < 2 else "Medium (2–5 yrs)" if TDFDurYrs <= 5 else "Long (>5 yrs)"
        st.write(f"TDF Exposure Category: {TDFCategory}")
        RetroDurMths = st.number_input("Retro Duration (months)", min_value=0.0, max_value=600.0, step=1.0, value=0.0)

    with col3:
        TotalDurOnARVyrs = st.number_input("Total ARV Duration (years)", min_value=0.0, max_value=60.0, step=0.1, value=0.0)
        HIVStage = st.selectbox("HIV Stage", ["Select", "Stage 1", "Stage 2", "Stage 3", "Stage 4"])
        BMICat = st.selectbox("BMI Category", ["Select", "Underweight", "Normal", "Overweight/Obese"])
        HTNPresence = st.selectbox("Hypertension Present?", ["Select", "Yes", "No"])

    col4, col5, col6 = st.columns(3)
    with col4:
        Alcohol = st.selectbox("Alcohol use?", ["Select", "Yes", "No"])
        Smoking = st.selectbox("Smoking?", ["Select", "Yes", "No"])
    with col5:
        Comorbidity = st.selectbox("Comorbidities present?", ["Select", "Yes", "No"])
        HyperDyslipidemia = st.selectbox("HyperDyslipidemia?", ["Select", "Yes", "No"])
    with col6:
        HepBC = st.selectbox("Hepatitis B coinfection?", ["Select", "Yes", "No"])
        SMK = 1 if Smoking == "Yes" else 0
        DRK = 1 if Alcohol == "Yes" else 0

    submitted = st.form_submit_button("🔍 Predict CKD Stage")

# -------------------------
# SUBMISSION LOGIC
# -------------------------
if submitted:
    # Compute eGFR
    try:
        computed_eGFR = compute_eGFR(SerumCreatinineMgPerdL, Age, Gender)
    except:
        computed_eGFR = None

    # CKD stage clinical
    ckd_stage_clinical = ckd_stage_from_egfr(computed_eGFR)

    # TDF recommendation
    hbv_pos = 1 if HepBC == "Yes" else 0
    tdf_recommendation = tdf_action_by_egfr(computed_eGFR, hbv_positive=hbv_pos)

    # Show clinical results
    st.subheader("TDF Dosage Recommendation")
    st.info(f"eGFR: {computed_eGFR} ml/min/1.73m² | HBV: {'Positive' if hbv_pos else 'Negative'}\nRecommendation: {tdf_recommendation}")

    st.subheader("CKD Stage Based on eGFR (Clinical)")
    st.info(f"{ckd_stage_clinical}")

    # Validate required fields
    required_numeric = [Age, SerumCreatinineMgPerdL, CurrentCD4, TDFDurYrs, RetroDurMths, TotalDurOnARVyrs]
    if any(v == 0 for v in required_numeric) or "Select" in [Gender, BMICat, HIVStage]:
        st.warning("Please fill in core numeric fields and select Gender, BMI, and HIV Stage.")
    else:
        try:
            input_df = build_input_df()[expected_features]
            X_scaled = scaler.transform(input_df)
            pred = model.predict(X_scaled)[0]

            st.subheader("Predicted CKD Stage")
            if str(pred).startswith("1"):
                st.success("Normal")
            elif str(pred).startswith("2"):
                st.warning("Mild Renal Impairment")
            elif str(pred).startswith("3"):
                st.error("Severe Renal Impairment")
            else:
                st.info(f"Predicted (raw): {pred}")

            if st.checkbox("Show input sent to model"):
                st.dataframe(input_df.T)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


    with tabs[2]:
        st.subheader("📈 Trend Visualization")
        demo_df = pd.DataFrame({
            "Month":[1,2,3,4,5],
            "eGFR":[computed_eGFR-5,computed_eGFR-3,computed_eGFR,computed_eGFR+2,computed_eGFR+1],
            "Creatinine":[SerumCreatinineMgPerdL+0.1*i for i in range(5)],
            "CD4":[CurrentCD4+10*i for i in range(5)]
        })
        st.line_chart(demo_df.set_index("Month"))

    # -------------------------
    with tabs[3]:
        st.subheader("Feature Explanation (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)
            fig, ax = plt.subplots(figsize=(8,4))
            shap.summary_plot(shap_values, X_scaled, feature_names=expected_features, show=False)
            st.pyplot(fig)
        except:
            st.info("SHAP plot not available for non-tree models.")

    # -------------------------
    with tabs[4]:
        st.subheader("💾 Download Report")
        report_df = pd.DataFrame({"Feature":input_df.columns, "Value":input_df.loc[0].values})
        csv = report_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Report", data=csv, file_name="patient_report.csv", mime="text/csv") 




# -------------------------
# FOOTER
# -------------------------
st.markdown("""
<footer>
Developed for academic research on **AI in HIV-Associated CKD Prediction** — Ghana.  
© 2025 | <b>AI for Healthcare Initiative</b>
</footer>
""", unsafe_allow_html=True)
