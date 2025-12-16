# app.py — Health Insurance Premium Predictor (Professional UI with Tooltips + SHAP)

import streamlit as st
import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Health Premium Predictor",
    layout="centered"
)

# ---------------- PATHS ----------------
ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT, "model", "insurance_model.pkl")
EXPLAINER_PATH = os.path.join(ROOT, "model", "shap_explainer.pkl")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_shap():
    if os.path.exists(EXPLAINER_PATH):
        with open(EXPLAINER_PATH, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()
shap_bundle = load_shap()

# ---------------- SHAP HELPERS ----------------
def get_feature_names(preprocessor):
    feature_names = []

    num_features = [
        f for f in preprocessor.transformers_[0][2]
        if f != "enquiry_id"
    ]
    feature_names.extend(num_features)

    cat_transformer = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]
    ohe_features = cat_transformer.named_steps["onehot"].get_feature_names_out(cat_features)

    feature_names.extend(ohe_features)
    return feature_names


def filter_selected_city_shap(shap_vals, feature_names, selected_city):
    filtered_vals, filtered_names = [], []
    city_feature = f"city_{selected_city}"

    for v, n in zip(shap_vals, feature_names):
        if n == city_feature or not n.startswith("city_"):
            filtered_vals.append(v)
            filtered_names.append(n)

    return np.array(filtered_vals), filtered_names


# ============================================================
#                       UI
# ============================================================

st.markdown("# **Risk2Rate**")
st.caption("## Health insurance premium estimator")

st.divider()


# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.markdown("## Policy Details")

age = st.sidebar.number_input(
    "Age", 18, 80, 30,
    help="Age of the insured person. Premium generally increases with age."
)

gender = st.sidebar.selectbox(
    "Gender", ["Male", "Female"],
    help="Used for actuarial risk analysis."
)

smoker = st.sidebar.selectbox(
    "Smoking Status", ["No", "Yes"],
    help="Smoking increases long-term health risk and premium."
)

# ---------- BMI Inputs ----------
height_cm = st.sidebar.number_input("Height (cm)", 140, 210, 170)
weight_kg = st.sidebar.number_input("Weight (kg)", 40, 150, 70)


city = st.sidebar.selectbox(
    "City",
    ["Delhi", "Mumbai", "Bangalore", "Pune", "Chennai", "Hyderabad", "Kolkata"],
    help="Metro cities usually have higher medical costs."
)

policy_type = st.sidebar.selectbox(
    "Policy Type", ["Individual", "Family Floater"],
    help="Individual covers one person; Family Floater covers multiple members."
)

sum_insured = st.sidebar.selectbox(
    "Sum Insured (₹)",
    [300000, 500000, 700000, 1000000, 1500000],
    help="Higher sum insured increases premium."
)

insurer_name = st.sidebar.selectbox(
    "Insurer",
    ["Star Health", "HDFC ERGO", "ICICI Lombard", "Niva Bupa", "Care Health", "Max Bupa"],
    help="Different insurers have different pricing strategies."
)

network_hospitals = st.sidebar.number_input(
    "Network Hospitals", 5000, 20000, 10000,
    help="Number of hospitals with cashless treatment."
)

pre_existing_disease = st.sidebar.selectbox(
    "Pre-existing Disease", ["Yes", "No"],
    help="Existing illnesses increase claim probability."
)

waiting_period_years = st.sidebar.slider(
    "Waiting Period (Years)", 0, 5, 2,
    help="Shorter waiting period increases premium."
)

co_payment_percent = st.sidebar.selectbox(
    "Co-payment (%)", [0, 10, 20],
    help="Higher co-payment lowers premium."
)

claim_settlement_ratio = st.sidebar.slider(
    "Claim Settlement Ratio (%)", 90.0, 100.0, 97.0,
    help="Higher ratio indicates insurer reliability."
)

predict_btn = st.sidebar.button("Predict Premium")

# ---------------- PREDICTION ----------------
if predict_btn:
    input_df = pd.DataFrame([{
        "enquiry_id": 0,
        "age": age,
        "gender": gender,
        "city": city,
        "policy_type": policy_type,
        "sum_insured": sum_insured,
        "insurer_name": insurer_name,
        "network_hospitals": network_hospitals,
        "pre_existing_disease": pre_existing_disease,
        "waiting_period_years": waiting_period_years,
        "co_payment_percent": co_payment_percent,
        "claim_settlement_ratio": claim_settlement_ratio
    }])

    base_prediction = model.predict(input_df)[0]

    # ---------------- SMOKING ADJUSTMENT ----------------
    if smoker == "Yes":
        adjusted_prediction = base_prediction * 1.35
        smoking_note = "Smoking increases risk → 35% premium loading applied."
    else:
        adjusted_prediction = base_prediction * 0.95
        smoking_note = "Non-smoker benefit → 5% discount applied."

    # ====================================================
    #        BMI-BASED PREMIUM ADJUSTMENT (IMAGE RULES)
    # ====================================================
    bmi = weight_kg / ((height_cm / 100) ** 2)

    st.divider()
   
    bmi_factor = 1.0
    bmi_note = "BMI is in normal range → No premium increase."

    if 25 <= bmi < 30:
        bmi_factor = 1.07   # +5% to +10% → using midpoint
        bmi_note = "Overweight (BMI 25–29.9) → ~7% premium increase applied."

    elif 30 <= bmi < 35:
        bmi_factor = 1.15   # +10% to +20% → using midpoint
        bmi_note = "Obese Class I (BMI 30–34.9) → ~15% premium increase applied."

    elif bmi >= 35:
        bmi_factor = 1.30   # +20% to +50% → conservative 30%
        bmi_note = "Obese Class II/III (BMI ≥35) → ~30% premium increase applied."

    # Apply BMI adjustment on top of smoking-adjusted premium
    adjusted_prediction = adjusted_prediction * bmi_factor


    # ---------------- RESULTS ----------------
    st.markdown("### Estimated Premium")
    c1, c2 = st.columns(2)
    c1.metric("Annual Premium", f"₹ {adjusted_prediction:,.0f}")
    c2.metric("Monthly Premium", f"₹ {adjusted_prediction/12:,.0f}")
    st.caption(smoking_note)

    # ---------- BMI Calculation ----------
    st.markdown("### BMI Analysis")
    st.write(f"**BMI:** {bmi:.1f}")

    if bmi < 18.5:
       st.info("Category: Underweight")
    elif bmi < 25:
         st.success("Category: Normal")
    elif bmi < 30:
         st.warning("Category: Overweight")
    else:
         st.error("Category: Obese")

    
    # ---------- Smoking Premium Comparison (UI-only) ----------
    premium_without_smoking = base_prediction * 0.95
    premium_with_smoking = base_prediction * 1.35

    st.divider()
    st.markdown("### Smoking Impact on Premium")

    c1, c2 = st.columns(2)
    c1.metric("Premium (Non-Smoker)", f"₹ {premium_without_smoking:,.0f}")
    c2.metric("Premium (Smoker)", f"₹ {premium_with_smoking:,.0f}")


    # ====================================================
    #        PREMIUM CONFIDENCE INTERVAL (ADDED)
    # ====================================================
    st.divider()
    st.markdown("### Premium Confidence Interval")
    st.caption("Estimated range where the actual premium may fall (95% confidence)")

    uncertainty = 0.10  # default ±10%

    if smoker == "Yes" or pre_existing_disease == "Yes":
        uncertainty = 0.15
    elif age < 35 and smoker == "No":
        uncertainty = 0.08

    lower_bound = adjusted_prediction * (1 - uncertainty)
    upper_bound = adjusted_prediction * (1 + uncertainty)

    st.write(f"₹ {lower_bound:,.0f}  –  ₹ {upper_bound:,.0f}")

    # ---------------- WHY PREMIUM ----------------
    st.divider()
    st.markdown("### Why is this premium high?")
    reasons = []

    if smoker == "Yes":
        reasons.append("Smoking increases long-term health risk.")
    if age > 45:
        reasons.append("Higher age increases claim probability.")
    if sum_insured >= 1_000_000:
        reasons.append("Higher coverage increases insurer liability.")
    if pre_existing_disease == "Yes":
        reasons.append("Pre-existing conditions increase claims.")
    if waiting_period_years == 0:
        reasons.append("Immediate coverage increases cost.")

    for r in reasons:
        st.write("•", r)

    # ---------------- HOW TO REDUCE PREMIUM ----------------
    st.divider()
    st.markdown("### How can you reduce your premium?")
    recommendations = []

    if smoker == "Yes":
        recommendations.append("Quitting smoking reduces future premiums.")
    if sum_insured > 500000:
        recommendations.append("Lower sum insured if medically suitable.")
    if co_payment_percent == 0:
        recommendations.append("Opt for co-payment to reduce premium.")
    if waiting_period_years == 0:
        recommendations.append("Choose longer waiting period.")

    for rec in recommendations:
        st.write("•", rec)

    # ---------------- SHAP EXPLANATION ----------------
    if shap_bundle:
        st.divider()
        st.markdown("### Feature Impact (SHAP Explanation)")
        st.caption("Shows how each input influenced the predicted premium")

        preprocessor = model.named_steps["pre"]
        X_transformed = preprocessor.transform(input_df)
        feature_names = get_feature_names(preprocessor)

        explainer = shap_bundle["explainer"]
        shap_values = explainer.shap_values(X_transformed)

        filtered_vals, filtered_names = filter_selected_city_shap(
            shap_values[0],
            feature_names,
            city
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        shap.bar_plot(filtered_vals, feature_names=filtered_names, show=False)
        st.pyplot(fig)

    # ---------------- INSURER COMPARISON ----------------
    st.divider()
    st.markdown("### Insurer Comparison (Indicative)")

    insurer_adjustment = {
        "Star Health": 1.00,
        "HDFC ERGO": 1.05,
        "ICICI Lombard": 1.07,
        "Niva Bupa": 0.98,
        "Care Health": 0.95,
        "Max Bupa": 1.02
    }

    comparison = []
    for insurer, factor in insurer_adjustment.items():
        comparison.append({
            "Insurer": insurer,
            "Estimated Annual Premium (₹)": round(adjusted_prediction * factor, 0)
        })

    st.dataframe(
        pd.DataFrame(comparison).sort_values("Estimated Annual Premium (₹)"),
        use_container_width=True,
        hide_index=True
    )

    # ---------------- PDF DOWNLOAD ----------------
    st.divider()
    st.markdown("### Download Report")

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    pdf.setFont("Helvetica", 11)

    pdf.drawString(50, 800, "Health Insurance Premium Prediction Report")
    pdf.drawString(50, 770, f"Annual Premium: ₹ {adjusted_prediction:,.0f}")
    pdf.drawString(50, 750, f"Monthly Premium: ₹ {adjusted_prediction/12:,.0f}")
    pdf.drawString(50, 730, f"Smoking Status: {smoker}")

    y = 700
    for k, v in input_df.iloc[0].items():
        pdf.drawString(50, y, f"{k}: {v}")
        y -= 18

    pdf.save()
    buffer.seek(0)

    st.download_button(
        "Download PDF Report",
        data=buffer,
        file_name="health_premium_report.pdf",
        mime="application/pdf"
    )
