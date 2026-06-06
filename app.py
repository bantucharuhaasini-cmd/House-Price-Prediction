import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -------------------------------
# LOAD MODEL
# -------------------------------
model = joblib.load("model/house_price_pipeline.pkl")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------
# HEADER
# -------------------------------
st.title("🏠 AI House Price Prediction System")
st.caption("End-to-End ML Pipeline + Interactive Dashboard")

st.markdown("---")

# -------------------------------
# TABS 
# -------------------------------
tab1, tab2, tab3 = st.tabs(["🏡 Predict", "📊 Insights", "ℹ️ About"])

# =====================================================
# TAB 1 - INPUT + PREDICTION
# =====================================================
with tab1:

    st.subheader("Enter Property Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (sq ft)", 1000, 10000, 3000)
        bedrooms = st.number_input("Bedrooms", 1, 10, 3)
        bathrooms = st.number_input("Bathrooms", 1, 10, 2)
        stories = st.number_input("Stories", 1, 5, 2)
        parking = st.number_input("Parking", 0, 5, 1)

    with col2:
        mainroad = st.selectbox("Main Road", ["yes", "no"])
        guestroom = st.selectbox("Guest Room", ["yes", "no"])
        basement = st.selectbox("Basement", ["yes", "no"])
        hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
        airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
        prefarea = st.selectbox("Preferred Area", ["yes", "no"])
        furnishingstatus = st.selectbox(
            "Furnishing Status",
            ["furnished", "semi-furnished", "unfurnished"]
        )

    input_data = pd.DataFrame([[
        area, bedrooms, bathrooms, stories,
        mainroad, guestroom, basement,
        hotwaterheating, airconditioning,
        parking, prefarea, furnishingstatus
    ]], columns=[
        "area", "bedrooms", "bathrooms", "stories",
        "mainroad", "guestroom", "basement",
        "hotwaterheating", "airconditioning",
        "parking", "prefarea", "furnishingstatus"
    ])

    st.markdown("---")

    if st.button("🔮 Predict Price"):

        prediction = model.predict(input_data)[0]

        # -------------------------------
        # KPI STYLE OUTPUT
        # -------------------------------
        c1, c2, c3 = st.columns(3)

        c1.metric("Area", f"{area} sq ft")
        c2.metric("Bedrooms", bedrooms)
        c3.metric("Estimated Price", f"₹ {int(prediction):,}")

        # -------------------------------
        # PRICE CATEGORY
        # -------------------------------
        if prediction < 3000000:
            st.info("💡 Budget Property")
        elif prediction < 6000000:
            st.warning("🏠 Mid Range Property")
        else:
            st.success("🏡 Premium Property")

        st.balloons()

# =====================================================
# TAB 2 - INSIGHTS 
# =====================================================
with tab2:

    st.subheader("📊 Model Insights")

    st.write("This section explains how your model behaves.")

    # Get preprocessor from pipeline
    preprocessor = model.named_steps['preprocessor']

    # Get feature names after encoding
    feature_names = preprocessor.get_feature_names_out()

    # Get regressor importance
    importances = model.named_steps['regressor'].feature_importances_

    importance_df=pd.DataFrame({
        "Feature":feature_names,
        "Importance":importances
    })

    importance_df=importance_df.sort_values(by="Importance",ascending=False).head(10)  

    st.subheader("📊 Real Feature Importance (Model Learned)")

    st.bar_chart(importance_df.set_index("Feature"))

    st.markdown("---")

    st.info("""
    📌 Model Type: Gradient Boosting Regressor  
    📌 Pipeline: Yes (Preprocessing + Model combined)  
    📌 Problem Type: Regression  
    📌 Output: Continuous Price Prediction  
    """)

# =====================================================
# TAB 3 - ABOUT
# =====================================================
with tab3:

    st.subheader("ℹ️ About This Project")

    st.markdown("""
    ### 🚀 What this project does:
    - Predicts house prices using Machine Learning
    - Uses a full sklearn Pipeline
    - Handles preprocessing automatically
    - Deploys as a web app using Streamlit

    ### 🧠 Skills shown:
    - Machine Learning
    - Feature Engineering
    - Model Pipeline
    - Web Deployment

    ### 💼 Why this is important:
    This is a **real-world ML system**, similar to what companies use.
    """)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with  using Streamlit + ML Pipeline + Gradient Boosting")