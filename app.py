import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# PAGE CONFIG
st.set_page_config(page_title="Titanic Survival App", page_icon="🚢", layout="wide")

# PATHS
base_path = "."
dataset_path = "TitanicAnalysis.csv"

# LOAD MODEL & ARTIFACTS
model = joblib.load("model_rf.pkl")
feature_columns = joblib.load("feature_column.pkl")
MODEL_ACCURACY = 0.85

# SIDEBAR
st.sidebar.title("🚢 Titanic Navigation")
menu = st.sidebar.selectbox("Go To", ["Dashboard", "Prediction", "EDA", "About"])

# DASHBOARD
if menu == "Dashboard":
    st.title("🚢 Titanic Survival Dashboard")

    df = pd.read_csv(dataset_path)

    total_passenger = len(df)
    survival_rate = df['Survived'].mean() * 100
    avg_fare = df['Fare'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Passengers", total_passenger)
    col2.metric("Survival Rate", f"{survival_rate:.2f}%")
    col3.metric("Average Fare", f"${avg_fare:.2f}")

    st.markdown("---")
    st.subheader("🛳 Passenger Class Impact")
    pclass_survival = df.groupby("Pclass")['Survived'].mean()*100
    st.write(pclass_survival)

    st.markdown("---")
    st.subheader("📊 Visual Evidence")
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    sns.countplot(x="Survived", data=df, ax=ax[0])
    ax[0].set_title("Survival Count")
    sns.countplot(x="Pclass", hue="Survived", data=df, ax=ax[1])
    ax[1].set_title("Survival by Pclass")
    st.pyplot(fig)

elif menu == "Prediction":

    st.title("🎯 Titanic Survival Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 1, 80, 25)
        sibsp = st.slider("Siblings/Spouses", 0, 10, 0)
        parch = st.slider("Parents/Children", 0, 10, 0)

    with col2:
        fare = st.slider("Fare", 0.0, 600.0, 50.0)
        embarked = st.selectbox("Embarked", ["C", "Q", "S"])

        st.caption("C → Cherbourg")
        st.caption("Q → Queenstown")
        st.caption("S → Southampton")

    # =========================
    # PREDICT BUTTON
    # =========================
    if st.button("Predict"):

        try:
            # ------------------------
            # CREATE INPUT
            # ------------------------
            input_data = pd.DataFrame([{
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "SibSp": sibsp,
                "Parch": parch,
                "Fare": fare,
                "Embarked": embarked
            }])

            # ------------------------
            # ENCODING
            # ------------------------
            input_data["Sex"] = input_data["Sex"].map({"male":1, "female":0})
            input_data["Embarked"] = input_data["Embarked"].map({"C":0, "Q":1, "S":2})

            # ------------------------
            # FEATURE ENGINEERING
            # ------------------------
            input_data["FamilySize"] = input_data["SibSp"] + input_data["Parch"]

            # input_data["IsAlone"] = 1
            # if input_data["FamilySize"][0] > 0:
            #     input_data["IsAlone"] = 0
            input_data["IsAlone"] = (input_data["FamilySize"] == 0).astype(int)

            # ------------------------
            # MATCH COLUMNS
            # ------------------------
            for col in feature_columns:
                if col not in input_data.columns:
                    input_data[col] = 0

            input_data = input_data[feature_columns]

            # ------------------------
            # PREDICTION
            # ------------------------
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.divider()

            if prediction == 1:
                st.success(f"🎉 Survived Probability: {probability*100:.2f}%")
            else:
                st.error(f"❌ Survival Probability: {probability*100:.2f}%")

        except Exception as e:
            st.error(f"Error: {e}")

# EDA
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")
    df = pd.read_csv(dataset_path)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.markdown("---")

    st.subheader("1️⃣ Survival Count")
    fig, ax = plt.subplots()
    sns.countplot(x="Survived", data=df, palette="Set2", ax=ax)
    ax.set_title("Survival Count (0=No, 1=Yes)")
    st.pyplot(fig)

    st.subheader("2️⃣ Survival by Gender")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Sex", hue="Survived", data=df, palette="Set1", ax=ax2)
    ax2.set_title("Survival by Gender")
    st.pyplot(fig2)

    st.subheader("3️⃣ Survival by Passenger Class")
    fig3, ax3 = plt.subplots()
    sns.countplot(x="Pclass", hue="Survived", data=df, palette="Set3", ax=ax3)
    ax3.set_title("Survival by Class (1,2,3)")
    st.pyplot(fig3)

    st.subheader("4️⃣ Embarked vs Survival")
    fig4, ax4 = plt.subplots()
    sns.countplot(x="Embarked", hue="Survived", data=df, palette="Set2", ax=ax4)
    ax4.set_title("Embarked Port vs Survival")
    st.pyplot(fig4)

# ABOUT
elif menu == "About":
    st.title("ℹ️ About This Project")
    st.subheader("📌 Project Objective")
    st.write("Predict Titanic passenger survival using Machine Learning.")

    st.subheader("⚙️ Feature Engineering Techniques")
    st.write("""
    - Handling Missing Values  
    - Label Encoding (manual mapping for prediction)  
    - Feature Scaling  
    - Derived Features (FamilySize)  
    """)

    st.subheader("🤖 Machine Learning Model")
    st.write("Random Forest Classifier")

    st.subheader("📈 Model Performance")
    st.write(f"Accuracy: {MODEL_ACCURACY*100:.2f}%")

    st.subheader("🛠 Technologies Used")
    st.write("""
    - Python  
    - Pandas  
    - NumPy  
    - Scikit-learn  
    - Streamlit  
    - Matplotlib & Seaborn  
    """)

