
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

st.title("Employee Recruitment Model")

uploaded_file = st.file_uploader("/content/recruitment_data.csv", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write(" Dataset Preview")
        st.dataframe(df.head())

        if df.empty or "HiringDecision" not in df.columns:
            st.error("Dataset is empty or missing 'HiringDecision' column.")
            st.stop()

        df.dropna(inplace=True)
        X = df.drop("HiringDecision", axis=1)
        y = df["HiringDecision"]

        categorical_cols = X.select_dtypes(include=["object"]).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        st.write("Predict Hiring Decision")
        user_input = {}
        for feature in X.columns:
            if feature in categorical_cols:
                value = st.selectbox(f"Select value for {feature}", df[feature].unique().tolist())
            else:
                value = st.number_input(f"Enter value for {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].mean()))
            user_input[feature] = value

        user_df = pd.DataFrame([user_input])
        missing_cols = set(X.columns) - set(user_df.columns)
        for col in missing_cols:
            user_df[col] = 0
        user_df = scaler.transform(user_df)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine": SVC(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        if st.button("Predict Hiring Decision"):
            selected_model = RandomForestClassifier()
            selected_model.fit(X_train, y_train)
            prediction = selected_model.predict(user_df)
            pred_class = label_encoder.inverse_transform(prediction)
            st.write(f"**Predicted Hiring Decision:** {pred_class[0]}")

        selected_model_name = st.selectbox("Select a classification model", list(models.keys()))
        selected_model = models[selected_model_name]

        st.write("Model Training and Evaluation")
        selected_model.fit(X_train, y_train)
        y_pred = selected_model.predict(X_test)

        train_accuracy = accuracy_score(y_train, selected_model.predict(X_train))
        test_accuracy = accuracy_score(y_test, y_pred)

        st.write(f"**Training Accuracy:** {train_accuracy:.2f}")
        st.write(f"**Testing Accuracy:** {test_accuracy:.2f}")

        if st.button("Show Classification Report"):
            st.text(classification_report(y_test, y_pred))

        if st.button("Compare Model Accuracies"):
            accuracy_results = {}
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy_results[model_name] = accuracy_score(y_test, y_pred)
            
            plt.figure(figsize=(8, 5))
            sns.barplot(x=list(accuracy_results.keys()), y=list(accuracy_results.values()), palette="coolwarm")
            plt.ylabel("Accuracy")
            plt.title("Comparison of Model Accuracies")
            st.pyplot(plt)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

