# app.py

import streamlit as st
import pandas as pd 
import sklearn
print(sklearn.__version__)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

st.title("Online News Popularity Prediction App")
st.write("This app predicts whether a news article is **popular** or **not popular** based on its features.")

uploaded_file = st.file_uploader("C:\\Users\\aarya\\OneDrive\\Desktop\\onlinenewspopularity\\OnlineNewsPopularity.csv")


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Data preprocessing
    df.columns = df.columns.str.strip()
    df = df.drop(['url', 'timedelta'], axis=1)
    df['popular'] = (df['shares'] > 1400).astype(int)
    df = df.drop('shares', axis=1)

    X = df.drop('popular', axis=1)
    y = df['popular']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)

    st.subheader("Classification Report")
    st.write(pd.DataFrame(report).transpose())
else:
    st.warning("Please upload a CSV file to proceed.")
