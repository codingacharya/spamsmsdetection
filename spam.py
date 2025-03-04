import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load pre-trained model and vectorizer
model = joblib.load("spam_model.pkl")  # Ensure you have this saved
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("📩 SMS Spam Detector")
st.write("Enter a message below to check if it's spam or not.")

# User input
user_input = st.text_area("Type your SMS here:")

if st.button("Check Message"):
    if user_input.strip():
        # Transform input
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        
        # Display result
        if prediction == 'spam':
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT spam.")
    else:
        st.warning("Please enter a message to check.")

# File Upload for Bulk Prediction
st.subheader("📂 Bulk SMS Spam Detection")
uploaded_file = st.file_uploader("Upload a CSV file with a 'message' column", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'message' in df.columns:
        df['prediction'] = model.predict(vectorizer.transform(df['message']))
        st.write(df)
        st.download_button("Download Results", df.to_csv(index=False), file_name="sms_predictions.csv")
    else:
        st.error("CSV must contain a 'message' column.")
