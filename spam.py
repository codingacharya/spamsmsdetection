import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained model and vectorizer
model = joblib.load("spam_model.pkl")  # Ensure this file exists
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("📩 Real-Time SMS Spam Detector")
st.write("Start typing a message to analyze it automatically.")

# User Input Box
user_input = st.text_area("Enter your SMS message:", height=100)

# Automatically analyze when text is entered
if user_input.strip():
    # Transform input text
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)[0]
    
    # Display Result
    if prediction == 'spam':
        st.error("🚨 This message is SPAM!")
    else:
        st.success("✅ This message is NOT spam.")
else:
    st.info("Waiting for input...")
