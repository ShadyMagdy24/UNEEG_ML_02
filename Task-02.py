import streamlit as st
import joblib

# Load trained models and vectorizer
log_reg = joblib.load("logistic_regression.pkl")
random_forest = joblib.load("random_forest.pkl")
svm_model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Set Streamlit page title
st.title("🔍 Sentiment Analysis App")
st.write("Enter text below to analyze its sentiment using pre-trained models.")

# User input
user_input = st.text_area("Enter your text here:", "")

# Model selection
model_choice = st.selectbox("Choose a model:", ["Logistic Regression", "Random Forest", "SVM"])

# Prediction function
def predict_sentiment(text, model):
    transformed_text = vectorizer.transform([text])  # Convert text to TF-IDF
    prediction = model.predict(transformed_text)[0]
    return prediction

# Predict button
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        if model_choice == "Logistic Regression":
            sentiment = predict_sentiment(user_input, log_reg)
        elif model_choice == "Random Forest":
            sentiment = predict_sentiment(user_input, random_forest)
        else:
            sentiment = predict_sentiment(user_input, svm_model)

        # Display sentiment result
        if sentiment == 1:
            st.success("😊 Positive Sentiment")
        elif sentiment == 0:
            st.info("😐 Neutral Sentiment")
        else:
            st.error("😠 Negative Sentiment")
