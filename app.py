import streamlit as st
import pickle

# Load model and vectorizer
with open("naive_bayes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("count_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# App title
st.title("📧 Spam Email Classifier")
st.markdown("Enter an email message below to check if it's **Spam** or **Not Spam**.")

# Text input
email_input = st.text_area("✉️ Enter your email message here:", height=150)

# Classify on button click
if st.button("Classify"):
    if not email_input.strip():
        st.warning("Please enter a message first.")
    else:
        # Transform input
        email_transformed = vectorizer.transform([email_input])
        prediction = model.predict(email_transformed)[0]
        probability = model.predict_proba(email_transformed)[0][1]  # prob it's spam

        if prediction == 1:
            st.error(f"🚫 Spam ")
        else:
            st.success(f"✅ Not Spam ")
