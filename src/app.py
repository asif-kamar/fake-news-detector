import streamlit as st
import joblib
import re

# Load the brain
model = joblib.load('models/news_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

st.title("📰 Fake News Detector")
st.subheader("Enter a news headline to verify its authenticity")

user_input = st.text_area("Paste news text here:")

if st.button("Analyze"):
    # Reuse your existing cleaning logic
    cleaned = user_input.lower()
    cleaned = re.sub("\\W"," ", cleaned)
    
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    
    if prediction[0] == 1:
        st.success("This news looks REAL. ✅")
    else:
        st.error("This news looks FAKE. ❌")