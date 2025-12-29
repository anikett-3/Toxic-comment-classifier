import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Loading model & vectorizer ()
model = pickle.load(open("toxic_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

st.set_page_config(page_title="Toxic Comment Classifier")

st.title("🛑 Toxic Comment Classification")
st.write("Enter a comment to check whether it is toxic or non-toxic.")

comment = st.text_area("Enter your comment")

if st.button("Predict"):
    if comment.strip() == "":
        st.warning("Please enter a comment.")
    else:
        cleaned = clean_text(comment)
        vector = vectorizer.transform([cleaned])
        result = model.predict(vector)[0]

        if result == 1:
            st.error("🚨 Toxic Comment Detected")
        else:
            st.success("✅ Non-Toxic Comment")
