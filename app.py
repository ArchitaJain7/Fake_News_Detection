import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load model
model = pickle.load(open('src/model.pkl', 'rb'))
tfidf = pickle.load(open('src/vectorizer.pkl', 'rb'))

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word, pos='v') for word in text if word not in stop_words]
    return ' '.join(text)

# UI
st.title("📰 Fake News Detection System")
st.write("Enter a news article to check whether it is Real or Fake.")

input_text = st.text_area("Enter News Text:")

if st.button("Predict"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed = preprocess(input_text)
        vector = tfidf.transform([processed])
        prediction = model.predict(vector)

        decision = model.decision_function(vector)
        confidence = abs(decision[0])

        if prediction[0] == 1:
            st.success("REAL NEWS 🟢")
        else:
            st.error("FAKE NEWS 🔴")

        st.write(f"Confidence Score: {confidence:.2f}")