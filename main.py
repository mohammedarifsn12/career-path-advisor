import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load pre-trained model & vectorizer
@st.cache_resource
def load_model():
    return joblib.load("career_recommender.pkl"), joblib.load("vectorizer.pkl")

model, vectorizer = load_model()

# Load career dataset
@st.cache_data
def load_data():
    return pd.read_csv("career_data_updated.csv")

career_data = load_data()

# Streamlit UI
st.title("🎯 AI-Based Career Path Advisor")
st.write("Personalized career recommendations with learning resources!")

# User Input
skills = st.text_area("Enter your skills & interests (comma-separated)", "Data Science, Machine Learning, SQL")

if st.button("Find My Career Path"):
    if skills:
        # Transform input text
        input_vector = vectorizer.transform([skills])

        # Find closest career matches
        _, indices = model.kneighbors(input_vector)

        # Display recommendations
        st.subheader("🔍 Recommended Career Paths:")
        for idx in indices[0]:
            career = career_data.iloc[idx]
            st.write(f"### {career['Career']}")
            st.write(f"📌 **Description**: {career['Description']}")
            st.write(f"🛠 **Required Skills**: {career['Skills']}")
            
            # Display learning materials
            st.write("📚 **Learning Resources**:")
            st.write(f"- 🎓 **Course**: [{career['Course_Link']}]({career['Course_Link']})")
            st.write(f"- 📖 **Book**: {career['Book']}")
            st.write(f"- 🏅 **Certification**: {career['Certification']}")
            st.write("---")

# About the App
st.sidebar.header("About This App")
st.sidebar.info("This AI-powered career advisor helps users find the best career path based on their skills and interests, while also providing learning materials to get started.")
