import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Load model and vectorizer
@st.cache_resource
def load_model():
    return joblib.load("career_recommender.pkl"), joblib.load("vectorizer.pkl")

model, vectorizer = load_model()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("career_data_updated.csv")

career_data = load_data()

# Skill Categories
skill_categories = {
    "Computer Science": ["Python", "Java", "SQL", "Machine Learning", "Data Science", "Cybersecurity"],
    "Electronics": ["Embedded Systems", "IoT", "PCB Design", "VLSI", "Signal Processing"],
    "Marketing": ["SEO", "Social Media Marketing", "Content Writing", "Google Ads", "Market Research"],
    "Business & Management": ["Project Management", "Business Analytics", "Financial Modeling", "Negotiation"],
}

# Streamlit UI
st.title("🎯 AI-Based Career & Skill Advisor")
st.write("Select and rate your skills to get personalized career recommendations!")

# Dropdown for skill selection
selected_skills = st.multiselect("Select your skills:", 
                                 sum(skill_categories.values(), []))  # Flatten skill lists

# Rating system for selected skills
user_ratings = {}
for skill in selected_skills:
    user_ratings[skill] = st.slider(f"Rate your expertise in {skill} (0 = No Experience, 5 = Expert)", 0, 5, 2)

if st.button("Analyze My Skills"):
    if not selected_skills:
        st.warning("⚠️ Please select at least one skill before proceeding!")
    else:
        # **Step 1: Convert user input into a full skill vector**
        all_skills = sum(skill_categories.values(), [])  # List of all possible skills
        user_vector = [user_ratings.get(skill, 0) for skill in all_skills]  # Assign 0 to unselected skills

        # **Step 2: Reshape and scale the user input**
        user_vector = np.array(user_vector).reshape(1, -1)  # Convert to 2D array
        scaler = MinMaxScaler()
        user_vector_scaled = scaler.fit_transform(user_vector)

        # **Step 3: Get career recommendations**
        _, indices = model.kneighbors(user_vector_scaled)

        # **Step 4: Display recommended careers**
        st.subheader("🔍 Recommended Careers Based on Your Skills")
        for idx in indices[0]:
            career = career_data.iloc[idx]
            st.write(f"### {career['Career']}")
            st.write(f"📌 **Description**: {career['Description']}")
            st.write(f"🛠 **Key Skills**: {career['Skills']}")
            st.write(f"📚 **Learning Resources:**")
            st.write(f"- 🎓 **Course**: [{career['Course_Link']}]({career['Course_Link']})")
            st.write(f"- 📖 **Book**: {career['Book']}")
            st.write(f"- 🏅 **Certification**: {career['Certification']}")
            st.write("---")

        # **Step 5: Show the highest-rated skills**
        sorted_skills = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)
        st.subheader("🔬 Your Strongest Skills")
        for skill, rating in sorted_skills[:5]:
            st.write(f"✅ **{skill}** - Rated {rating}/5")

        # **Step 6: Provide skill descriptions**
        st.subheader("📘 Skill Descriptions")
        for skill, rating in sorted_skills[:5]:
            st.write(f"### {skill}")
            st.write(f"🔹 **Description**: Learn more about {skill} in these [Google Search Results](https://www.google.com/search?q={skill}+skill+description)")
            st.write("---")

# Sidebar - About the App
st.sidebar.header("About This App")
st.sidebar.info("This AI-powered advisor helps users identify the best career based on their skill ratings and provides learning resources to enhance their expertise.")


