import streamlit as st
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

# Load model and dataset
@st.cache_resource
def load_model():
    return joblib.load("career_recommender.pkl"), joblib.load("vectorizer.pkl")

model, vectorizer = load_model()

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
st.write("Select your skills and rate your expertise!")

# Select skills from dropdown
selected_category = st.selectbox("Select a Skill Category", list(skill_categories.keys()))
selected_skills = st.multiselect("Choose Skills", skill_categories[selected_category])

# Rating skills
user_ratings = {}
for skill in selected_skills:
    user_ratings[skill] = st.slider(f"Rate your expertise in {skill}", 0, 5, 2)

if st.button("Analyze My Skills") and user_ratings:
    # Convert ratings into a DataFrame
    user_skills_df = pd.DataFrame([user_ratings])

    # Normalize the ratings
    scaler = MinMaxScaler()
    user_skills_scaled = scaler.fit_transform(user_skills_df)

    # Find closest matching careers
    _, indices = model.kneighbors(user_skills_scaled)

    # Display recommendations
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

    # Show the highest-rated skills
    sorted_skills = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)
    st.subheader("🔬 Your Strongest Skills")
    for skill, rating in sorted_skills:
        st.write(f"✅ **{skill}** - Rated {rating}/5")

    st.subheader("📘 Skill Descriptions")
    for skill, rating in sorted_skills:
        st.write(f"### {skill}")
        st.write(f"🔹 **Description**: Learn more about {skill} in these [Google Search Results](https://www.google.com/search?q={skill}+skill+description)")
        st.write("---")

# About the App
st.sidebar.header("About This App")
st.sidebar.info("This AI-powered advisor helps users identify the best career based on their skill ratings and provides learning resources to enhance their expertise.")

