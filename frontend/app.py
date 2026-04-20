import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from backend import load_data, prepare_dataframe, filter_dataset, ResumeMatcher

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = "data/resume_dataset_1200.csv"

df = load_data(DATA_PATH)
df = prepare_dataframe(df)

matcher = ResumeMatcher(df)

# UI
st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("📄 Resume Matching System")

st.sidebar.header("Filters")

skills = st.sidebar.text_input("Skills (comma separated)")
experience = st.sidebar.number_input("Experience", 0, 20, 0)
field = st.sidebar.text_input("Field of Study")
degree = st.sidebar.text_input("Degree (e.g., bachelor, master)")

top_n = st.sidebar.slider("Top Results", 1, 10, 5)

# SEARCH
if st.sidebar.button("Search"):

    user_skills = [s.strip().lower() for s in skills.split(",") if s.strip()]

    query = f"Skills: {skills} Field: {field} Degree: {degree}"

    filtered_df = filter_dataset(df, experience, degree, field)

    st.write(f"### Filtered Results: {len(filtered_df)}")

    if filtered_df.empty:
        st.error("❌ No matching resumes found. Try relaxing filters.")
    else:
        results = matcher.search(query, filtered_df, user_skills, top_n)

        st.subheader("Top Matching Resumes")

        for r in results:
            st.markdown(f"### Row ID: {r['row_id']} | Score: {r['score']}%")
            st.write(f"📊 Skill Match: {r['skill_ratio']}")
            st.write("✅ Matched Skills:", r["matched_skills"])
            st.write("❌ Missing Skills:", r["missing_skills"])
            st.write("📌 Preview:")
            st.write(r["resume"][:300] + "...")
            st.markdown("---")