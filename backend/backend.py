import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

def load_data(path):
    return pd.read_csv(path)

def safe_col(df, name):
    return df[name].astype(str) if name in df.columns else ""

# PREPROCESSING
def preprocess(text):
    text = str(text).lower()

    text = re.sub(r"[^a-z0-9+\s]", " ", text)

    # coreference normalization
    text = re.sub(r"\b(he|she|they|him|her|their)\b", "candidate", text)

    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]

    return " ".join(words)

skills_list = [
    "python", "java", "c++", "sql", "machine learning",
    "deep learning", "data science", "nlp", "pandas",
    "numpy", "aws", "git", "tensorflow", "react",
    "node js", "mongodb", "linux"
]

def extract_skills(text):
    text = text.lower()

    text = re.sub(r"[^a-z0-9+\s]", " ", text)

    words = text.split()

    found_skills = []

    for skill in skills_list:
        skill_tokens = skill.split()

        if len(skill_tokens) == 1:
            if skill in words:
                found_skills.append(skill)
        else:
            if skill in text:
                found_skills.append(skill)

    return found_skills


def prepare_dataframe(df):
    df["resume_text"] = (
        "Skills: " + safe_col(df, "Skills") +
        " Experience: " + safe_col(df, "Experience") +
        " Job: " + safe_col(df, "Current_Job") +
        " Education: " + safe_col(df, "Education") +
        " Degree: " + safe_col(df, "Degrees")
    )

    df["resume_clean"] = df["resume_text"].apply(preprocess)

    return df

# ---------------------------
# FILTER
def filter_dataset(df, experience, degree, field):
    filtered = df.copy()

    if experience > 0 and "Experience" in filtered.columns:
        filtered["Experience"] = pd.to_numeric(filtered["Experience"], errors='coerce')
        filtered = filtered[filtered["Experience"] >= experience]

    if degree and "Degrees" in filtered.columns:
        degree = degree.lower().strip()
        filtered = filtered[
            filtered["Degrees"].astype(str).str.lower().str.contains(degree, na=False)
        ]

    if field and "Field_of_Study" in filtered.columns:
        filtered = filtered[
            filtered["Field_of_Study"].astype(str).str.lower().str.contains(field.lower(), na=False)
        ]

    return filtered

# ---------------------------
class ResumeMatcher:
    def __init__(self, df):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
        self.vectorizer.fit(df["resume_clean"])

    def search(self, query, data, user_skills, top_n=5):

        if data.empty:
            return []

        query_clean = preprocess(query)

        query_vec = self.vectorizer.transform([query_clean])
        resume_vecs = self.vectorizer.transform(data["resume_clean"])

        similarities = cosine_similarity(query_vec, resume_vecs)[0]
        top_indices = similarities.argsort()[::-1][:top_n]

        results = []

        for idx in top_indices:
            row = data.iloc[idx]
            resume_text = row["resume_text"]

            r_skills = extract_skills(resume_text)

            matched = list(set(r_skills) & set(user_skills))
            missing = list(set(user_skills) - set(r_skills)) if user_skills else []

            semantic_score = similarities[idx]
            skill_score = (len(matched) / len(user_skills)) if user_skills else 0

            final_score = (0.6 * semantic_score) + (0.4 * skill_score)

            # penalty 
            if user_skills and len(matched) == 0:
                final_score *= 0.3

            final_score = round(final_score * 100, 2)

            results.append({
                "row_id": row.name,
                "score": final_score,
                "resume": resume_text,
                "matched_skills": matched,
                "missing_skills": missing,
                "skill_ratio": f"{len(matched)}/{len(user_skills)}" if user_skills else "N/A"
            })

        return results