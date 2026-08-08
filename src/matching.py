from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def calculate_match_score(resume_text, job_description):

    embeddings = model.encode(
        [resume_text, job_description]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    score = max(0, min(100, similarity * 100))

    return round(score, 2)


def extract_skills(text):

    skills_database = [
        "Python",
        "SQL",
        "Pandas",
        "NumPy",
        "Machine Learning",
        "Deep Learning",
        "TensorFlow",
        "PyTorch",
        "Scikit-learn",
        "Power BI",
        "Tableau",
        "Excel",
        "Java",
        "JavaScript",
        "React",
        "Node.js",
        "Django",
        "Flask",
        "FastAPI",
        "AWS",
        "Azure",
        "Docker",
        "Kubernetes",
        "Git",
        "GitHub",
        "Linux",
        "NLP",
        "CNN",
        "LSTM",
        "Transformers"
    ]

    text_lower = text.lower()

    found = []

    for skill in skills_database:

        if skill.lower() in text_lower:
            found.append(skill)

    return found


def find_missing_skills(resume_text, job_description):

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    missing = [
        skill
        for skill in job_skills
        if skill not in resume_skills
    ]

    return resume_skills, missing