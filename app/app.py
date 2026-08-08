import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PyPDF2 import PdfReader


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = "models/resume_bert_model"


# =========================================================
# SKILLS DATABASE
# =========================================================

SKILLS = [
    "Python",
    "Java",
    "C++",
    "C#",
    "JavaScript",
    "TypeScript",
    "React",
    "Angular",
    "Vue",
    "HTML",
    "CSS",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "MongoDB",
    "Pandas",
    "NumPy",
    "Scikit-learn",
    "TensorFlow",
    "PyTorch",
    "Keras",
    "Deep Learning",
    "Machine Learning",
    "NLP",
    "Natural Language Processing",
    "Computer Vision",
    "Power BI",
    "Tableau",
    "Excel",
    "AWS",
    "Azure",
    "Google Cloud",
    "Docker",
    "Kubernetes",
    "Git",
    "GitHub",
    "FastAPI",
    "Django",
    "Flask",
    "REST API",
    "Linux",
    "Spark",
    "Hadoop",
    "Data Analysis",
    "Data Visualization",
    "Statistics"
]


# =========================================================
# JOB REQUIREMENTS
# =========================================================

JOB_REQUIREMENTS = {

    "Data Science": [
        "Python",
        "Pandas",
        "NumPy",
        "SQL",
        "Machine Learning",
        "Scikit-learn",
        "Statistics",
        "Data Visualization"
    ],

    "Deep Learning": [
        "Python",
        "NumPy",
        "PyTorch",
        "TensorFlow",
        "Deep Learning",
        "Machine Learning",
        "NLP",
        "Computer Vision"
    ],

    "Web Development": [
        "HTML",
        "CSS",
        "JavaScript",
        "React",
        "SQL",
        "Git"
    ],

    "Backend Development": [
        "Python",
        "Java",
        "SQL",
        "REST API",
        "FastAPI",
        "Django",
        "Git"
    ],

    "Cloud DevOps": [
        "AWS",
        "Azure",
        "Docker",
        "Kubernetes",
        "Linux",
        "Git"
    ],

    "Cyber Security": [
        "Linux",
        "Python",
        "Networking",
        "Git"
    ]
}


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================================================
# SKILL EXTRACTION
# =========================================================

def extract_skills(text):

    text_lower = text.lower()

    found_skills = []

    for skill in SKILLS:

        if skill.lower() in text_lower:

            found_skills.append(skill)

    return found_skills


# =========================================================
# JOB MATCH CALCULATION
# =========================================================

def calculate_job_match(
    predicted_label,
    detected_skills
):

    required_skills = JOB_REQUIREMENTS.get(
        predicted_label,
        []
    )

    if not required_skills:

        return 0.0, []

    detected_lower = [
        skill.lower()
        for skill in detected_skills
    ]

    matched_skills = []

    missing_skills = []

    for skill in required_skills:

        if skill.lower() in detected_lower:

            matched_skills.append(skill)

        else:

            missing_skills.append(skill)

    score = (
        len(matched_skills)
        / len(required_skills)
    ) * 100

    return float(score), missing_skills


# =========================================================
# RESUME SCORE
# =========================================================

def calculate_resume_score(
    resume_text,
    detected_skills,
    confidence
):

    score = 0.0

    # Skills component
    skill_score = min(
        len(detected_skills) * 4,
        40
    )

    score += skill_score

    # Resume length component
    word_count = len(
        resume_text.split()
    )

    if word_count >= 300:

        score += 20

    elif word_count >= 150:

        score += 15

    elif word_count >= 75:

        score += 10

    else:

        score += 5

    # Important sections
    important_sections = [
        "education",
        "experience",
        "skills",
        "project",
        "certification"
    ]

    section_count = 0

    text_lower = resume_text.lower()

    for section in important_sections:

        if section in text_lower:

            section_count += 1

    score += section_count * 4

    # AI confidence
    score += float(confidence) * 20

    return min(
        float(score),
        100.0
    )


# =========================================================
# HEADER
# =========================================================

st.title(
    "🤖 AI Resume Screening System"
)

st.write(
    "Deep Learning powered resume classification "
    "using DistilBERT."
)

st.divider()


# =========================================================
# LOAD MODEL
# =========================================================

try:

    tokenizer, model = load_model()

except Exception as e:

    st.error(
        "Model loading failed."
    )

    st.code(str(e))

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ System Information")

    st.write(
        "**Model:** DistilBERT"
    )

    st.write(
        "**Task:** Resume Classification"
    )

    st.write(
        "**Categories:** 6"
    )

    st.write(
        "**Framework:** PyTorch"
    )

    st.write(
        "**Interface:** Streamlit"
    )


# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📄 Upload Resume PDF",
    type=["pdf"]
)


# =========================================================
# ANALYZE
# =========================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button(
        "🚀 Analyze Resume",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "AI is analyzing the resume..."
        ):

            # =============================================
            # EXTRACT PDF TEXT
            # =============================================

            resume_text = extract_text_from_pdf(
                uploaded_file
            )

            if not resume_text.strip():

                st.error(
                    "Could not extract text from this PDF."
                )

                st.stop()


            # =============================================
            # TOKENIZE
            # =============================================

            inputs = tokenizer(
                resume_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )


            # =============================================
            # PREDICTION
            # =============================================

            with torch.no_grad():

                outputs = model(
                    **inputs
                )

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=1
                )

                predicted_class = torch.argmax(
                    probabilities,
                    dim=1
                ).item()


            # =============================================
            # PREDICTED LABEL
            # =============================================

            predicted_label = model.config.id2label[
                predicted_class
            ]

            confidence = float(
                probabilities[
                    0,
                    predicted_class
                ].item()
            )


            # =============================================
            # SKILLS
            # =============================================

            detected_skills = extract_skills(
                resume_text
            )


            # =============================================
            # JOB MATCH
            # =============================================

            job_match_score, missing_skills = (
                calculate_job_match(
                    predicted_label,
                    detected_skills
                )
            )


            # =============================================
            # RESUME SCORE
            # =============================================

            resume_score = calculate_resume_score(
                resume_text,
                detected_skills,
                confidence
            )


            # =============================================
            # RESULTS HEADER
            # =============================================

            st.divider()

            st.header(
                "🎯 AI Screening Results"
            )


            # =============================================
            # KPI CARDS
            # =============================================

            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Recommended Role",
                    predicted_label
                )


            with col2:

                st.metric(
                    "AI Confidence",
                    f"{confidence * 100:.2f}%"
                )


            with col3:

                st.metric(
                    "Job Match",
                    f"{job_match_score:.2f}%"
                )


            with col4:

                st.metric(
                    "Resume Score",
                    f"{resume_score:.2f}/100"
                )


            st.divider()


            # =============================================
            # CONFIDENCE
            # =============================================

            st.subheader(
                "🤖 AI Confidence"
            )

            st.progress(
                float(confidence)
            )


            # =============================================
            # TOP PREDICTIONS
            # =============================================

            st.subheader(
                "📊 Top Predictions"
            )

            probs = probabilities[
                0
            ].tolist()


            results = []

            for index, probability in enumerate(
                probs
            ):

                label = model.config.id2label[
                    index
                ]

                results.append(
                    (
                        label,
                        float(probability)
                    )
                )


            results.sort(
                key=lambda x: x[1],
                reverse=True
            )


            for label, probability in results:

                st.write(
                    f"**{label}** — "
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    float(probability)
                )


            st.divider()


            # =============================================
            # SKILLS
            # =============================================

            col1, col2 = st.columns(2)


            with col1:

                st.subheader(
                    "🛠️ Detected Skills"
                )

                if detected_skills:

                    for skill in detected_skills:

                        st.success(
                            f"✓ {skill}"
                        )

                else:

                    st.warning(
                        "No known skills detected."
                    )


            # =============================================
            # MISSING SKILLS
            # =============================================

            with col2:

                st.subheader(
                    "❌ Missing Skills"
                )

                if missing_skills:

                    for skill in missing_skills:

                        st.warning(
                            f"• {skill}"
                        )

                else:

                    st.success(
                        "No major required skills missing."
                    )


            st.divider()


            # =============================================
            # JOB MATCH
            # =============================================

            st.subheader(
                "🎯 Job Match Analysis"
            )

            st.write(
                f"Recommended role: "
                f"**{predicted_label}**"
            )

            st.write(
                f"Job match score: "
                f"**{job_match_score:.2f}%**"
            )

            st.progress(
                float(job_match_score / 100)
            )


            # =============================================
            # RESUME SCORE
            # =============================================

            st.subheader(
                "📄 Resume Quality Score"
            )

            st.progress(
                float(resume_score / 100)
            )

            if resume_score >= 80:

                st.success(
                    "Excellent resume profile."
                )

            elif resume_score >= 60:

                st.info(
                    "Good resume, but there is room for improvement."
                )

            elif resume_score >= 40:

                st.warning(
                    "Average resume. Consider adding more skills and projects."
                )

            else:

                st.error(
                    "Resume needs significant improvement."
                )


            # =============================================
            # RESUME TEXT
            # =============================================

            with st.expander(
                "📄 View Extracted Resume Text"
            ):

                st.write(
                    resume_text
                )


            # =============================================
            # SUMMARY
            # =============================================

            st.divider()

            st.header(
                "📋 Screening Summary"
            )

            st.write(
                f"""
                **Resume:** {uploaded_file.name}

                **Predicted Role:** {predicted_label}

                **AI Confidence:** {confidence * 100:.2f}%

                **Job Match Score:** {job_match_score:.2f}%

                **Resume Score:** {resume_score:.2f}/100

                **Skills Detected:** {len(detected_skills)}

                **Missing Skills:** {len(missing_skills)}
                """
            )

else:

    st.info(
        "👆 Upload a PDF resume to begin AI screening."
    )