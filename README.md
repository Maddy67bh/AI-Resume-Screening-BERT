# AI Resume Screening using BERT

An AI-powered resume screening system that uses **BERT-based Natural Language Processing (NLP)** to analyze resumes and identify the most relevant candidates for a given job description.

## 🚀 Features

- Resume text extraction and preprocessing
- BERT-based semantic understanding
- Resume-to-job-description matching
- Candidate relevance scoring
- Automated resume screening
- Structured project architecture
- Support for dataset-based training and evaluation

## 🛠️ Technologies Used

- Python
- BERT
- Natural Language Processing (NLP)
- PyTorch / Transformers
- Pandas
- Scikit-learn
- FastAPI / Streamlit
- Git & GitHub

## 📁 Project Structure

```text
AI-Resume-Screening-BERT/
│
├── app/                 # Application interface
├── data/                # Dataset files
├── models/              # Trained/model-related files
├── src/                 # Source code
├── create_dataset.py    # Dataset creation script
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignored files
└── README.md            # Project documentation
```

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Maddy67bh/AI-Resume-Screening-BERT.git
cd AI-Resume-Screening-BERT
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

Run the application using the project's main application file or entry point.

Example:

```bash
python create_dataset.py
```

If the project uses an application server, run the appropriate entry point from the `app` directory.

## 🧠 How It Works

1. Resume documents are collected.
2. Resume text is extracted and cleaned.
3. Job descriptions are processed.
4. BERT generates contextual representations of the text.
5. Resume and job-description representations are compared.
6. Candidates receive relevance scores.
7. Recruiters can use the ranking to identify suitable candidates faster.

## 📊 Use Cases

- Automated candidate screening
- Recruitment automation
- HR analytics
- Resume ranking
- Candidate-job matching
- Intelligent recruitment systems

## 🔮 Future Improvements

- Add multilingual resume support
- Improve ranking using fine-tuned BERT
- Add recruiter dashboard
- Add explainable AI for screening decisions
- Integrate multiple resume formats
- Deploy the system as a cloud application

## 👩‍💻 Author

**Maddy67bh**

Built as an AI/NLP project focused on intelligent resume screening and candidate matching.
