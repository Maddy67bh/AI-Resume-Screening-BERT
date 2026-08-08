import pandas as pd
import os

data = [
    # Data Science
    ["Python pandas numpy machine learning SQL data analysis statistics", "Data Science"],
    ["Python scikit learn regression classification data visualization pandas", "Data Science"],
    ["SQL Python Tableau Power BI Excel data analyst dashboard", "Data Science"],
    ["Python machine learning EDA statistics matplotlib seaborn", "Data Science"],
    ["Data analysis Python SQL pandas numpy visualization", "Data Science"],

    # Deep Learning
    ["Python TensorFlow PyTorch CNN neural networks deep learning", "Deep Learning"],
    ["PyTorch TensorFlow neural network CNN computer vision", "Deep Learning"],
    ["Deep learning NLP LSTM CNN transformers Python", "Deep Learning"],
    ["TensorFlow Keras neural networks image classification", "Deep Learning"],
    ["PyTorch CNN RNN LSTM artificial intelligence", "Deep Learning"],

    # Web Development
    ["HTML CSS JavaScript React frontend web development", "Web Development"],
    ["React JavaScript HTML CSS Tailwind frontend developer", "Web Development"],
    ["JavaScript React Node.js REST API web application", "Web Development"],
    ["HTML CSS Bootstrap JavaScript responsive websites", "Web Development"],
    ["React Next.js JavaScript TypeScript frontend development", "Web Development"],

    # Backend Development
    ["Python Django Flask REST API PostgreSQL backend development", "Backend Development"],
    ["Java Spring Boot REST API MySQL backend developer", "Backend Development"],
    ["Node.js Express MongoDB REST API backend development", "Backend Development"],
    ["Python FastAPI PostgreSQL API development", "Backend Development"],
    ["Java Spring Hibernate SQL backend software development", "Backend Development"],

    # Cyber Security
    ["network security ethical hacking penetration testing Linux", "Cyber Security"],
    ["Cybersecurity Python Linux penetration testing vulnerability assessment", "Cyber Security"],
    ["firewall network security SIEM threat detection", "Cyber Security"],
    ["ethical hacking Kali Linux security testing", "Cyber Security"],
    ["SOC analyst SIEM incident response cybersecurity", "Cyber Security"],

    # Cloud / DevOps
    ["AWS Docker Kubernetes CI CD DevOps Linux", "Cloud DevOps"],
    ["Azure AWS cloud computing Docker Kubernetes", "Cloud DevOps"],
    ["Docker Jenkins GitHub Actions CI CD DevOps", "Cloud DevOps"],
    ["AWS EC2 S3 Lambda cloud engineer", "Cloud DevOps"],
    ["Kubernetes Docker Terraform AWS infrastructure", "Cloud DevOps"],
]

df = pd.DataFrame(data, columns=["resume_text", "job_category"])

os.makedirs("data", exist_ok=True)

df.to_csv("data/resumes.csv", index=False)

print("Dataset created successfully!")
print(f"Total records: {len(df)}")
print("\nCategories:")
print(df["job_category"].value_counts())