import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ==========================================
# MODEL PATH
# ==========================================

MODEL_PATH = "models/resume_bert_model"


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading trained model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


# ==========================================
# RESUME TEXT
# ==========================================

resume_text = """
I am a Data Science student with experience in
Python, Pandas, NumPy, SQL, Machine Learning,
Deep Learning and Data Visualization.

I have worked on projects involving
classification, regression, data analysis,
Power BI dashboards and predictive modeling.
"""


# ==========================================
# TOKENIZE
# ==========================================

inputs = tokenizer(
    resume_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
)


# ==========================================
# PREDICTION
# ==========================================

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


# ==========================================
# RESULT
# ==========================================

predicted_label = model.config.id2label[
    predicted_class
]

confidence = probabilities[
    0,
    predicted_class
].item()


print("\n========================================")
print("AI RESUME SCREENING RESULT")
print("========================================")

print(
    "\nPredicted Job Category:"
)

print(
    predicted_label
)

print(
    f"\nConfidence: {confidence * 100:.2f}%"
)

print("\n========================================")