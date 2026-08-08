import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATASET_PATH = "data/resumes.csv"

df = pd.read_csv(DATASET_PATH)

print("\n========================================")
print("DATASET LOADED")
print("========================================")

print("Total records:", len(df))

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "resume_text",
    "job_category"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# Remove empty rows

df = df.dropna(
    subset=[
        "resume_text",
        "job_category"
    ]
).reset_index(drop=True)


print("\nRecords after cleaning:", len(df))


# ============================================================
# 3. ENCODE JOB CATEGORIES
# ============================================================

label_encoder = LabelEncoder()

df["label"] = label_encoder.fit_transform(
    df["job_category"]
)


print("\n========================================")
print("JOB CATEGORIES")
print("========================================")

for i, category in enumerate(
    label_encoder.classes_
):

    print(
        f"{i} = {category}"
    )


# ============================================================
# 4. CREATE MODELS DIRECTORY
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# 5. SAVE LABEL MAPPING
# ============================================================

label_mapping = {
    int(i): category
    for i, category in enumerate(
        label_encoder.classes_
    )
}


with open(
    "models/label_mapping.txt",
    "w",
    encoding="utf-8"
) as f:

    for i, category in label_mapping.items():

        f.write(
            f"{i}: {category}\n"
        )


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


print("\n========================================")
print("DATA SPLIT")
print("========================================")

print(
    "Training records:",
    len(train_df)
)

print(
    "Testing records:",
    len(test_df)
)


# ============================================================
# 7. CREATE HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[
        [
            "resume_text",
            "label"
        ]
    ],
    preserve_index=False
)


test_dataset = Dataset.from_pandas(
    test_df[
        [
            "resume_text",
            "label"
        ]
    ],
    preserve_index=False
)


# ============================================================
# 8. LOAD DISTILBERT TOKENIZER
# ============================================================

MODEL_NAME = "distilbert-base-uncased"

print("\n========================================")
print("LOADING TOKENIZER")
print("========================================")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 9. TOKENIZATION FUNCTION
# ============================================================

def tokenize_function(examples):

    return tokenizer(
        examples["resume_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


# ============================================================
# 10. TOKENIZE TRAIN DATA
# ============================================================

print("\nTokenizing training data...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)


# ============================================================
# 11. TOKENIZE TEST DATA
# ============================================================

print("\nTokenizing test data...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)


# ============================================================
# 12. LOAD DISTILBERT MODEL
# ============================================================

print("\n========================================")
print("LOADING DISTILBERT MODEL")
print("========================================")


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(
        label_encoder.classes_
    ),
    id2label=label_mapping,
    label2id={
        category: i
        for i, category
        in label_mapping.items()
    }
)


# ============================================================
# 13. METRICS
# ============================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predicted_labels
    )

    return {
        "accuracy": accuracy
    }


# ============================================================
# 14. TRAINING CONFIGURATION
# ============================================================

print("\n========================================")
print("CREATING TRAINING CONFIGURATION")
print("========================================")


training_args = TrainingArguments(

    output_dir="./models/bert_results",

    eval_strategy="epoch",

    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    num_train_epochs=5,

    weight_decay=0.01,

    logging_steps=1,

    report_to="none",

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",

    greater_is_better=True
)


# ============================================================
# 15. CREATE TRAINER
# ============================================================

print("\n========================================")
print("CREATING TRAINER")
print("========================================")


trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics
)


# ============================================================
# 16. START TRAINING
# ============================================================

print("\n")
print("========================================")
print("STARTING DISTILBERT TRAINING")
print("========================================")
print("\n")


trainer.train()


# ============================================================
# 17. EVALUATE MODEL
# ============================================================

print("\n")
print("========================================")
print("EVALUATING MODEL")
print("========================================")
print("\n")


results = trainer.evaluate()


print(
    "Evaluation Results:"
)

print(results)


# ============================================================
# 18. PREDICTIONS
# ============================================================

print("\n")
print("========================================")
print("GENERATING PREDICTIONS")
print("========================================")
print("\n")


prediction_output = trainer.predict(
    test_dataset
)


predicted_labels = np.argmax(
    prediction_output.predictions,
    axis=1
)


true_labels = (
    prediction_output.label_ids
)


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("========================================")
print("CLASSIFICATION REPORT")
print("========================================")
print("\n")


print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 20. SAVE MODEL
# ============================================================

MODEL_PATH = "models/resume_bert_model"


os.makedirs(
    MODEL_PATH,
    exist_ok=True
)


print("\n")
print("========================================")
print("SAVING MODEL")
print("========================================")
print("\n")


trainer.save_model(
    MODEL_PATH
)


tokenizer.save_pretrained(
    MODEL_PATH
)


# ============================================================
# 21. SAVE LABEL ENCODER
# ============================================================

import json


with open(
    "models/label_mapping.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        label_mapping,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 22. FINAL MESSAGE
# ============================================================

print("\n")
print("========================================")
print("MODEL TRAINING COMPLETED!")
print("========================================")

print(
    "\nModel saved at:"
)

print(
    MODEL_PATH
)

print("\nLabel mapping saved at:")

print(
    "models/label_mapping.json"
)

print("\n========================================")
print("PROJECT TRAINING SUCCESSFUL")
print("========================================")