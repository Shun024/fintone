"""
Train FinTone-DistilBERT locally and push to HuggingFace Hub.
"""

import os
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset as HFDataset
import pandas as pd
from huggingface_hub import login

# Login
HF_TOKEN = input("Paste your HuggingFace WRITE token: ")
login(token=HF_TOKEN)

# Load data
print("Loading dataset...")
ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
df = pd.DataFrame(ds["train"])
label_map = {0: "negative", 1: "bullish", 2: "neutral"}
df["label_text"] = df["label"].map(label_map).replace("bullish", "positive")
df = df.rename(columns={"text": "sentence"})

label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {0: "negative", 1: "neutral", 2: "positive"}

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label_text"]
)

# Tokenise
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

train_hf = HFDataset.from_pandas(train_df[["sentence", "label_text"]].reset_index(drop=True))
test_hf = HFDataset.from_pandas(test_df[["sentence", "label_text"]].reset_index(drop=True))

train_hf = train_hf.map(lambda x: {"label": label2id[x["label_text"]]})
test_hf = test_hf.map(lambda x: {"label": label2id[x["label_text"]]})
train_hf = train_hf.map(tokenize, batched=True)
test_hf = test_hf.map(tokenize, batched=True)
train_hf.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_hf.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}

training_args = TrainingArguments(
    output_dir="./fintone-distilbert",
    num_train_epochs=3,              # reduced from 5
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="no",              # don't save checkpoints to disk
    load_best_model_at_end=False,    # can't load best without saving
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=test_hf,
    compute_metrics=compute_metrics,
)

print("Training FinTone-DistilBERT...")
trainer.train()
print("Training complete!")

# Run SHAP immediately while model is in memory
print("\nRunning SHAP analysis...")
import shap
import numpy as np
from pathlib import Path

Path("outputs/shap").mkdir(parents=True, exist_ok=True)

import torch
model.eval()

def predict_proba(texts):
    inputs = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).numpy()
    return probs

examples = [
    "The company reported record profits this quarter.",
    "Sales declined significantly due to weak market conditions.",
    "The merger is expected to close in Q3 pending regulatory approval.",
    "Operating costs increased by 15% year over year.",
    "The firm announced a special dividend for shareholders.",
    "Revenue fell short of analyst expectations by a wide margin.",
    "Barclays CET1 ratio improved to 13.8% reflecting strong capital generation.",
    "Credit losses widened as economic conditions deteriorated.",
]

print("Computing SHAP values...")
explainer = shap.Explainer(
    predict_proba,
    shap.maskers.Text(r"\W+"),
    output_names=["negative", "neutral", "positive"],
)
shap_values = explainer(examples, fixed_context=1)

# Save plots
import matplotlib.pyplot as plt

for i, sentence in enumerate(examples[:4]):
    pred = predict_proba([sentence])[0]
    pred_label = id2label[np.argmax(pred)]
    pred_score = pred[np.argmax(pred)]
    print(f"\n[{i+1}] {sentence[:60]}")
    print(f"     Prediction: {pred_label} ({pred_score:.2%})")

    plt.figure(figsize=(12, 2))
    shap.plots.text(shap_values[i, :, np.argmax(pred)], display=False)
    plt.tight_layout()
    plt.savefig(f"outputs/shap/shap_text_{i+1}_{pred_label}.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"     Saved: outputs/shap/shap_text_{i+1}_{pred_label}.png")

# Global word importance
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values[:, :, 2].mean(0), show=False, max_display=15)
plt.title("Top Words Driving POSITIVE Sentiment")
plt.tight_layout()
plt.savefig("outputs/shap/shap_positive_words.png", dpi=100, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values[:, :, 0].mean(0), show=False, max_display=15)
plt.title("Top Words Driving NEGATIVE Sentiment")
plt.tight_layout()
plt.savefig("outputs/shap/shap_negative_words.png", dpi=100, bbox_inches="tight")
plt.close()

print("\nSHAP analysis complete. Outputs saved to outputs/shap/")
print("\nTop SHAP words for POSITIVE sentiment:")
pos_shap = shap_values[:, :, 2].mean(0)
top_pos = sorted(zip(pos_shap.feature_names, pos_shap.values), key=lambda x: x[1], reverse=True)[:10]
for word, score in top_pos:
    print(f"  {word:<20} {score:+.4f}")

print("\nTop SHAP words for NEGATIVE sentiment:")
neg_shap = shap_values[:, :, 0].mean(0)
top_neg = sorted(zip(neg_shap.feature_names, neg_shap.values), key=lambda x: x[1], reverse=True)[:10]
for word, score in top_neg:
    print(f"  {word:<20} {score:+.4f}")