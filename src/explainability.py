"""
SHAP explainability for FinTone financial sentiment classifier.
Answers: which words drive positive/negative/neutral sentiment predictions?
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    pipeline,
)
from pathlib import Path


MODEL_ID = "SLYM06/fintone-distilbert-financial-sentiment"

# Label mapping
id2label = {0: "negative", 1: "neutral", 2: "positive"}

EXAMPLE_SENTENCES = [
    "The company reported record profits this quarter.",
    "Sales declined significantly due to weak market conditions.",
    "The merger is expected to close in Q3 pending regulatory approval.",
    "Operating costs increased by 15% year over year.",
    "The firm announced a special dividend for shareholders.",
    "Revenue fell short of analyst expectations by a wide margin.",
    "Barclays CET1 ratio improved to 13.8% reflecting strong capital generation.",
    "Credit losses widened as economic conditions deteriorated.",
    "The acquisition will strengthen our position in the UK retail market.",
    "The company faces potential bankruptcy proceedings after covenant breach.",
]


def load_model():
    """Load fine-tuned FinTone model from HuggingFace Hub."""
    print(f"Loading {MODEL_ID}...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_ID)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer


def run_shap_analysis(output_dir: str = "outputs/shap") -> None:
    """
    Run SHAP text explainability on FinTone.
    Uses shap.Explainer with the HuggingFace pipeline.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("FinTone — SHAP Text Explainability")
    print("=" * 50)

    # Build pipeline for SHAP
    clf_pipeline = pipeline(
        "text-classification",
        model=MODEL_ID,
        return_all_scores=True,
        device=-1,  # CPU
    )

    def predict_proba(texts):
        """Return probability array for SHAP."""
        results = clf_pipeline(list(texts))
        probs = []
        for r in results:
            # Sort by label order: negative, neutral, positive
            label_map = {item["label"]: item["score"] for item in r}
            probs.append([
                label_map.get("negative", 0),
                label_map.get("neutral", 0),
                label_map.get("positive", 0),
            ])
        return np.array(probs)

    print("\nInitialising SHAP explainer...")
    explainer = shap.Explainer(
        predict_proba,
        shap.maskers.Text(r"\W+"),
        output_names=["negative", "neutral", "positive"],
    )

    print(f"Computing SHAP values for {len(EXAMPLE_SENTENCES)} sentences...")
    shap_values = explainer(EXAMPLE_SENTENCES, fixed_context=1)

    # --- Plot 1: Text plots for each sentence ---
    print("\nGenerating individual text explanation plots...")
    for i, sentence in enumerate(EXAMPLE_SENTENCES[:5]):  # first 5
        pred = predict_proba([sentence])[0]
        pred_label = id2label[np.argmax(pred)]
        pred_score = pred[np.argmax(pred)]

        print(f"\n[{i+1}] {sentence[:60]}...")
        print(f"     Prediction: {pred_label} ({pred_score:.2%})")

        plt.figure(figsize=(12, 3))
        shap.plots.text(
            shap_values[i, :, np.argmax(pred)],
            display=False,
        )
        plt.title(
            f"Prediction: {pred_label} ({pred_score:.2%})\n{sentence[:80]}",
            fontsize=10,
        )
        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/shap_text_{i+1}_{pred_label}.png",
            dpi=120,
            bbox_inches="tight",
        )
        plt.close()
        print(f"     Saved: {output_dir}/shap_text_{i+1}_{pred_label}.png")

    # --- Plot 2: Global word importance across all sentences ---
    print("\nGenerating global word importance...")

    # For positive class
    plt.figure(figsize=(10, 6))
    shap.plots.bar(
        shap_values[:, :, 2].mean(0),  # positive class
        show=False,
        max_display=15,
    )
    plt.title("Top Words Driving POSITIVE Sentiment", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_positive_words.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/shap_positive_words.png")

    # For negative class
    plt.figure(figsize=(10, 6))
    shap.plots.bar(
        shap_values[:, :, 0].mean(0),  # negative class
        show=False,
        max_display=15,
    )
    plt.title("Top Words Driving NEGATIVE Sentiment", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/shap_negative_words.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/shap_negative_words.png")

    print("\n" + "=" * 50)
    print("SHAP analysis complete.")
    print(f"Outputs saved to: {output_dir}/")
    print("=" * 50)


if __name__ == "__main__":
    run_shap_analysis()