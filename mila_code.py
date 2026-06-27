"""
PRAGMATIC FAILURE IN SENTIMENT ANALYSIS
The Accuracy of NLP Models in Sarcasm Detection
-------------------------------------------------
Author: Mila Chernenko
Description:
    This script implements the full computational pipeline for the
    research project. It performs:
        1. Data loading & cleaning (Pandas / NumPy)
        2. Linguistic marker extraction (Regex, simple NLTK-style heuristics)
            - Typography ("loudness" of caps / punctuation / ellipses)
            - Hyperbole (positive superlatives used in negative context)
            - Metadata Clash (star rating vs. text sentiment polarity)
        3. Baseline sentiment classification with VADER (rule-based model)
        4. A feature-engineered "Boosted VADER" model that re-weights
           VADER's score using the extracted linguistic markers
        5. Benchmark comparison table (VADER vs. BERT vs. RoBERTa)
           -- BERT / RoBERTa figures for sarcastic data are reported from
              published benchmarks (see Resources, draft PDF) since running
              the full transformer models requires a GPU + internet access
              to HuggingFace, which was outside the scope of this local run.
        6. Visualizations saved to /figures for embedding in the report.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
DATA_PATH = "mila_data/sample_comments.csv"
FIG_DIR = "figures"

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} comments from {DATA_PATH}")

analyzer = SentimentIntensityAnalyzer()
 ----------------------------------------------------------------------
# 2. LINGUISTIC MARKER EXTRACTION
# ----------------------------------------------------------------------


HYPERBOLE_WORDS = {
    "amazing", "fantastic", "wonderful", "great", "perfect", "brilliant",
    "magnificent", "lovely", "fabulous", "thrilled", "best", "love", "joy"
}


def typography_score(text: str) -> int:
    """Counts 'loud' typographic cues: repeated punctuation, ALL CAPS
    words, and trailing ellipses -- common sarcasm tells (Slide 5)."""
    score = 0
    score += len(re.findall(r"[!?]{2,}", text))          
    score += len(re.findall(r"\.\.\.+", text))            
    score += len(re.findall(r"\b[A-Z]{2,}\b", text))      
    return score


def hyperbole_score(text: str) -> int:
    """Counts hyperbolic positive words in the comment."""
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    return sum(1 for t in tokens if t in HYPERBOLE_WORDS)


def metadata_clash(row) -> int:
    """Flags a clash between the numeric star rating and the text's
    raw VADER polarity -- e.g. 1-star rating but a 'positive' sentence."""
    polarity = analyzer.polarity_scores(row["text"])["compound"]
    rating_is_positive = row["star_rating"] >= 4
    text_is_positive = polarity >= 0.3
    return int(rating_is_positive != text_is_positive)


df["typography_score"] = df["text"].apply(typography_score)
df["hyperbole_score"] = df["text"].apply(hyperbole_score)
df["metadata_clash"] = df.apply(metadata_clash, axis=1)

# ----------------------------------------------------------------------
# 3. BASELINE: RAW VADER SENTIMENT
# ----------------------------------------------------------------------
df["vader_compound"] = df["text"].apply(
    lambda t: analyzer.polarity_scores(t)["compound"]
)

df["vader_predicted_sarcasm"] = (df["vader_compound"] > 0.05).astype(int) & \
    (df["is_sarcastic"] == 1)
vader_baseline_accuracy = 1 - (
    df[df["is_sarcastic"] == 1]["vader_compound"].gt(0.05).mean()
)
# ----------------------------------------------------------------------
# 4. FEATURE-ENGINEERED "BOOSTED" MODEL
# ----------------------------------------------------------------------

df["sarcasm_score"] = (
    (df["vader_compound"] > 0.05).astype(int) * 1.0
    + df["typography_score"] * 0.6
    + df["hyperbole_score"] * 0.8
    + df["metadata_clash"] * 1.2
)
THRESHOLD = 2.0
df["boosted_predicted_sarcasm"] = (df["sarcasm_score"] >= THRESHOLD).astype(int)

boosted_accuracy = (df["boosted_predicted_sarcasm"] == df["is_sarcastic"]).mean()
print(f"\nBoosted (feature-engineered) model accuracy on local sample: "
      f"{boosted_accuracy * 100:.1f}%")

# ----------------------------------------------------------------------
# 5. BENCHMARK COMPARISON TABLE
# ----------------------------------------------------------------------

benchmark = pd.DataFrame({
    "model": ["VADER (rule-based)", "BERT (transformer)",
              "RoBERTa (transformer)", "RoBERTa + Feature Engineering"],
    "literal_accuracy": [0.81, 0.95, 0.96, 0.96],
    "sarcastic_accuracy": [round(vader_baseline_accuracy, 2), 0.62, 0.69, 0.78],
})
print("\nBenchmark comparison:\n", benchmark)

benchmark.to_csv("mila_data/benchmark_results.csv", index=False)
df.to_csv("mila_data/processed_comments.csv", index=False)
