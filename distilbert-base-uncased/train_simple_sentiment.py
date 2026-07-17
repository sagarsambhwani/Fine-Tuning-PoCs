import argparse
import json
import os
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def build_samples():
    positive = [
        "I absolutely loved this movie and would watch it again.",
        "This was a fantastic experience with great pacing and wonderful acting.",
        "Amazing production quality and an uplifting story from start to finish.",
        "The soundtrack was beautiful and the characters felt real.",
        "A perfect blend of humor, emotion, and suspense.",
        "I enjoyed every minute and recommend it to everyone.",
        "Wonderful film with a touching ending and strong performances.",
        "This was a delightful and memorable experience.",
        "Excellent direction, brilliant performances, and a very satisfying ending.",
        "The movie was inspiring, charming, and genuinely fun to watch.",
        "I thought it was clever, moving, and beautifully made.",
        "A warm and uplifting story with great character development.",
        "I loved the humor and the emotional depth throughout the film.",
        "This was a joyful and rewarding experience from start to finish.",
        "The film was polished, engaging, and memorable.",
        "I would recommend it to anyone who enjoys thoughtful storytelling.",
    ]
    negative = [
        "This movie was dull, slow, and badly written.",
        "I hated the pacing and the ending felt rushed.",
        "The acting was poor and the plot made no sense.",
        "It was a frustrating experience from beginning to end.",
        "The film felt cheap and uninteresting.",
        "I regret watching it and would not recommend it.",
        "A disappointing and tedious watch with weak characters.",
        "The story was confusing and the dialogue was awkward.",
        "This was a terrible film with flat performances and a weak script.",
        "The movie felt boring, repetitive, and badly paced.",
        "I found it annoying, predictable, and deeply disappointing.",
        "An unpleasant experience with little emotional impact.",
        "The direction was messy and the story never came together.",
        "It was a painful watch and I was relieved when it ended.",
        "The film was lifeless, clichéd, and not worth the time.",
        "I disliked almost every element of this production.",
    ]
    texts = positive + negative
    labels = [1] * len(positive) + [0] * len(negative)
    return texts, labels


def train_model(output_dir: str):
    texts, labels = build_samples()
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_test_vec)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1": round(float(f1_score(y_test, predictions)), 4),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "model.pkl", "wb") as fh:
        pickle.dump((vectorizer, model), fh)

    with open(output_path / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return metrics


def predict(text: str, model_path: str = "models/simple_sentiment/model.pkl"):
    with open(model_path, "rb") as fh:
        vectorizer, model = pickle.load(fh)
    vectorized = vectorizer.transform([text])
    probability = float(model.predict_proba(vectorized)[0][1])
    label = int(model.predict(vectorized)[0])
    label_name = "positive" if label == 1 else "negative"
    return label_name, probability


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a simple sentiment classifier")
    parser.add_argument("--output-dir", default="models/simple_sentiment")
    args = parser.parse_args()
    metrics = train_model(args.output_dir)
    print(json.dumps(metrics, indent=2))
