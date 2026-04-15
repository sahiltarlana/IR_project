"""
Query Complexity Classifier for Adaptive-RAG.
Base: TF-IDF + Logistic Regression (simulating T5-based classifier).
Enhanced: TF-IDF + linguistic features + GradientBoosting.
"""
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def extract_features(question):
    """Extract linguistic features for enhanced classifier."""
    words = question.split()
    return [
        len(words),
        len(question),
        int(any(w.lower() in ["who", "whom"] for w in words)),
        int(any(w.lower() in ["what", "which"] for w in words)),
        int(any(w.lower() in ["when", "where"] for w in words)),
        int(any(w.lower() == "how" for w in words)),
        question.count(","),
        len(re.findall(r'[A-Z][a-z]+', question)),
        int(any(w in question.lower() for w in ["both", "between", "compare"])),
        int(any(w in question.lower() for w in ["before", "after", "during"])),
        int(any(w in question.lower() for w in ["why", "because", "cause"])),
        len(re.findall(r',\s*(who|which|that|where|when)', question.lower())),
    ]

LABEL_MAP = {"A": 0, "B": 1, "C": 2}
INV_MAP = {0: "A", 1: "B", 2: "C"}


class BaseClassifier:
    """TF-IDF + Logistic Regression (simulates T5-based from paper)."""
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def train(self, questions, labels):
        X = self.vec.fit_transform(questions)
        self.model.fit(X, [LABEL_MAP[l] for l in labels])

    def predict(self, question):
        X = self.vec.transform([question])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        return INV_MAP[pred], float(max(proba))

    def evaluate(self, questions, labels):
        preds = [self.predict(q)[0] for q in questions]
        return {
            "accuracy": accuracy_score(labels, preds),
            "report": classification_report(labels, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(labels, preds, labels=["A","B","C"]).tolist(),
        }


class EnhancedClassifier:
    """TF-IDF + linguistic features + GradientBoosting."""
    def __init__(self):
        self.vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

    def _feats(self, questions, fit=False):
        tfidf = (self.vec.fit_transform(questions) if fit else self.vec.transform(questions)).toarray()
        ling = np.array([extract_features(q) for q in questions])
        return np.hstack([tfidf, ling])

    def train(self, questions, labels):
        X = self._feats(questions, fit=True)
        self.model.fit(X, [LABEL_MAP[l] for l in labels])

    def predict(self, question):
        X = self._feats([question])
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        return INV_MAP[pred], float(max(proba))

    def evaluate(self, questions, labels):
        preds = [self.predict(q)[0] for q in questions]
        return {
            "accuracy": accuracy_score(labels, preds),
            "report": classification_report(labels, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(labels, preds, labels=["A","B","C"]).tolist(),
        }
