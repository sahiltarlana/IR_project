"""
Main experiment runner for Adaptive-RAG project.
Runs all experiments: base methods, base Adaptive-RAG, enhanced Adaptive-RAG.
"""
import sys, os, json, random, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data_loader import load_all_datasets, SIMPLE_QUESTIONS
from retrieval import BM25Retriever, DenseRetriever, build_corpus_from_data
from classifier import BaseClassifier, EnhancedClassifier
from strategies import no_retrieval, single_step, multi_step, adaptive_rag, get_qa_pipeline
from metrics import exact_match, f1_score, accuracy, evaluate_results

random.seed(42)
np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_strategy_on_data(data, strategy_fn, desc=""):
    """Run a single strategy on all data samples."""
    results = []
    for sample in tqdm(data, desc=desc):
        try:
            out = strategy_fn(sample["question"])
            results.append({
                "question": sample["question"],
                "gold": sample["answer"],
                "predicted": out["answer"],
                "strategy": out["strategy"],
                "steps": out["steps"],
                "time": out["time"],
                "dataset": sample["dataset"],
                "em": exact_match(out["answer"], sample["answer"]),
                "f1": f1_score(out["answer"], sample["answer"]),
                "acc": accuracy(out["answer"], sample["answer"]),
            })
        except Exception as e:
            print(f"  Error: {e}")
    return results


def run_adaptive_on_data(data, classifier, retriever, use_fallback=False, desc=""):
    """Run adaptive-RAG on all data samples."""
    results = []
    for sample in tqdm(data, desc=desc):
        try:
            out = adaptive_rag(sample["question"], classifier, retriever, use_fallback=use_fallback)
            results.append({
                "question": sample["question"],
                "gold": sample["answer"],
                "predicted": out["answer"],
                "strategy": out["strategy"],
                "steps": out["steps"],
                "time": out["time"],
                "dataset": sample["dataset"],
                "em": exact_match(out["answer"], sample["answer"]),
                "f1": f1_score(out["answer"], sample["answer"]),
                "acc": accuracy(out["answer"], sample["answer"]),
            })
        except Exception as e:
            print(f"  Error: {e}")
    return results


def main():
    print("=" * 60)
    print("ADAPTIVE-RAG EXPERIMENT RUNNER")
    print("=" * 60)

    # --- Step 1: Load data ---
    print("\n[1/6] Loading datasets...")
    dataset_samples = load_all_datasets(n_per_dataset=30)
    all_data = SIMPLE_QUESTIONS + dataset_samples

    # --- Step 2: Build retrieval corpus & retrievers ---
    print("\n[2/6] Building retrievers...")
    corpus = build_corpus_from_data(all_data)
    print(f"  Corpus size: {len(corpus)} chunks")
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)

    # --- Step 3: Train classifiers ---
    print("\n[3/6] Training classifiers...")
    train_q = [s["question"] for s in all_data]
    train_l = [s["complexity"] for s in all_data]
    # Stratified split to ensure all classes in both sets
    train_questions, test_questions, train_labels, test_labels = train_test_split(
        train_q, train_l, test_size=0.25, random_state=42, stratify=train_l
    )

    base_clf = BaseClassifier()
    base_clf.train(train_questions, train_labels)
    base_eval = base_clf.evaluate(test_questions, test_labels)
    print(f"  Base classifier accuracy: {base_eval['accuracy']:.3f}")

    enh_clf = EnhancedClassifier()
    enh_clf.train(train_questions, train_labels)
    enh_eval = enh_clf.evaluate(test_questions, test_labels)
    print(f"  Enhanced classifier accuracy: {enh_eval['accuracy']:.3f}")

    # --- Step 4: Preload QA model ---
    print("\n[4/6] Loading QA model...")
    get_qa_pipeline()

    # --- Step 5: Run experiments ---
    print("\n[5/6] Running experiments...")
    test_data = [s for s in all_data if s["question"] in set(test_questions)]

    # Experiment 1: No Retrieval (Strategy A)
    print("\n--- Exp 1: No Retrieval ---")
    res_no_ret = run_strategy_on_data(test_data, no_retrieval, "No Retrieval")

    # Experiment 2: Single-step with BM25 (Strategy B)
    print("\n--- Exp 2: Single-step BM25 ---")
    res_single_bm25 = run_strategy_on_data(
        test_data, lambda q: single_step(q, bm25), "Single-step BM25")

    # Experiment 3: Multi-step with BM25 (Strategy C)
    print("\n--- Exp 3: Multi-step BM25 ---")
    res_multi_bm25 = run_strategy_on_data(
        test_data, lambda q: multi_step(q, bm25), "Multi-step BM25")

    # Experiment 4: Adaptive-RAG (Base classifier + BM25)
    print("\n--- Exp 4: Adaptive-RAG (Base) ---")
    res_adaptive_base = run_adaptive_on_data(
        test_data, base_clf, bm25, desc="Adaptive-RAG Base")

    # Experiment 5: Adaptive-RAG (Enhanced classifier + Dense retrieval)
    print("\n--- Exp 5: Adaptive-RAG (Enhanced) ---")
    res_adaptive_enh = run_adaptive_on_data(
        test_data, enh_clf, dense, desc="Adaptive-RAG Enhanced")

    # Experiment 6: Adaptive-RAG (Enhanced + Confidence Fallback)
    print("\n--- Exp 6: Adaptive-RAG (Enhanced + Fallback) ---")
    res_adaptive_fb = run_adaptive_on_data(
        test_data, enh_clf, dense, use_fallback=True, desc="Adaptive-RAG Fallback")

    # --- Step 6: Aggregate & save ---
    print("\n[6/6] Computing metrics and saving results...")
    all_results = {
        "No Retrieval": evaluate_results(res_no_ret),
        "Single-step (BM25)": evaluate_results(res_single_bm25),
        "Multi-step (BM25)": evaluate_results(res_multi_bm25),
        "Adaptive-RAG (Base)": evaluate_results(res_adaptive_base),
        "Adaptive-RAG (Enhanced)": evaluate_results(res_adaptive_enh),
        "Adaptive-RAG (Enhanced+Fallback)": evaluate_results(res_adaptive_fb),
    }

    # Classifier results
    classifier_results = {
        "base_classifier": base_eval,
        "enhanced_classifier": enh_eval,
    }

    # Save
    with open(os.path.join(RESULTS_DIR, "experiment_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "classifier_results.json"), "w") as f:
        json.dump(classifier_results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "detailed_no_retrieval.json"), "w") as f:
        json.dump(res_no_ret, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "detailed_adaptive_enhanced.json"), "w") as f:
        json.dump(res_adaptive_enh, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Method':<35} {'EM':>6} {'F1':>6} {'Acc':>6} {'Steps':>6} {'Time':>8}")
    print("-" * 80)
    for method, m in all_results.items():
        print(f"{method:<35} {m['EM']:>6.2f} {m['F1']:>6.2f} {m['Acc']:>6.2f} {m['Avg_Steps']:>6.2f} {m['Avg_Time']:>8.3f}")
    print("=" * 80)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
