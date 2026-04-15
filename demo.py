#!/usr/bin/env python3
"""
Adaptive-RAG: Complete Demo Script
===================================
This script runs the entire project pipeline end-to-end:
  1. Downloads datasets (SQuAD, NQ, TriviaQA, HotpotQA)
  2. Builds BM25 and Dense retrievers
  3. Trains Base and Enhanced classifiers
  4. Runs 6 experiments (No-Retrieval, Single-step, Multi-step, Adaptive variants)
  5. Generates result plots
  6. Generates the PDF report

Usage:
    python demo.py              # Run full pipeline
    python demo.py --quick      # Quick demo with fewer samples (faster)
    python demo.py --test-only  # Skip training, just run QA on a few example queries
"""
import sys, os, argparse, json, time

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def run_full_pipeline(n_samples=30):
    """Run the complete experiment pipeline."""
    print("=" * 60)
    print("  ADAPTIVE-RAG: Full Experiment Pipeline")
    print("=" * 60)
    start = time.time()

    # Step 1: Run experiments
    print("\n>>> Running experiments (this downloads models + datasets on first run)...\n")
    os.system(f'{sys.executable} src/run_experiments.py')

    # Step 2: Generate plots
    print("\n>>> Generating plots...\n")
    os.system(f'{sys.executable} src/generate_plots.py')

    # Step 3: Generate report PDF
    print("\n>>> Generating report PDF...\n")
    os.system(f'{sys.executable} generate_report_pdf.py')

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE in {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}")
    print(f"\nOutputs:")
    print(f"  Results:  results/experiment_results.json")
    print(f"  Plots:    plots/*.png (5 figures)")
    print(f"  Report:   report.pdf")


def run_interactive_demo():
    """Interactive demo: ask questions and see which strategy Adaptive-RAG picks."""
    print("=" * 60)
    print("  ADAPTIVE-RAG: Interactive Demo")
    print("=" * 60)

    from data_loader import load_all_datasets, SIMPLE_QUESTIONS
    from retrieval import BM25Retriever, DenseRetriever, build_corpus_from_data
    from classifier import BaseClassifier, EnhancedClassifier
    from strategies import no_retrieval, single_step, multi_step, adaptive_rag, get_qa_pipeline

    # Load minimal data for corpus
    print("\nLoading datasets for retrieval corpus...")
    data = load_all_datasets(n_per_dataset=20)
    all_data = SIMPLE_QUESTIONS + data

    # Build retrievers
    print("Building retrievers...")
    corpus = build_corpus_from_data(all_data)
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)

    # Train classifier
    print("Training classifier...")
    questions = [s["question"] for s in all_data]
    labels = [s["complexity"] for s in all_data]
    clf = EnhancedClassifier()
    clf.train(questions, labels)

    # Load QA model
    print("Loading QA model (first time downloads ~300MB)...")
    get_qa_pipeline()

    strategy_names = {"A": "No Retrieval", "B": "Single-step", "C": "Multi-step"}

    print("\n" + "=" * 60)
    print("Ready! Type a question (or 'quit' to exit)")
    print("=" * 60)

    # Run some example queries first
    examples = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
    ]
    print("\n--- Example Queries ---\n")
    for q in examples:
        label, conf = clf.predict(q)
        result = adaptive_rag(q, clf, dense, use_fallback=True)
        print(f"Q: {q}")
        print(f"  Classifier: Level {label} ({strategy_names[label]}) | Confidence: {conf:.2f}")
        print(f"  Answer: {result['answer']}")
        print(f"  Steps: {result['steps']} | Time: {result['time']:.3f}s")
        print()

    # Interactive loop
    print("--- Your Turn (type 'quit' to exit) ---\n")
    while True:
        try:
            q = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        label, conf = clf.predict(q)
        result = adaptive_rag(q, clf, dense, use_fallback=True)
        print(f"  Classifier: Level {label} ({strategy_names[label]}) | Confidence: {conf:.2f}")
        print(f"  Answer: {result['answer']}")
        print(f"  Steps: {result['steps']} | Time: {result['time']:.3f}s\n")

    print("\nDemo complete!")


def main():
    parser = argparse.ArgumentParser(description="Adaptive-RAG Project Demo")
    parser.add_argument("--quick", action="store_true", help="Quick demo with example queries only")
    parser.add_argument("--test-only", action="store_true", help="Interactive Q&A demo (skip training/experiments)")
    args = parser.parse_args()

    if args.quick or args.test_only:
        run_interactive_demo()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
