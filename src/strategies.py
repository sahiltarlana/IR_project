"""
QA Strategies for Adaptive-RAG: No-retrieval, Single-step, Multi-step.
Uses a small T5 model for answer generation (feasible on CPU/MPS).
"""
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

_qa_pipeline = None

def get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        print("Loading QA model (FLAN-T5-Small)...")
        _qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-small",
                                max_new_tokens=64, device=-1)
    return _qa_pipeline


def no_retrieval(question):
    """Strategy A: LLM answers from parametric knowledge only."""
    t0 = time.time()
    pipe = get_qa_pipeline()
    prompt = f"Answer the following question concisely:\n{question}"
    answer = pipe(prompt)[0]["generated_text"].strip()
    return {"answer": answer, "strategy": "A", "steps": 0, "time": time.time() - t0, "docs_used": []}


def single_step(question, retriever, top_k=3):
    """Strategy B: Retrieve once, then answer with context."""
    t0 = time.time()
    pipe = get_qa_pipeline()
    docs, scores = retriever.retrieve(question, top_k=top_k)
    context = " ".join(docs[:top_k])[:1500]
    prompt = f"Given the context: {context}\n\nAnswer the question: {question}"
    answer = pipe(prompt)[0]["generated_text"].strip()
    return {"answer": answer, "strategy": "B", "steps": 1, "time": time.time() - t0, "docs_used": docs}


def multi_step(question, retriever, max_steps=3, top_k=2):
    """Strategy C: Iterative retrieval + reasoning over multiple steps."""
    t0 = time.time()
    pipe = get_qa_pipeline()
    accumulated_context = ""
    intermediate_answers = []

    for step in range(max_steps):
        if step == 0:
            search_query = question
        else:
            search_query = f"{question} {intermediate_answers[-1]}"
        docs, scores = retriever.retrieve(search_query, top_k=top_k)
        accumulated_context += " " + " ".join(docs)
        accumulated_context = accumulated_context[:2000]

        prompt = f"Given context: {accumulated_context}\n\nAnswer: {question}"
        ans = pipe(prompt)[0]["generated_text"].strip()
        intermediate_answers.append(ans)

        # Early stop if answer is confident (repeated)
        if len(intermediate_answers) >= 2 and intermediate_answers[-1] == intermediate_answers[-2]:
            break

    return {"answer": intermediate_answers[-1], "strategy": "C", "steps": step + 1,
            "time": time.time() - t0, "docs_used": docs}


def adaptive_rag(question, classifier, retriever, use_fallback=False, confidence_threshold=0.6):
    """Core Adaptive-RAG: route query based on classifier prediction."""
    label, confidence = classifier.predict(question)

    if use_fallback and confidence < confidence_threshold:
        # Confidence-based fallback: escalate to next level
        label = {"A": "B", "B": "C", "C": "C"}[label]

    if label == "A":
        return no_retrieval(question)
    elif label == "B":
        return single_step(question, retriever)
    else:
        return multi_step(question, retriever)
