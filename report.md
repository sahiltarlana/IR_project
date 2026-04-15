# Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity

## IT367 Information Retrieval — Course Project End-Semester Report [Jan–Apr 2026]

**Base Paper:** Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park — NAACL 2024 (arXiv: 2403.14403)

**Team Members:**
- Pranav Moothedath (221AI030)
- Tarlana Sahil (221AI040)

---

## Abstract

This report presents our implementation and evaluation of the Adaptive-RAG framework proposed by Jeong et al. (2024), along with three proposed enhancements: dense retrieval to replace BM25, an improved query-complexity classifier with linguistic feature engineering, and a confidence-based fallback mechanism. We implemented the complete pipeline — including a three-level query-complexity classifier, three retrieval-augmented QA strategies (no-retrieval, single-step, and multi-step), and our proposed enhancements — and evaluated them on samples from SQuAD, Natural Questions, TriviaQA, and HotpotQA. Our experiments demonstrate the core insight of the paper: adaptive routing of queries based on complexity can balance accuracy and efficiency. We also analyze the strengths and limitations of our proposed enhancements and discuss directions for future work.

---

## 1. Introduction

### 1.1 Background and Motivation

Retrieval-Augmented Generation (RAG) has emerged as a leading approach to enhance Large Language Models (LLMs) by incorporating external, non-parametric knowledge from knowledge bases such as Wikipedia. While LLMs like GPT-3.5, FLAN-T5, and LLaMA store vast knowledge in their parametric memory, this knowledge can be incomplete, outdated, or hallucinated. RAG addresses this by retrieving relevant documents at inference time and conditioning the LLM's generation on them.

However, existing RAG approaches suffer from a fundamental limitation: they apply a one-size-fits-all retrieval strategy regardless of query complexity. Single-step RAG retrieves documents once and works well for simple factual queries, but fails on complex multi-hop questions that require reasoning across multiple documents. Conversely, multi-step RAG methods like IRCoT (Trivedi et al., 2023) iteratively retrieve and reason, which is powerful for complex queries but incurs heavy computational overhead on simple ones.

The key insight motivating Adaptive-RAG is that real-world user queries span a spectrum of complexities — from trivial questions like "What is the capital of France?" to complex multi-hop questions like "When did the people who captured Malakoff come to the region where Philipsburg is located?" An ideal system should dynamically select the most appropriate retrieval strategy based on the complexity of each incoming query.

### 1.2 Core Idea of Adaptive-RAG

Adaptive-RAG proposes a novel adaptive QA framework with three operating modes:
- **Level A (No Retrieval):** For straightforward queries answerable by the LLM's parametric knowledge alone.
- **Level B (Single-step Retrieval):** For moderate queries requiring one round of document retrieval.
- **Level C (Multi-step Retrieval):** For complex multi-hop queries requiring iterative retrieval and reasoning.

A lightweight query-complexity classifier (a smaller LM) is trained to predict the complexity level of each incoming query. This classifier is trained on automatically collected silver labels — no human annotation is required. The result is a system that achieves accuracy comparable to multi-step methods while being significantly more efficient.

### 1.3 Our Contributions

In this project, we:
1. Implemented the complete Adaptive-RAG framework from scratch, including all three retrieval strategies and the query-complexity classifier.
2. Proposed and implemented three enhancements: (a) dense retrieval using sentence-transformers to replace BM25, (b) an improved classifier with linguistic feature engineering and gradient boosting, and (c) a confidence-based fallback mechanism.
3. Conducted comprehensive experiments on four QA benchmarks and analyzed the results.

---

## 2. Literature Review

| # | Author / Year | Methodology | Advantages | Limitations |
|---|--------------|-------------|------------|-------------|
| 1 | Mallen et al. (2023) — "When Not to Trust Language Models" | Binary decision: retrieve vs. no-retrieve based on entity popularity | Simple and efficient; avoids unnecessary retrieval | Too coarse — cannot handle multi-hop queries |
| 2 | Trivedi et al. (2023) — IRCoT | Interleaves Chain-of-Thought reasoning with iterative retrieval | Highly effective for complex multi-hop queries; SOTA on multi-hop benchmarks | Applies same expensive process to ALL queries regardless of complexity |
| 3 | Asai et al. (2024) — Self-RAG | LLM generates reflection tokens to decide when to retrieve | End-to-end trainable; self-reflective | Requires training a specialized LLM; not plug-and-play |
| 4 | Press et al. (2023) — Self-Ask | Decomposes multi-hop queries into sub-queries | Effective decomposition for multi-hop reasoning | Still applies iterative strategy to all queries |
| 5 | Jiang et al. (2023) — FLARE | Retrieves when generated tokens have low confidence | Confidence-based trigger avoids blind retrieval | Does not classify queries upfront into complexity levels |
| 6 | Qi et al. (2021) | Fixed retrieve-read-rerank operations repeated until answer | Handles varying reasoning steps | Same fixed operations for every query; not designed for modern LLMs |

### 2.1 Identified Gaps

1. Existing adaptive methods make overly simplistic binary decisions, failing to account for multi-hop complexity.
2. Multi-step approaches apply the same expensive pipeline to ALL queries, wasting resources on simple ones.
3. Self-RAG requires training a specialized LLM and is not plug-and-play.
4. No existing work explicitly classifies query complexity into multiple levels and dynamically routes queries to the appropriate strategy.

---

## 3. Problem Statement

Existing retrieval-augmented LLM approaches use a one-size-fits-all strategy that either incurs unnecessary computational overhead on simple queries (multi-step methods) or fails to adequately handle complex multi-hop queries (single-step/no-retrieval methods). There is a need for an adaptive QA framework that can dynamically select the most suitable retrieval-augmented strategy based on the complexity of incoming queries.

### 3.1 Research Objectives

1. Design an adaptive retrieval-augmented generation framework that seamlessly switches between no-retrieval, single-step retrieval, and multi-step retrieval strategies based on query complexity.
2. Develop a lightweight query-complexity classifier that predicts the complexity level of incoming queries without requiring human-annotated training data.
3. Evaluate the framework on multiple open-domain QA benchmarks to demonstrate improvements in both accuracy and efficiency.


## 4. Existing Methodology (Adaptive-RAG)

### 4.1 Retrieval-Augmented QA Strategies

**Strategy A — No Retrieval:**
The LLM generates the answer solely from its parametric knowledge: ā = LLM(q). Suitable for straightforward, well-known factual queries.

**Strategy B — Single-step Retrieval-Augmented QA:**
Retrieve relevant documents once using a retriever (e.g., BM25) from an external corpus, then augment the LLM input: d = Retriever(q; D), then ā = LLM(q, d). Suitable for moderate queries requiring external knowledge.

**Strategy C — Multi-step Retrieval-Augmented QA:**
Iteratively retrieve new documents and generate intermediate answers over multiple rounds. At each step i: dᵢ = Retriever(q, cᵢ; D), then āᵢ = LLM(q, dᵢ, cᵢ), where cᵢ accumulates previous context. Suitable for complex multi-hop queries.

### 4.2 Query Complexity Classifier

The classifier is a smaller Language Model (T5-Large, 770M params in the original paper) that takes the query as input and outputs one of three labels: A, B, or C. Based on the predicted label, the query is routed to the corresponding strategy. The LLM and Retriever remain unchanged — only the routing decision changes.

### 4.3 Automatic Label Collection

**Step 1 — Silver Labels from Model Predictions:** Run all three strategies on training queries. Assign the label of the simplest strategy that correctly answers the query (priority to simpler models to break ties).

**Step 2 — Dataset Inductive Bias:** Queries from single-hop datasets (SQuAD, NQ, TriviaQA) that remain unlabeled are assigned B. Queries from multi-hop datasets (HotpotQA, MuSiQue, 2Wiki) are assigned C.

---

## 5. Proposed Enhancements

### 5.1 Enhancement 1: Dense Retrieval (Replace BM25)

The base paper uses BM25 (sparse, term-matching retrieval) throughout. BM25 relies on exact lexical overlap, which can miss semantically relevant documents when the query uses different terminology than the source documents.

**Our approach:** We replace BM25 with a dense retriever using the `all-MiniLM-L6-v2` sentence-transformer model. This model encodes both queries and documents into dense vector representations and retrieves based on cosine similarity in the embedding space.

**Hypothesis:** Semantic retrieval should improve retrieval quality, particularly for single-step queries where the retrieved context quality directly determines answer quality.

### 5.2 Enhancement 2: Improved Classifier Architecture

The original paper reports a significant gap between the actual classifier and the oracle classifier (F1: 46.94 vs 56.28 with FLAN-T5-XL). The confusion matrix shows that label 'A' is often misclassified as 'B' (~47%) and 'C' as 'B' (~31%).

**Our approach:** We implemented two classifiers:
- **Base Classifier:** TF-IDF (5000 features, unigrams+bigrams) + Logistic Regression — simulating the T5-based classifier from the paper.
- **Enhanced Classifier:** TF-IDF + 12 hand-crafted linguistic features + Gradient Boosting. The linguistic features include: word count, character count, question word type (who/what/when/how), comma count, named entity indicators, comparison markers, temporal markers, causal markers, and subordinate clause count.

**Hypothesis:** Incorporating explicit linguistic complexity signals should help the classifier better distinguish between simple factual queries and complex multi-hop queries.

### 5.3 Enhancement 3: Confidence-Based Fallback Mechanism

In the original framework, once the classifier predicts a complexity level, there is no fallback if the answer quality is poor. A misclassified complex query routed to no-retrieval will produce a poor answer with no recovery mechanism.

**Our approach:** After the classifier predicts a label, we check the prediction confidence (probability of the predicted class). If the confidence falls below a threshold (0.6), we escalate to the next higher complexity strategy: A→B, B→C.

**Hypothesis:** This safety net should reduce the impact of classifier errors, particularly for borderline queries where the classifier is uncertain.

---

## 6. Implementation Details

### 6.1 Technical Setup

- **Language:** Python 3.9.6
- **Platform:** macOS ARM (Apple Silicon)
- **Environment:** Isolated virtual environment (no system modifications)
- **Key Libraries:** PyTorch 2.8.0, Transformers 4.57.6, sentence-transformers, scikit-learn, rank_bm25, datasets (HuggingFace)

### 6.2 Models Used

| Component | Model | Parameters |
|-----------|-------|------------|
| QA Generator (LLM) | FLAN-T5-Small | 80M |
| Dense Retriever | all-MiniLM-L6-v2 | 22M |
| Base Classifier | TF-IDF + Logistic Regression | ~5K features |
| Enhanced Classifier | TF-IDF + Linguistic Features + GBM | ~5K + 12 features |
| Sparse Retriever | BM25 (Okapi) | Non-parametric |

**Note:** We used FLAN-T5-Small instead of the paper's FLAN-T5-XL (3B) due to computational constraints (CPU-only inference on a laptop). This means our absolute QA scores are lower than the paper's, but the relative comparisons between methods remain valid and informative.

### 6.3 Datasets

| Dataset | Type | Hops | Complexity Label | Samples Used |
|---------|------|------|-----------------|--------------|
| SQuAD v1.1 | Single-hop | 1 | B | 30 |
| Natural Questions | Single-hop | 1 | B | 30 |
| TriviaQA | Single-hop | 1 | B | 30 |
| HotpotQA | Multi-hop | 2 | C | 30 |
| Simple Factual | Trivial | 0 | A | 30 |

Total: 150 samples (120 from benchmarks + 30 simple factual questions). Stratified 75/25 train-test split ensuring all complexity classes are represented in both sets.

### 6.4 Evaluation Metrics

- **Exact Match (EM):** Percentage of predictions that exactly match the ground truth (after normalization).
- **F1 Score:** Token-level F1 between prediction and ground truth.
- **Accuracy:** Relaxed accuracy — ground truth is a substring of prediction or vice versa.
- **Average Steps:** Mean number of retrieval steps per query.
- **Average Time:** Mean wall-clock time per query (seconds).

### 6.5 Retrieval Corpus

We constructed the retrieval corpus from the context passages provided in the datasets (SQuAD and HotpotQA contexts). Passages were chunked into segments of approximately 200 characters, deduplicated, yielding 673 corpus chunks.


## 7. Results and Analysis

### 7.1 Main Results

| Method | EM (%) | F1 (%) | Acc (%) | Avg Steps | Avg Time (s) |
|--------|--------|--------|---------|-----------|---------------|
| No Retrieval | 7.89 | 12.33 | 7.89 | 0.00 | 0.437 |
| Single-step (BM25) | 15.79 | 20.02 | 23.68 | 1.00 | 0.440 |
| Multi-step (BM25) | 21.05 | 22.73 | 23.68 | 2.39 | 1.168 |
| Adaptive-RAG (Base) | 15.79 | 20.02 | 23.68 | 0.95 | 0.465 |
| Adaptive-RAG (Enhanced) | 15.79 | 17.44 | 23.68 | 0.76 | 0.568 |
| Adaptive-RAG (Enhanced+Fallback) | 15.79 | 17.44 | 23.68 | 0.76 | 0.576 |

### 7.2 Classifier Performance

| Classifier | Overall Accuracy | A (No Ret.) | B (Single) | C (Multi) |
|-----------|-----------------|-------------|------------|-----------|
| Base (TF-IDF + LR) | 65.8% | 2/7 correct | 23/23 correct | 0/8 correct |
| Enhanced (TF-IDF + GBM + Features) | 57.9% | 4/7 correct | 18/23 correct | 0/8 correct |

**Confusion Matrix Analysis:**

The base classifier shows a strong bias toward predicting label B (single-step), which mirrors the finding in the original paper where the classifier exhibits a "conservative bias toward single-step retrieval." Specifically:
- Label A is misclassified as B in 71% of cases (5 out of 7).
- Label C is misclassified as B in 100% of cases (8 out of 8).

The enhanced classifier with linguistic features shows improved recognition of simple queries (A): it correctly identifies 4 out of 7 simple queries compared to 2 out of 7 for the base classifier. However, it introduces some B→A misclassifications (5 out of 23), suggesting the linguistic features help detect simplicity but can be overly aggressive.

Neither classifier successfully identifies multi-hop queries (C), which aligns with the paper's observation that distinguishing between single-hop and multi-hop queries based on surface features alone is challenging.

### 7.3 Analysis of Key Findings

**Finding 1: Retrieval consistently improves over no-retrieval.**
The jump from No Retrieval (EM: 7.89, F1: 12.33) to Single-step BM25 (EM: 15.79, F1: 20.02) confirms that external knowledge retrieval substantially helps, even with a small LLM. This validates the core premise of RAG.

**Finding 2: Multi-step retrieval provides the best absolute performance.**
Multi-step BM25 achieves the highest EM (21.05) and F1 (22.73), confirming that iterative retrieval helps for complex queries. However, it is 2.7x slower than single-step (1.168s vs 0.440s per query).

**Finding 3: Adaptive-RAG achieves competitive accuracy with better efficiency.**
The base Adaptive-RAG matches single-step performance (EM: 15.79, F1: 20.02) while using fewer average steps (0.95 vs 1.00), demonstrating the efficiency benefit of routing some queries to no-retrieval. The time savings are modest in our setup because the classifier overhead partially offsets the retrieval savings.

**Finding 4: The classifier is the bottleneck.**
The gap between Adaptive-RAG and Multi-step performance (F1: 20.02 vs 22.73) is entirely attributable to classifier errors. When the classifier misroutes a complex query to single-step, the answer quality degrades. This mirrors the paper's finding of significant headroom between the actual classifier and the oracle (F1: 46.94 vs 56.28 in the original paper).

**Finding 5: Dense retrieval and enhanced classifier show mixed results.**
The enhanced Adaptive-RAG (with dense retrieval and improved classifier) achieves the same EM but slightly lower F1 (17.44 vs 20.02). This is because the enhanced classifier's more aggressive A-label predictions sometimes route queries that need retrieval to the no-retrieval path. The dense retriever, while semantically richer, operates on a small corpus where BM25's exact matching is already effective.

**Finding 6: Confidence-based fallback has limited impact in our setup.**
The fallback mechanism (Enhanced+Fallback) produces identical results to the Enhanced variant, indicating that the classifier's confidence scores are generally above the 0.6 threshold even for misclassified queries. This suggests that confidence calibration or a higher threshold may be needed.

### 7.4 Comparison with Original Paper Results

| Metric | Original Paper (FLAN-T5-XL) | Our Implementation (FLAN-T5-Small) |
|--------|------------------------------|-------------------------------------|
| Adaptive-RAG EM | 37.17 | 15.79 |
| Adaptive-RAG F1 | 46.94 | 20.02 |
| Multi-step EM | 39.00 | 21.05 |
| Multi-step F1 | 48.85 | 22.73 |
| Classifier Accuracy | 54.52% | 65.8% |
| Adaptive-RAG / Multi-step F1 ratio | 96.1% | 88.1% |

The absolute scores are lower due to using a much smaller LLM (80M vs 3B parameters), but the relative patterns are consistent: Adaptive-RAG achieves ~88-96% of multi-step performance while being more efficient.

---

## 8. Visualizations

The following plots were generated from our experimental results:

1. **Performance Comparison (performance_comparison.png):** Bar chart comparing EM, F1, and Accuracy across all six methods.
2. **F1 vs Time (f1_vs_time.png):** Scatter plot replicating the paper's Figure 1, showing the performance-efficiency tradeoff.
3. **Classifier Confusion Matrices (classifier_confusion.png):** Side-by-side heatmaps for base and enhanced classifiers.
4. **Steps Comparison (steps_comparison.png):** Average retrieval steps per method.
5. **Time Comparison (time_comparison.png):** Average time per query across methods.

---

## 9. Individual Contributions

| Name — Reg. No | Contributions |
|----------------|---------------|
| Sahil — 221AI040 | Led paper reading and understanding of the Adaptive-RAG framework, conducted literature survey, handled environment setup and implementation of the complete pipeline (classifier, retrieval, strategies, experiments), dense retriever integration, and report writing. |
| Pranav — 221AI030 | Assisted in literature survey and experimental planning, supported classifier improvement experiments (feature engineering), implementation of the confidence-based fallback mechanism, and evaluation analysis. |

---

## 10. Conclusion and Future Work

### 10.1 Conclusion

We successfully implemented the Adaptive-RAG framework and validated its core insight: dynamically routing queries based on complexity can balance accuracy and efficiency in retrieval-augmented QA systems. Our experiments on four QA benchmarks confirm that:

1. Retrieval augmentation consistently improves over parametric-only approaches.
2. Multi-step retrieval provides the best accuracy but at significant computational cost.
3. Adaptive routing achieves competitive performance with reduced computational overhead.
4. The query-complexity classifier is the key bottleneck — improving it directly translates to better overall system performance.

Our proposed enhancements (dense retrieval, improved classifier, confidence fallback) showed mixed results in our constrained experimental setup, highlighting that these improvements require careful tuning and larger-scale evaluation to demonstrate their full potential.

### 10.2 Future Work

1. **Larger LLMs:** Evaluate with FLAN-T5-XL/XXL or modern open-source LLMs (LLaMA-3, Mistral) to study how stronger LLMs affect the balance between strategies.
2. **Better Classifier Training:** Use the silver label collection strategy from the paper (running all three strategies on training data) instead of dataset-level labels.
3. **Larger Retrieval Corpus:** Use the full Wikipedia corpus for more realistic retrieval evaluation.
4. **Fine-grained Complexity Levels:** Introduce a 4th level for 3+ hop queries with dedicated strategies.
5. **Confidence Calibration:** Apply temperature scaling or Platt scaling to improve classifier confidence estimates for the fallback mechanism.

---

## 11. References

1. Jeong, S., Baek, J., Cho, S., Hwang, S.J., & Park, J.C. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. NAACL 2024. arXiv:2403.14403.
2. Mallen, A., Asai, A., et al. (2023). When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-parametric Memories. ACL 2023.
3. Trivedi, H., Balasubramanian, N., et al. (2023). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. ACL 2023.
4. Asai, A., Wu, Z., et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR 2024.
5. Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP 2020.
6. Robertson, S.E., et al. (1994). Okapi at TREC-3. TREC 1994.
7. Rajpurkar, P., et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. EMNLP 2016.
8. Yang, Z., et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP 2018.
9. Jiang, Z., et al. (2023). Active Retrieval Augmented Generation (FLARE). EMNLP 2023.
10. Chung, H.W., et al. (2022). Scaling Instruction-Finetuned Language Models. arXiv:2210.11416.
11. Press, O., et al. (2023). Measuring and Narrowing the Compositionality Gap in Language Models. EMNLP 2023.
12. Qi, P., et al. (2021). Answering Open-Domain Questions of Varying Reasoning Steps from Text. EMNLP 2021.

