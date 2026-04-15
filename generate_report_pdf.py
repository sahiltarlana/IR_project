"""Generate project report PDF from markdown content."""
import os
from fpdf import FPDF

class ReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "IT367 Information Retrieval - Adaptive-RAG Project Report", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.ln(4)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(150)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.ln(2)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bold_text(self, text):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 10)
        self.multi_cell(0, 5, text)
        self.set_font("Helvetica", "", 10)

    def add_table(self, headers, rows, col_widths=None):
        avail = self.w - self.l_margin - self.r_margin
        if col_widths is None:
            col_widths = [avail / len(headers)] * len(headers)
        # Scale if too wide
        total = sum(col_widths)
        if total > avail:
            col_widths = [w * avail / total for w in col_widths]
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(230, 230, 230)
        self.set_x(self.l_margin)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 6, h, border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 9)
        for row in rows:
            self.set_x(self.l_margin)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 5.5, str(val)[:40], border=1, align="C")
            self.ln()
        self.ln(2)
        self.set_x(self.l_margin)

    def add_image_centered(self, path, w=160):
        if os.path.exists(path):
            x = (self.w - w) / 2
            self.image(path, x=x, w=w)
            self.ln(3)


pdf = ReportPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font("Helvetica", "B", 16)
pdf.multi_cell(0, 8, "Adaptive-RAG: Learning to Adapt Retrieval-Augmented\nLarge Language Models through Question Complexity", align="C")
pdf.ln(3)
pdf.set_font("Helvetica", "", 10)
pdf.cell(0, 5, "IT367 Information Retrieval - Course Project End-Semester Report [Jan-Apr 2026]", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.set_font("Helvetica", "I", 10)
pdf.cell(0, 5, "Base Paper: Jeong et al. - NAACL 2024 (arXiv: 2403.14403)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(3)
pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 5, "Team Members: Pranav Moothedath (221AI030) | Tarlana Sahil (221AI040)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(6)

# Abstract
pdf.section_title("Abstract")
pdf.body_text("This report presents our implementation and evaluation of the Adaptive-RAG framework proposed by Jeong et al. (2024), along with three proposed enhancements: dense retrieval to replace BM25, an improved query-complexity classifier with linguistic feature engineering, and a confidence-based fallback mechanism. We implemented the complete pipeline including a three-level query-complexity classifier, three retrieval-augmented QA strategies (no-retrieval, single-step, and multi-step), and our proposed enhancements. We evaluated them on samples from SQuAD, Natural Questions, TriviaQA, and HotpotQA. Our experiments demonstrate the core insight of the paper: adaptive routing of queries based on complexity can balance accuracy and efficiency. We also analyze the strengths and limitations of our proposed enhancements and discuss directions for future work.")

# 1. Introduction
pdf.section_title("1. Introduction")
pdf.sub_title("1.1 Background and Motivation")
pdf.body_text("Retrieval-Augmented Generation (RAG) has emerged as a leading approach to enhance Large Language Models (LLMs) by incorporating external, non-parametric knowledge from knowledge bases such as Wikipedia. While LLMs store vast knowledge in their parametric memory, this knowledge can be incomplete, outdated, or hallucinated. RAG addresses this by retrieving relevant documents at inference time.")
pdf.body_text("However, existing RAG approaches suffer from a fundamental limitation: they apply a one-size-fits-all retrieval strategy regardless of query complexity. Single-step RAG works well for simple factual queries but fails on complex multi-hop questions. Multi-step RAG methods like IRCoT iteratively retrieve and reason, which is powerful for complex queries but incurs heavy computational overhead on simple ones.")
pdf.body_text("The key insight motivating Adaptive-RAG is that real-world user queries span a spectrum of complexities. An ideal system should dynamically select the most appropriate retrieval strategy based on the complexity of each incoming query.")

pdf.sub_title("1.2 Core Idea of Adaptive-RAG")
pdf.body_text("Adaptive-RAG proposes a novel adaptive QA framework with three operating modes:\n- Level A (No Retrieval): For straightforward queries answerable by the LLM's parametric knowledge.\n- Level B (Single-step Retrieval): For moderate queries requiring one round of document retrieval.\n- Level C (Multi-step Retrieval): For complex multi-hop queries requiring iterative retrieval and reasoning.\nA lightweight query-complexity classifier is trained to predict the complexity level of each incoming query using automatically collected silver labels, requiring no human annotation.")

pdf.sub_title("1.3 Our Contributions")
pdf.body_text("1. Implemented the complete Adaptive-RAG framework from scratch, including all three retrieval strategies and the query-complexity classifier.\n2. Proposed and implemented three enhancements: (a) dense retrieval using sentence-transformers, (b) an improved classifier with linguistic feature engineering and gradient boosting, and (c) a confidence-based fallback mechanism.\n3. Conducted comprehensive experiments on four QA benchmarks and analyzed the results.")

# 2. Literature Review
pdf.section_title("2. Literature Review")
pdf.add_table(
    ["#", "Author/Year", "Method", "Limitation"],
    [
        ["1", "Mallen et al. (2023)", "Binary retrieve/no-retrieve", "Too coarse for multi-hop"],
        ["2", "Trivedi et al. (2023) IRCoT", "Iterative CoT + retrieval", "Same cost for all queries"],
        ["3", "Asai et al. (2024) Self-RAG", "Reflection tokens", "Needs specialized LLM"],
        ["4", "Press et al. (2023) Self-Ask", "Query decomposition", "No complexity assessment"],
        ["5", "Jiang et al. (2023) FLARE", "Confidence-based retrieval", "No upfront classification"],
    ],
    col_widths=[8, 42, 50, 50]
)
pdf.body_text("Key Gap: No existing work explicitly classifies query complexity into multiple levels and dynamically routes queries to the appropriate strategy, balancing both accuracy and efficiency.")

# 3. Problem Statement
pdf.section_title("3. Problem Statement and Research Objectives")
pdf.body_text("Existing retrieval-augmented LLM approaches use a one-size-fits-all strategy that either incurs unnecessary computational overhead on simple queries or fails to adequately handle complex multi-hop queries. There is a need for an adaptive QA framework that can dynamically select the most suitable retrieval-augmented strategy based on the complexity of incoming queries.")

pdf.sub_title("3.1 Research Objectives")
pdf.body_text("1. Design an adaptive retrieval-augmented generation framework that seamlessly switches between no-retrieval, single-step retrieval, and multi-step retrieval strategies based on query complexity.\n2. Develop a lightweight query-complexity classifier that predicts the complexity level of incoming queries without requiring human-annotated training data.\n3. Evaluate the framework on multiple open-domain QA benchmarks to demonstrate improvements in both accuracy and efficiency.\n4. Propose and evaluate enhancements including dense retrieval, improved classifier architecture, and confidence-based fallback.")

# 4. Existing Methodology
pdf.section_title("4. Existing Methodology (Adaptive-RAG)")
pdf.sub_title("4.1 Retrieval-Augmented QA Strategies")
pdf.bold_text("Strategy A - No Retrieval:")
pdf.body_text("The LLM generates the answer solely from parametric knowledge: a = LLM(q). Suitable for straightforward factual queries.")
pdf.bold_text("Strategy B - Single-step Retrieval:")
pdf.body_text("Retrieve documents once: d = Retriever(q; D), then a = LLM(q, d). Suitable for moderate queries requiring external knowledge.")
pdf.bold_text("Strategy C - Multi-step Retrieval:")
pdf.body_text("Iteratively retrieve and reason: at each step i, d_i = Retriever(q, c_i; D), then a_i = LLM(q, d_i, c_i). Context c_i accumulates previous reasoning. Suitable for complex multi-hop queries.")

pdf.sub_title("4.2 Query Complexity Classifier")
pdf.body_text("A smaller Language Model classifies queries into three complexity levels (A, B, C). The classifier is trained on automatically collected silver labels: run all three strategies on training queries and assign the label of the simplest strategy that correctly answers each query. Dataset inductive bias is used for remaining unlabeled queries.")

# 5. Proposed Enhancements
pdf.section_title("5. Proposed Enhancements")
pdf.sub_title("5.1 Dense Retrieval (Replace BM25)")
pdf.body_text("The base paper uses BM25 (sparse, term-matching retrieval). We replace it with a dense retriever using the all-MiniLM-L6-v2 sentence-transformer model, which encodes queries and documents into dense vectors and retrieves based on cosine similarity. Hypothesis: Semantic retrieval should improve retrieval quality, particularly for single-step queries.")

pdf.sub_title("5.2 Improved Classifier Architecture")
pdf.body_text("We implemented two classifiers:\n- Base: TF-IDF (5000 features, unigrams+bigrams) + Logistic Regression.\n- Enhanced: TF-IDF + 12 hand-crafted linguistic features (word count, question type, entity indicators, temporal/causal markers, subordinate clauses) + Gradient Boosting.\nHypothesis: Linguistic complexity signals should help distinguish simple from complex queries.")

pdf.sub_title("5.3 Confidence-Based Fallback Mechanism")
pdf.body_text("After the classifier predicts a label, we check prediction confidence. If confidence falls below 0.6, we escalate to the next higher complexity strategy (A->B, B->C). Hypothesis: This safety net should reduce the impact of classifier errors on borderline queries.")

# 6. Implementation Details
pdf.section_title("6. Implementation Details")
pdf.sub_title("6.1 Technical Setup")
pdf.body_text("Language: Python 3.9.6 | Platform: macOS ARM (Apple Silicon) | Environment: Isolated virtual environment\nKey Libraries: PyTorch 2.8.0, Transformers 4.57.6, sentence-transformers, scikit-learn, rank_bm25")

pdf.sub_title("6.2 Models Used")
pdf.add_table(
    ["Component", "Model", "Parameters"],
    [
        ["QA Generator", "FLAN-T5-Small", "80M"],
        ["Dense Retriever", "all-MiniLM-L6-v2", "22M"],
        ["Base Classifier", "TF-IDF + LogReg", "~5K features"],
        ["Enhanced Classifier", "TF-IDF + GBM", "~5K + 12 features"],
        ["Sparse Retriever", "BM25 (Okapi)", "Non-parametric"],
    ],
    col_widths=[50, 50, 50]
)
pdf.body_text("Note: We used FLAN-T5-Small (80M) instead of the paper's FLAN-T5-XL (3B) due to computational constraints. Absolute QA scores are lower, but relative comparisons between methods remain valid.")

pdf.sub_title("6.3 Datasets")
pdf.add_table(
    ["Dataset", "Type", "Complexity", "Samples"],
    [
        ["SQuAD v1.1", "Single-hop", "B", "30"],
        ["Natural Questions", "Single-hop", "B", "30"],
        ["TriviaQA", "Single-hop", "B", "30"],
        ["HotpotQA", "Multi-hop", "C", "30"],
        ["Simple Factual", "Trivial", "A", "30"],
    ],
    col_widths=[45, 35, 35, 35]
)
pdf.body_text("Total: 150 samples. Stratified 75/25 train-test split ensuring all complexity classes are represented.")

# 7. Results
pdf.section_title("7. Results and Analysis")
pdf.sub_title("7.1 Main Results")
pdf.add_table(
    ["Method", "EM(%)", "F1(%)", "Acc(%)", "Steps", "Time(s)"],
    [
        ["No Retrieval", "7.89", "12.33", "7.89", "0.00", "0.437"],
        ["Single-step (BM25)", "15.79", "20.02", "23.68", "1.00", "0.440"],
        ["Multi-step (BM25)", "21.05", "22.73", "23.68", "2.39", "1.168"],
        ["Adaptive-RAG (Base)", "15.79", "20.02", "23.68", "0.95", "0.465"],
        ["Adaptive-RAG (Enhanced)", "15.79", "17.44", "23.68", "0.76", "0.568"],
        ["Adaptive+Fallback", "15.79", "17.44", "23.68", "0.76", "0.576"],
    ],
    col_widths=[48, 18, 18, 18, 18, 18]
)

pdf.sub_title("7.2 Classifier Performance")
pdf.add_table(
    ["Classifier", "Accuracy", "A correct", "B correct", "C correct"],
    [
        ["Base (TF-IDF+LR)", "65.8%", "2/7", "23/23", "0/8"],
        ["Enhanced (TF-IDF+GBM)", "57.9%", "4/7", "18/23", "0/8"],
    ],
    col_widths=[40, 25, 25, 25, 25]
)

pdf.sub_title("7.3 Key Findings")
pdf.bold_text("Finding 1: Retrieval consistently improves over no-retrieval.")
pdf.body_text("The jump from No Retrieval (F1: 12.33) to Single-step BM25 (F1: 20.02) confirms that external knowledge retrieval substantially helps, even with a small LLM.")

pdf.bold_text("Finding 2: Multi-step retrieval provides the best absolute performance.")
pdf.body_text("Multi-step BM25 achieves the highest EM (21.05) and F1 (22.73), confirming iterative retrieval helps for complex queries. However, it is 2.7x slower than single-step.")

pdf.bold_text("Finding 3: Adaptive-RAG achieves competitive accuracy with better efficiency.")
pdf.body_text("Base Adaptive-RAG matches single-step performance while using fewer average steps (0.95 vs 1.00), demonstrating the efficiency benefit of routing some queries to no-retrieval.")

pdf.bold_text("Finding 4: The classifier is the bottleneck.")
pdf.body_text("The gap between Adaptive-RAG and Multi-step (F1: 20.02 vs 22.73) is attributable to classifier errors. This mirrors the paper's finding of significant headroom between actual and oracle classifiers.")

pdf.bold_text("Finding 5: Enhanced classifier shows improved A-detection but overall mixed results.")
pdf.body_text("The enhanced classifier correctly identifies 4/7 simple queries vs 2/7 for the base, but introduces B->A misclassifications. Neither classifier successfully identifies multi-hop queries (C), aligning with the paper's observation.")

pdf.sub_title("7.4 Comparison with Original Paper")
pdf.add_table(
    ["Metric", "Paper (FLAN-T5-XL)", "Ours (FLAN-T5-Small)"],
    [
        ["Adaptive-RAG F1", "46.94", "20.02"],
        ["Multi-step F1", "48.85", "22.73"],
        ["Classifier Acc", "54.52%", "65.8%"],
        ["Adaptive/Multi-step ratio", "96.1%", "88.1%"],
    ],
    col_widths=[50, 50, 50]
)
pdf.body_text("Absolute scores are lower due to the smaller LLM (80M vs 3B), but relative patterns are consistent.")

# 8. Visualizations
pdf.section_title("8. Visualizations")
for img, caption in [
    ("plots/performance_comparison.png", "Figure 1: Performance Comparison Across Methods"),
    ("plots/f1_vs_time.png", "Figure 2: Performance vs Efficiency (F1 vs Time per Query)"),
    ("plots/classifier_confusion.png", "Figure 3: Classifier Confusion Matrices (Base vs Enhanced)"),
    ("plots/steps_comparison.png", "Figure 4: Average Retrieval Steps per Method"),
    ("plots/time_comparison.png", "Figure 5: Average Time per Query"),
]:
    if os.path.exists(img):
        pdf.add_image_centered(img, w=150)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

# 9. Individual Contributions
pdf.section_title("9. Individual Contributions")
pdf.add_table(
    ["Name - Reg. No", "Contributions"],
    [
        ["Sahil - 221AI040", "Paper reading, literature survey, environment setup,\nfull pipeline implementation, dense retriever, report"],
        ["Pranav - 221AI030", "Literature survey, classifier improvement,\nfallback mechanism, evaluation analysis"],
    ],
    col_widths=[45, 105]
)

# 10. Conclusion
pdf.section_title("10. Conclusion and Future Work")
pdf.body_text("We successfully implemented the Adaptive-RAG framework and validated its core insight: dynamically routing queries based on complexity can balance accuracy and efficiency. Our experiments confirm that: (1) retrieval augmentation consistently improves over parametric-only approaches, (2) multi-step retrieval provides the best accuracy but at significant cost, (3) adaptive routing achieves competitive performance with reduced overhead, and (4) the classifier is the key bottleneck.")
pdf.body_text("Future work includes: evaluating with larger LLMs (FLAN-T5-XL, LLaMA-3), using silver label collection from the paper, larger retrieval corpora, fine-grained complexity levels, and confidence calibration for the fallback mechanism.")

# 11. References
pdf.section_title("11. Code Availability")
pdf.body_text("Original paper repository: https://github.com/starsuzi/Adaptive-RAG\nOur implementation code, experiment scripts, and results are available in the project submission folder (src/ directory). The implementation is self-contained and reproducible using the provided virtual environment setup.")

pdf.section_title("12. References")
refs = [
    "Jeong, S., et al. (2024). Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity. NAACL 2024.",
    "Mallen, A., et al. (2023). When Not to Trust Language Models. ACL 2023.",
    "Trivedi, H., et al. (2023). Interleaving Retrieval with Chain-of-Thought Reasoning. ACL 2023.",
    "Asai, A., et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique. ICLR 2024.",
    "Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain QA. EMNLP 2020.",
    "Robertson, S.E., et al. (1994). Okapi at TREC-3. TREC 1994.",
    "Rajpurkar, P., et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension. EMNLP 2016.",
    "Yang, Z., et al. (2018). HotpotQA: A Dataset for Diverse Multi-hop QA. EMNLP 2018.",
    "Jiang, Z., et al. (2023). Active Retrieval Augmented Generation (FLARE). EMNLP 2023.",
    "Chung, H.W., et al. (2022). Scaling Instruction-Finetuned Language Models. arXiv:2210.11416.",
    "Press, O., et al. (2023). Measuring and Narrowing the Compositionality Gap. EMNLP 2023.",
]
pdf.set_font("Helvetica", "", 9)
for i, ref in enumerate(refs, 1):
    pdf.multi_cell(0, 4.5, f"[{i}] {ref}")
    pdf.ln(0.5)

pdf.output("report.pdf")
print("PDF generated: report.pdf")
print(f"Pages: {pdf.page_no()}")
