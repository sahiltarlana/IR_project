"""
Generate all plots for the Adaptive-RAG project report.
"""
import json, os, sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "experiment_results.json")) as f:
    results = json.load(f)
with open(os.path.join(RESULTS_DIR, "classifier_results.json")) as f:
    clf_results = json.load(f)

sns.set_theme(style="whitegrid", font_scale=1.1)

# --- Plot 1: Performance comparison bar chart ---
methods = list(results.keys())
short_names = ["No Ret.", "Single\nBM25", "Multi\nBM25", "Adapt.\nBase", "Adapt.\nEnhanced", "Adapt.\n+Fallback"]
em = [results[m]["EM"] for m in methods]
f1 = [results[m]["F1"] for m in methods]
acc = [results[m]["Acc"] for m in methods]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(methods))
w = 0.25
ax.bar(x - w, em, w, label="EM", color="#2196F3")
ax.bar(x, f1, w, label="F1", color="#4CAF50")
ax.bar(x + w, acc, w, label="Accuracy", color="#FF9800")
ax.set_xticks(x)
ax.set_xticklabels(short_names)
ax.set_ylabel("Score (%)")
ax.set_title("Performance Comparison Across Methods")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "performance_comparison.png"), dpi=150)
plt.close()
print("Saved: performance_comparison.png")

# --- Plot 2: F1 vs Time (replicating paper's Figure 1) ---
fig, ax = plt.subplots(figsize=(8, 6))
times = [results[m]["Avg_Time"] for m in methods]
colors = ["#9E9E9E", "#2196F3", "#F44336", "#FF9800", "#4CAF50", "#8BC34A"]
for i, m in enumerate(methods):
    ax.scatter(times[i], f1[i], s=150, c=colors[i], zorder=5, edgecolors='black')
    ax.annotate(short_names[i].replace('\n', ' '), (times[i], f1[i]),
                textcoords="offset points", xytext=(10, 5), fontsize=9)
ax.set_xlabel("Avg Time per Query (s)")
ax.set_ylabel("F1 Score (%)")
ax.set_title("Performance vs Efficiency (F1 vs Time)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "f1_vs_time.png"), dpi=150)
plt.close()
print("Saved: f1_vs_time.png")

# --- Plot 3: Classifier confusion matrices ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, key) in zip(axes, [("Base Classifier", "base_classifier"),
                                    ("Enhanced Classifier", "enhanced_classifier")]):
    cm = np.array(clf_results[key]["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["A", "B", "C"], yticklabels=["A", "B", "C"])
    acc_val = clf_results[key]["accuracy"]
    ax.set_title(f"{name}\n(Accuracy: {acc_val:.1%})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "classifier_confusion.png"), dpi=150)
plt.close()
print("Saved: classifier_confusion.png")

# --- Plot 4: Steps comparison ---
fig, ax = plt.subplots(figsize=(10, 5))
steps = [results[m]["Avg_Steps"] for m in methods]
bars = ax.bar(short_names, steps, color=colors, edgecolor='black')
ax.set_ylabel("Avg Retrieval Steps")
ax.set_title("Average Retrieval Steps per Method")
for bar, val in zip(bars, steps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "steps_comparison.png"), dpi=150)
plt.close()
print("Saved: steps_comparison.png")

# --- Plot 5: Efficiency gain (time savings) ---
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(short_names, times, color=colors, edgecolor='black')
ax.set_ylabel("Avg Time per Query (s)")
ax.set_title("Efficiency: Average Time per Query")
for bar, val in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}s', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "time_comparison.png"), dpi=150)
plt.close()
print("Saved: time_comparison.png")

print(f"\nAll plots saved to {PLOTS_DIR}/")
