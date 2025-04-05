import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from jinja2 import Template
from fpdf import FPDF
import os

# ----------------------------
# Data Loading and Alignment
# ----------------------------

def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Define file paths
ground_truth_path = "ground_truth_dataset.jsonl"
mut_predictions_path = "mut_classification_predictions_dataset.jsonl"

ground_truth_data = load_jsonl(ground_truth_path)
mut_predictions_data = load_jsonl(mut_predictions_path)

def align_datasets_ignore_case(ground_truth_data, mut_predictions_data):
    """Aligns datasets by file_name and normalizes document type labels (case-insensitive)."""
    gt_dict = {entry["file_name"]: entry for entry in ground_truth_data}
    mut_dict = {entry["file_name"]: entry for entry in mut_predictions_data}
    
    aligned_data = []
    for file_name in gt_dict:
        if file_name in mut_dict:
            aligned_data.append({
                "file_name": file_name,
                "ground_truth_label": gt_dict[file_name]["ground_truth"]["docType"].strip().lower(),
                "ground_truth_confidence": gt_dict[file_name]["ground_truth"]["confidence"],
                "MUT_prediction_label": mut_dict[file_name]["ground_truth"]["docType"].strip().lower(),
                "MUT_confidence": mut_dict[file_name]["ground_truth"]["confidence"]
            })
    return pd.DataFrame(aligned_data)

df = align_datasets_ignore_case(ground_truth_data, mut_predictions_data)

# ----------------------------
# Compute Core Metrics
# ----------------------------

# Overall Accuracy
correct_count = (df["ground_truth_label"] == df["MUT_prediction_label"]).sum()
total_count = len(df)
accuracy = accuracy_score(df["ground_truth_label"], df["MUT_prediction_label"])

# Define full set of labels with specified order (including "none of the above")
labels = ["invoice", "receipt", "w2", "bank-statement", "none of the above"]

# Classification Report (per document type) with full label set
class_report = classification_report(df["ground_truth_label"], df["MUT_prediction_label"], labels=labels, output_dict=True)
# Exclude overall averages from individual doc type analysis
doc_types = {k: v for k, v in class_report.items() if k not in ["accuracy", "macro avg", "weighted avg"]}
best_doc_type = max(doc_types, key=lambda x: doc_types[x]["f1-score"]) if doc_types else None
worst_doc_type = min(doc_types, key=lambda x: doc_types[x]["f1-score"]) if doc_types else None

# Confusion Matrix (with specified label order)
conf_matrix = confusion_matrix(df["ground_truth_label"], df["MUT_prediction_label"], labels=labels)

# Compute confidence difference between ground truth and MUT predictions
df["confidence_difference"] = abs(df["ground_truth_confidence"] - df["MUT_confidence"])

# ----------------------------
# Function to Format Confusion Matrix as Text
# ----------------------------

def format_confusion_matrix(conf_matrix, labels):
    """
    Returns a neatly aligned string representation of the confusion matrix.
    The function computes each column's width based on the header (label)
    and the maximum length of the numeric values in that column.
    """
    n = len(labels)
    # Compute widths for each predicted column based on header and data
    col_widths = []
    for j in range(n):
        header_val = labels[j].capitalize()
        max_width = len(header_val)
        for i in range(n):
            max_width = max(max_width, len(str(conf_matrix[i, j])))
        # Add padding
        col_widths.append(max_width + 2)
    
    # Determine the width of the row header ("Actual | " plus the longest actual label)
    row_label_width = max(len(label.capitalize()) for label in labels)
    row_header = "Actual | "
    row_header_width = len(row_header) + row_label_width

    # Build top header for predicted columns
    top_header = " " * row_header_width
    for j in range(n):
        top_header += labels[j].capitalize().rjust(col_widths[j])
    top_header += "\n"

    # Build a separator line
    separator = " " * row_header_width + "-" * (sum(col_widths)) + "\n"

    # Build the rows for each actual label
    rows = ""
    for i in range(n):
        # Format the row header with the actual label (right-justified to the row_label_width)
        row = row_header + labels[i].capitalize().rjust(row_label_width)
        for j in range(n):
            row += str(conf_matrix[i, j]).rjust(col_widths[j])
        rows += row + "\n"

    return top_header + separator + rows

# Generate the text version of the confusion matrix
actual_conf_matrix_text = format_confusion_matrix(conf_matrix, labels)

# ----------------------------
# Generate Actual Confidence Chart Text
# ----------------------------

bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
hist, bin_edges = np.histogram(df["confidence_difference"], bins=bins)
max_count = hist.max() if hist.max() > 0 else 1
actual_conf_chart_lines = []
for count in hist:
    num_blocks = int(count / max_count * 15)
    actual_conf_chart_lines.append("|" + "*" * num_blocks)
actual_conf_chart_text = "\n".join(actual_conf_chart_lines)
actual_conf_chart_text += "\n" + "----------------------------\n"
actual_conf_chart_text += "  " + "   ".join([f"{edge:.1f}" for edge in bin_edges[:-1]])

# ----------------------------
# Define Ideal Texts for Reference
# ----------------------------

ideal_conf_matrix_text = """             Predicted
             ---------------------
             Invoice  Receipt  W2  Bank
Actual | Invoice    10       0     0    0
       | Receipt     0      10     0    0
       | W2          0       0    10    0
       | Bank        0       0     0   10"""

ideal_conf_chart_text = """|*************
|*****************
|*******************
|*************
|*****
|*
----------------------------
  0.0   0.2   0.4   0.6   0.8
  (Small Confidence Differences - Good Agreement)"""

# ----------------------------
# Generate Plots
# ----------------------------

reports_folder = "reports"
os.makedirs(reports_folder, exist_ok=True)

# Confusion Matrix Plot
conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label (Azure MUT)")
plt.ylabel("Ground Truth Label (Mistral OCR+LLM)")
plt.title("Confusion Matrix: MUT vs. Ground Truth")
confusion_matrix_path = os.path.join(reports_folder, "confusion_matrix.png")
plt.savefig(confusion_matrix_path)
plt.close()

# Confidence Score Distribution Plot
plt.figure(figsize=(8, 5))
sns.histplot(df["confidence_difference"], bins=10, kde=True, color="purple")
plt.xlabel("Absolute Confidence Score Difference (Mistral vs. Azure)")
plt.ylabel("Count of Documents")
plt.title("Confidence Score Variations Between Mistral and Azure")
confidence_distribution_path = os.path.join(reports_folder, "confidence_distribution.png")
plt.savefig(confidence_distribution_path)
plt.close()

# ----------------------------
# Dynamic Calculations for Report Sections
# ----------------------------

# Section 2: Build an HTML table for per document type metrics
table_rows = ""
for doc, metrics in doc_types.items():
    table_rows += (
        f"<tr>"
        f"<td>{doc.capitalize()}</td>"
        f"<td>{metrics['precision']:.2f}</td>"
        f"<td>{metrics['recall']:.2f}</td>"
        f"<td>{metrics['f1-score']:.2f}</td>"
        f"<td>{metrics['support']}</td>"
        f"</tr>"
    )

"""
# Section 3: Misclassification Insights
misclassification_insights = {}
for doc in sorted(df["ground_truth_label"].unique()):
    gt_subset = df[df["ground_truth_label"] == doc]
    total = len(gt_subset)
    correct = (gt_subset["MUT_prediction_label"] == doc).sum()
    recall_val = correct / total if total > 0 else 0
    misclassified_count = total - correct
    most_common = None
    if misclassified_count > 0:
        misclassified = gt_subset[gt_subset["MUT_prediction_label"] != doc]["MUT_prediction_label"].value_counts()
        if not misclassified.empty:
            most_common = misclassified.idxmax()
    misclassification_insights[doc] = {
        "total": total,
        "correct": correct,
        "recall": recall_val,
        "misclassified": misclassified_count,
        "common_mis": most_common
    }

misclassification_html = "<ul>"
for doc, stats in misclassification_insights.items():
    insight = (
        f"{doc.capitalize()}: Recall = {stats['recall']:.2%} (Correct: {stats['correct']} of {stats['total']}); "
        f"Misclassified: {stats['misclassified']}"
    )
    if stats['common_mis']:
         insight += f", Most common misclassification: {stats['common_mis'].capitalize()}"
    misclassification_html += f"<li>{insight}</li>"
misclassification_html += "</ul>"
"""

# ----------------------------
# Section 3: Misclassification Insights (Updated to Table Format)
# ----------------------------
misclassification_insights = {}
for doc in sorted(df["ground_truth_label"].unique()):
    gt_subset = df[df["ground_truth_label"] == doc]
    total = len(gt_subset)
    correct = (gt_subset["MUT_prediction_label"] == doc).sum()
    recall_val = correct / total if total > 0 else 0
    misclassified_count = total - correct
    most_common = None
    if misclassified_count > 0:
        misclassified = gt_subset[gt_subset["MUT_prediction_label"] != doc]["MUT_prediction_label"].value_counts()
        if not misclassified.empty:
            most_common = misclassified.idxmax()
    misclassification_insights[doc] = {
        "total": total,
        "correct": correct,
        "recall": recall_val,
        "misclassified": misclassified_count,
        "common_mis": most_common
    }

# Build the Misclassification Insights table HTML
misclassification_html = """
<table>
  <tr>
    <th>Document Type</th>
    <th>Recall</th>
    <th>Correct</th>
    <th>Total</th>
    <th>Misclassified</th>
    <th>Most Common Misclassification</th>
  </tr>
"""
for doc, stats in misclassification_insights.items():
    misclassification_html += (
        f"<tr><td>{doc.capitalize()}</td>"
        f"<td>{stats['recall']:.2%}</td>"
        f"<td>{stats['correct']}</td>"
        f"<td>{stats['total']}</td>"
        f"<td>{stats['misclassified']}</td>"
        f"<td>{stats['common_mis'].capitalize() if stats['common_mis'] else 'N/A'}</td></tr>"
    )
misclassification_html += "</table>"


"""
# Section 4: Confidence Score Analysis
confidence_stats = {}
for doc in sorted(df["ground_truth_label"].unique()):
    subset = df[df["ground_truth_label"] == doc]
    avg_mut = subset["MUT_confidence"].mean()
    avg_gt = subset["ground_truth_confidence"].mean()
    drift = abs(avg_mut - avg_gt)
    confidence_stats[doc] = {"avg_mut": avg_mut, "avg_gt": avg_gt, "drift": drift}

low_conf_count = (df["MUT_confidence"] <= 0.25).sum()
highest_conf_doc = max(confidence_stats, key=lambda x: confidence_stats[x]["avg_mut"]) if confidence_stats else None

confidence_html = "<ul>"
for doc, stats in confidence_stats.items():
    confidence_html += (
        f"<li>{doc.capitalize()}: Average MUT Confidence = {stats['avg_mut']:.2f}, "
        f"Ground Truth Confidence = {stats['avg_gt']:.2f}, Drift = {stats['drift']:.2f}</li>"
    )
confidence_html += "</ul>"
confidence_summary = (
    f"Overall, there are {low_conf_count} low-confidence MUT predictions (less than or equal to 0.25). "
    f"The highest average MUT confidence is observed for {highest_conf_doc.capitalize() if highest_conf_doc else 'N/A'}."
)
"""

# ----------------------------
# Section 4: Confidence Score Analysis (Updated to Table Format)
# ----------------------------
confidence_stats = {}
for doc in sorted(df["ground_truth_label"].unique()):
    subset = df[df["ground_truth_label"] == doc]
    avg_mut = subset["MUT_confidence"].mean()
    avg_gt = subset["ground_truth_confidence"].mean()
    drift = abs(avg_mut - avg_gt)
    confidence_stats[doc] = {"avg_mut": avg_mut, "avg_gt": avg_gt, "drift": drift}

low_conf_count = (df["MUT_confidence"] <= 0.25).sum()
highest_conf_doc = max(confidence_stats, key=lambda x: confidence_stats[x]["avg_mut"]) if confidence_stats else None

# Build the Confidence Score Analysis table HTML
confidence_html = """
<table>
  <tr>
    <th>Document Type</th>
    <th>Average MUT Confidence</th>
    <th>Ground Truth Confidence</th>
    <th>Drift</th>
  </tr>
"""
for doc, stats in confidence_stats.items():
    confidence_html += (
        f"<tr><td>{doc.capitalize()}</td>"
        f"<td>{stats['avg_mut']:.2f}</td>"
        f"<td>{stats['avg_gt']:.2f}</td>"
        f"<td>{stats['drift']:.2f}</td></tr>"
    )
confidence_html += "</table>"

confidence_summary = (
    f"Overall, there are {low_conf_count} low-confidence MUT predictions (less than or equal to 0.25). "
    f"The highest average MUT confidence is observed for {highest_conf_doc.capitalize() if highest_conf_doc else 'N/A'}."
)

# ----------------------------
# New Insights: Understanding the Charts (Ideal vs. Actual)
# ----------------------------

ideal_vs_actual_html = f"""
<div class="section">
  <h2>Understanding the Charts: Ideal vs. Actual</h2>
  
  <h3>1. Understanding the Confusion Matrix</h3>
  <h4>What is an "Ideal" Confusion Matrix?</h4>
  <p>An ideal confusion matrix would have high values on the diagonal (correct classifications) and low values elsewhere (misclassifications).</p>
  <h4>Ideal Example (Perfect Classifier)</h4>
  <pre>{ideal_conf_matrix_text}</pre>
  <p>Perfect Model - All correct classifications are along the diagonal, with no misclassifications.</p>
  
  <h4>What Does Our Actual Confusion Matrix Show?</h4>
  <pre>{actual_conf_matrix_text}</pre>
  <p>Dynamic Insights: Compare the counts along the diagonal with the off-diagonal values to understand misclassifications.</p>
  
  <h3>2. Understanding the Confidence Score Chart</h3>
  <h4>What Would an "Ideal" Confidence Score Chart Look Like?</h4>
  <p>Ideally, Mistral and Azure should have very similar confidence scores for each document. The histogram bars should be concentrated around 0.0 - 0.2 (small differences).</p>
  <h4>Ideal Confidence Score Chart</h4>
  <pre>{ideal_conf_chart_text}</pre>
  
  <h4>What Does Our Actual Confidence Score Chart Show?</h4>
  <pre>{actual_conf_chart_text}</pre>
  <p>Dynamic Insights: Larger bars toward higher differences indicate greater disagreement between models.</p>
</div>
"""

# ----------------------------
# Define a helper to clean PDF text
# ----------------------------

def clean_pdf_text(text):
    # Replace any em dash with a standard hyphen.
    return text.replace("\u2014", "-")

# ----------------------------
# HTML Report Generation
# ----------------------------

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; }
        .container { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0px 0px 10px #ccc; }
        h1 { color: #333366; text-align: center; }
        h2 { color: #222; }
        h3 { color: #333; }
        h4 { color: #444; }
        pre { background: #eee; padding: 10px; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        table, th, td { border: 1px solid #ccc; }
        th, td { padding: 8px; text-align: center; }
        .section { margin-bottom: 20px; }
        .chart { text-align: center; }
    </style>
</head>
<body>
    <div class="container">
        <h1>AI-Assisted Evaluation Report</h1>
        
        <!-- Definitions Section -->
        <div class="section">
            <h2>Definitions</h2>
            <p><strong>Precision:</strong> The fraction of predicted positive instances that are truly positive. It measures the accuracy of the positive predictions.</p>
            <p><strong>Recall:</strong> The fraction of actual positive instances that are correctly predicted. It measures the ability to identify all positive instances.</p>
            <p><strong>F1-Score:</strong> The harmonic mean of precision and recall, providing a balance between the two—especially useful when the class distribution is imbalanced.</p>
        </div>
        
        <div class="section">
            <h2>1. Overall Accuracy</h2>
            <p>MUT correctly classified {{ correct_count }} out of {{ total_count }} documents.</p>
            <p><b>Overall Accuracy:</b> {{ accuracy_percentage }}</p>
        </div>
        
        <div class="section">
            <h2>2. Precision, Recall, and F1-Score (Per Document Type)</h2>
            <p><b>Best Performance:</b> {{ best_doc_type }} (F1-score: {{ best_f1 }})</p>
            <p><b>Worst Performance:</b> {{ worst_doc_type }} (F1-score: {{ worst_f1 }})</p>
            <table>
                <tr>
                    <th>Document Type</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
                {{ table_rows|safe }}
            </table>
        </div>
        
        <div class="section">
            <h2>3. Misclassification Insights</h2>
            {{ misclassification_html|safe }}
        </div>
        
        <div class="section">
            <h2>4. Confidence Score Analysis</h2>
            {{ confidence_html|safe }}
            <p>{{ confidence_summary }}</p>
        </div>
        
        <div class="section">
            <h2>Visual Insights</h2>
            <div class="chart">
                <img src="confusion_matrix.png" width="600px"><br>
                <img src="confidence_distribution.png" width="600px">
            </div>
        </div>
        
        {{ ideal_vs_actual_html|safe }}
        
    </div>
</body>
</html>
"""

html_report_path = os.path.join(reports_folder, "evaluation_report.html")
with open(html_report_path, "w") as f:
    f.write(Template(html_template).render(
        correct_count=correct_count,
        total_count=total_count,
        accuracy_percentage=f"{accuracy:.2%}",
        best_doc_type=best_doc_type.capitalize() if best_doc_type else "N/A",
        best_f1=f"{doc_types[best_doc_type]['f1-score']:.2f}" if best_doc_type else "N/A",
        worst_doc_type=worst_doc_type.capitalize() if worst_doc_type else "N/A",
        worst_f1=f"{doc_types[worst_doc_type]['f1-score']:.2f}" if worst_doc_type else "N/A",
        table_rows=table_rows,
        misclassification_html=misclassification_html,
        confidence_html=confidence_html,
        confidence_summary=confidence_summary,
        ideal_vs_actual_html=ideal_vs_actual_html
    ))

# ----------------------------
# PDF Report Generation using FPDF
# ----------------------------

pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# Title
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, clean_pdf_text("AI-Assisted Evaluation Report"), ln=1, align="C")
pdf.ln(5)

# Definitions Section in PDF
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, clean_pdf_text("Definitions"), ln=1)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 10, clean_pdf_text(
    "Precision: The fraction of predicted positive instances that are truly positive. It measures the accuracy of the positive predictions.\n\n"
    "Recall: The fraction of actual positive instances that are correctly predicted. It measures the ability to identify all positive instances.\n\n"
    "F1-Score: The harmonic mean of precision and recall, providing a balance between the two—especially useful when the class distribution is imbalanced."
))
pdf.ln(5)

# Section 1: Overall Accuracy
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, clean_pdf_text("1. Overall Accuracy"), ln=1)
pdf.set_font("Arial", '', 12)
pdf.multi_cell(0, 10, clean_pdf_text(f"MUT correctly classified {correct_count} out of {total_count} documents.\nOverall Accuracy: {accuracy:.2%}"))
pdf.ln(5)

# Section 2: Precision, Recall, and F1-Score
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, clean_pdf_text("2. Precision, Recall, and F1-Score (Per Document Type)"), ln=1)
pdf.set_font("Arial", '', 12)
best_f1_text = f"{doc_types[best_doc_type]['f1-score']:.2f}" if best_doc_type else "N/A"
worst_f1_text = f"{doc_types[worst_doc_type]['f1-score']:.2f}" if worst_doc_type else "N/A"
pdf.multi_cell(0, 10, clean_pdf_text(
    f"Best Performance: {best_doc_type.capitalize() if best_doc_type else 'N/A'} (F1-score: {best_f1_text})\n"
    f"Worst Performance: {worst_doc_type.capitalize() if worst_doc_type else 'N/A'} (F1-score: {worst_f1_text})\n"
    "Detailed Metrics per Document Type:"
))
for doc, metrics in doc_types.items():
    pdf.cell(0, 10, clean_pdf_text(f"{doc.capitalize()}: Precision = {metrics['precision']:.2f}, Recall = {metrics['recall']:.2f}, F1-Score = {metrics['f1-score']:.2f}, Support = {metrics['support']}"), ln=1)
pdf.ln(5)

# Section 3: Misclassification Insights
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, clean_pdf_text("3. Misclassification Insights"), ln=1)
pdf.set_font("Arial", '', 12)
for doc, stats in misclassification_insights.items():
    insight = f"{doc.capitalize()}: Recall = {stats['recall']:.2%} (Correct: {stats['correct']} of {stats['total']}), Misclassified: {stats['misclassified']}"
    if stats['common_mis']:
        insight += f", Most common misclassification: {stats['common_mis'].capitalize()}"
    pdf.multi_cell(0, 10, clean_pdf_text(insight))
pdf.ln(5)

# Section 4: Confidence Score Analysis
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, clean_pdf_text("4. Confidence Score Analysis"), ln=1)
pdf.set_font("Arial", '', 12)
for doc, stats in confidence_stats.items():
    pdf.cell(0, 10, clean_pdf_text(f"{doc.capitalize()}: Average MUT Confidence = {stats['avg_mut']:.2f}, Ground Truth Confidence = {stats['avg_gt']:.2f}, Drift = {stats['drift']:.2f}"), ln=1)
pdf.ln(5)
pdf.multi_cell(0, 10, clean_pdf_text(confidence_summary))
pdf.ln(10)

# Visual Insights
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, clean_pdf_text("Visual Insights"), ln=1)
pdf.ln(5)
pdf.image(confusion_matrix_path, x=10, w=pdf.w - 20)
pdf.ln(85)
pdf.image(confidence_distribution_path, x=10, w=pdf.w - 20)
pdf.ln(10)

# New Section: Understanding the Charts: Ideal vs. Actual
pdf.set_font("Arial", 'B', 14)
pdf.cell(0, 10, clean_pdf_text("Understanding the Charts: Ideal vs. Actual"), ln=1)
pdf.ln(3)

# Confusion Matrix Subsection
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, clean_pdf_text("1. Understanding the Confusion Matrix"), ln=1)
pdf.set_font("Arial", '', 11)
pdf.multi_cell(0, 8, clean_pdf_text("What is an 'Ideal' Confusion Matrix?\nAn ideal confusion matrix would have high values on the diagonal (correct classifications) and low values elsewhere (misclassifications)."))
pdf.ln(2)
pdf.set_font("Arial", 'B', 11)
pdf.cell(0, 8, clean_pdf_text("Ideal Example (Perfect Classifier):"), ln=1)
pdf.set_font("Courier", '', 10)
pdf.multi_cell(0, 8, clean_pdf_text(ideal_conf_matrix_text))
pdf.ln(2)
pdf.set_font("Arial", '', 11)
pdf.multi_cell(0, 8, clean_pdf_text("Perfect Model - All correct classifications are along the diagonal, with no misclassifications."))
pdf.ln(2)
pdf.set_font("Arial", 'B', 11)
pdf.cell(0, 8, clean_pdf_text("What Does Our Actual Confusion Matrix Show?"), ln=1)
pdf.set_font("Courier", '', 10)
pdf.multi_cell(0, 8, clean_pdf_text(actual_conf_matrix_text))
pdf.ln(2)
pdf.set_font("Arial", '', 11)
pdf.multi_cell(0, 8, clean_pdf_text("Dynamic Insights: Compare the counts along the diagonal with the off-diagonal values to understand misclassifications."))
pdf.ln(5)

# Confidence Score Chart Subsection
pdf.set_font("Arial", 'B', 12)
pdf.cell(0, 10, clean_pdf_text("2. Understanding the Confidence Score Chart"), ln=1)
pdf.set_font("Arial", '', 11)
pdf.multi_cell(0, 8, clean_pdf_text("What Would an 'Ideal' Confidence Score Chart Look Like?\nIdeally, Mistral and Azure should have very similar confidence scores for each document. The histogram bars should be concentrated around 0.0 - 0.2 (small differences)."))
pdf.ln(2)
pdf.set_font("Arial", 'B', 11)
pdf.cell(0, 8, clean_pdf_text("Ideal Confidence Score Chart:"), ln=1)
pdf.set_font("Courier", '', 10)
pdf.multi_cell(0, 8, clean_pdf_text(ideal_conf_chart_text))
pdf.ln(2)
pdf.set_font("Arial", 'B', 11)
pdf.cell(0, 8, clean_pdf_text("What Does Our Actual Confidence Score Chart Show?"), ln=1)
pdf.set_font("Courier", '', 10)
pdf.multi_cell(0, 8, clean_pdf_text(actual_conf_chart_text))
pdf.ln(2)
pdf.set_font("Arial", '', 11)
pdf.multi_cell(0, 8, clean_pdf_text("Dynamic Insights: Larger bars toward higher differences indicate greater disagreement between models."))
pdf.ln(5)

pdf_report_path = os.path.join(reports_folder, "evaluation_report.pdf")
pdf.output(pdf_report_path)

print("HTML report generated at:", html_report_path)
print("PDF report generated at:", pdf_report_path)
print("Confusion matrix image at:", confusion_matrix_path)
print("Confidence distribution image at:", confidence_distribution_path)