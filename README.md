# Who’s Testing AI, and How? An AI-Assisted Evaluation Framework

Who is testing AI, and how? Traditional AI evaluation relies on static datasets, manual validation, and predefined benchmarks, but these methods struggle to scale with evolving AI models, that are often iteratively trained to improve their performance. Project Ganymede, an AI-Assisted Evaluation Framework automates AI model testing by using one AI model to generate ground truth data and benchmark another—without much human intervention.

A key innovation in this approach is leveraging Mistral, a large language model (LLM), in a discriminative role rather than a purely generative one. By classifying documents based on OCR-extracted text, the LLM functions as an adaptive ground truth generator, enabling dynamic, AI-driven evaluation of other AI models performing classification or extraction tasks.

While this project demonstrates the concept in financial document classification, the methodology is universal and can be extended to evaluate AI models in NLP, computer vision, and other machine learning applications.

Highlights:
✅ How AI can autonomously generate ground truth datasets for model evaluation
✅ How LLMs can function as discriminative models for classification benchmarking
✅ Applying precision, recall, F1-score, and confidence calibration in AI evaluation
✅ The broader impact of AI-driven evaluation beyond document classification

## Project Overview

The project consists of three main components that work together in sequence:

1. **Ground Truth Generator** (`ground_truth_gen.py`): Uses Mistral AI to create a reliable ground truth dataset
2. **Azure Document Classifier** (`az_fin_doc_classifier.py`): Processes documents using Azure Document Intelligence (the Model Under Test)
3. **Automated Evaluation** (`automated_evaluation.py`): Compares results and generates comprehensive evaluation reports

## Architecture and Workflow

### 1. Ground Truth Generation

The `ground_truth_gen.py` script:
- Connects to Azure Blob Storage to access financial documents
- Uses Mistral AI to perform OCR and classify documents into categories (Invoice, Receipt, W-2, Bank Statement)
- Stores results in JSONL format with document name, classification, confidence score, and supporting text
- Implements retry logic and batch processing for reliability

```
python ground_truth_gen.py
```

### 2. Azure Document Classification

This is our MUT - Model Under Test (aka. Application Under Test)

The `az_fin_doc_classifier.py` script:
- Connects to the same Azure Blob Storage to access the documents
- Submits each document to Azure Document Intelligence using a custom classification model
- Collects classification results and confidence scores
- Stores results in JSONL format for comparison

```
python az_fin_doc_classifier.py
```

### 3. Automated Evaluation

The `automated_evaluation.py` script:
- Aligns the ground truth data with Azure's predictions
- Calculates comprehensive metrics (accuracy, precision, recall, F1-score)
- Generates confusion matrices and confidence distribution charts
- Creates detailed HTML and PDF reports with insights and visualizations

```
python automated_evaluation.py
```

## AI-Driven Evaluation Approach

This project uses an innovative approach to evaluate AI models:

### Using AI to Generate Ground Truth

Traditional evaluation requires manual labeling, which is time-consuming and expensive. Our approach:

1. **AI-Generated Ground Truth**: We use Mistral AI, a powerful large language model, to analyze documents and generate high-quality classifications
2. **Document Understanding**: Mistral AI performs OCR and uses its knowledge to classify documents based on their content and structure
3. **Confidence Scoring**: Each classification includes a confidence score to indicate reliability

### Comprehensive Evaluation Framework

The evaluation framework provides:

1. **Quantitative Metrics**: Accuracy, precision, recall, and F1-scores for overall and per-document type performance
2. **Visual Analysis**: Confusion matrices and confidence distribution charts
3. **Comparative Insights**: Analysis of where the Model Under Test (MUT) agrees or disagrees with the ground truth
4. **Educational Components**: The reports include explanations of ideal vs. actual performance to help interpret results

## Setup and Requirements

1. Run the setup script to configure environment variables:
```
bash setup_env.sh
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

3. Required credentials:
   - Azure Document Intelligence endpoint, key, and custom model ID
   - Azure Blob Storage URL and SAS token
   - Mistral API key

## Reports

The evaluation generates two types of reports:
- HTML report with interactive elements (`reports/evaluation_report.html`)
- PDF report for sharing and printing (`reports/evaluation_report.pdf`)

Both reports include visualizations and detailed analysis of the model's performance.
