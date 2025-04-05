Extraction Workflow Evaluation Metrics

A. Field-Level Accuracy Metrics
 Precision : % of extracted fields that are correct (by field name + value).
 Recall : % of expected fields that were correctly extracted.
 F1-Score : Balance between precision and recall — reported per field and overall.

B. Matching and Similarity Metrics
 Exact Match Rate : % of fields with an exact match between extracted and ground truth key-value pairs.
 Levenshtein Distance : Character-level difference between extracted and ground truth values (for fuzzy matches).
 Normalized Value Accuracy : Accuracy after normalization (e.g., stripping currency symbols, reformatting dates).

C. Coverage & Completeness
 Field Coverage Rate : Expected field not found in model attempted to extract.

D. Confidence-Based Evaluation
 Confidence Score Distribution : Distribution of model confidence across extracted fields.
 Confidence vs. Accuracy Correlation : Compare confidence for correct vs incorrect predictions.
 Low-Confidence Failure Rate : % of incorrect predictions below a threshold (e.g., 0.25).

E. Error Categorization
 Missing Field Errors : Expected field not found in model output.
 Incorrect Value Errors : Field found but extracted value is wrong.
 Mismatched Keys (Wrong Association) : Value extracted under the wrong field/key.

F. Per-Document Performance
 Per-Document F1-Score : Granular view of how each document performs.
 Field-wise Errors by Document : Track which fields fail on which files.

G. Table/Line Item Extraction
 Row Match Rate : % of rows correctly extracted in tables (e.g., invoice line items).
 Cell-Level Accuracy : Similar to field-level metrics but applied at table cell level.
 Structural Accuracy : How well the model preserved table structure and order.

