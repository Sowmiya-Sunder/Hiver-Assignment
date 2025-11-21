README – Part A: Email Tagging Mini-System 

1. Approach

I combined each email’s subject + body, converted them into TF-IDF vectors, and trained a simple Logistic Regression classifier as the baseline. A lightweight rule-engine (patterns + anti-patterns) was added to improve accuracy for keyword-driven tags like billing, CSAT, and feature requests.

2. Model / Prompt

Model: TF-IDF + Logistic Regression

Patterns: direct keyword → tag overrides (e.g., “billing”, “invoice” → billing)

Anti-patterns: detect misleading keywords and adjust or flag low-confidence predictions (e.g., “rule”, “workflow”, “delay”).

Confidence: model probability score is returned alongside the predicted tag.

3. Customer Isolation

Each customer has their own tag set.
Before predicting, I filter the model’s output to only the tags belonging to that customer:

final_prediction = argmax(probabilities over allowed_tags_for_customer)

This ensures zero tag leakage between customers.

4. Error Analysis

Misclassifications were inspected manually.
Most errors came from:

very small dataset size,
overlapping vocabulary (“delay”, “rule”),
short or ambiguous emails.

