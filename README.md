# NLP-Text-Classification---SMS-Spam-Detection
Developed a spam detection system utilizing TF-IDF and logistic regression. Reached high accuracy through effective preprocessing and text vectorization
# 📩 NLP Text Classification - SMS Spam Detection

This project applies Natural Language Processing (NLP) and machine learning to classify SMS messages as either **spam** or **ham (not spam)** using the classic **SMS Spam Collection Dataset**.

## 🧠 Objective

Detect whether a given SMS message is spam or not using a logistic regression classifier trained on TF-IDF features extracted from the message text.

---

## 📁 Dataset

- **Source**: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- Columns used:
  - `label`: `"ham"` or `"spam"`
  - `message`: Text of the SMS

---

## ⚙️ Workflow

1. **Data Preprocessing**:
   - Renamed columns to `label` and `message`
   - Encoded `label`: `ham → 0`, `spam → 1`
   - Vectorized messages using **TF-IDF** with 3000 features

2. **Modeling**:
   - Used `LogisticRegression` from scikit-learn
   - Trained on 80% of the dataset and tested on the remaining 20%

3. **Evaluation**:
   - Accuracy, confusion matrix, classification report

4. **Interactive Prediction**:
   - Accepts user input to classify a custom message as spam or ham

---

## 🔍 Example Output

```text
Enter a message to classify as spam or ham: You’ve won $500!
Prediction: spam

Model Performance
Model: Logistic Regression

Vectorization: TF-IDF (max_features=3000)

Displays classification report and heatmap confusion matrix

