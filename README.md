# Bug Report Classification using Machine Learning and DistilBERT

This project improves bug report classification by experimenting with four models:
- Naive Bayes (NB)
- Support Vector Machine (SVM)
- XGBoost
- Fine-tuned DistilBERT

I aim to outperform the baseline (Naive Bayes + TF-IDF) provided in Lab 1 <a href="https://github.com/ideas-labo/ISE-solution/tree/main/lab1">https://github.com/ideas-labo/ISE-solution/tree/main/lab1</a>

## 🔍 Problem Statement
Bug reports often contain noisy text and vary in length and quality. My goal is to classify them as valid or invalid using smarter models that can capture context and semantics.

## 📊 Dataset
- Used bug report datasets like `tensorflow.csv` containing title, body, and labels.
- Merged `Title` and `Body` into a single `text` column.
- Performed text cleaning, stopword removal, and vectorization.

## 🧠 Models & Techniques
- **NB, SVM, XGBoost:** Used TF-IDF vectorized inputs with standard classifiers.
- **DistilBERT:** Fine-tuned with class weighting, dropout tuning, and tokenized using HuggingFace transformers.

## ⚙️ Installation

```bash
pip install transformers datasets scikit-learn xgboost pandas seaborn matplotlib nltk
```
## Folder Structure
<img width="564" alt="image" src="https://github.com/user-attachments/assets/c96542c0-527d-4c2c-b4ed-23c311341464" />

## 🚀 How to Run

1. Place the CSV dataset (e.g., `tensorflow.csv`) in the root folder.
2. To run only BERT, use `Bug_Report_Classification_Bert.ipynb`.
3. Download the result (e.g., `tensorflow_BERT.csv`) and upload it in NB+SVM+XG+DistilBERT.ipynb
4. Run the notebook `NB+SVM_+XG+DistilBERT.ipynb` to compare all models.
   

## 📈 Results

DistilBERT significantly outperforms all baselines on accuracy, precision, recall, F1, and AUC in repeated runs.

## 📁 Files Included

- `requirements.pdf` – Dependencies list
- `manual.pdf` – How to use and run the models
- `replication.pdf` – Steps to replicate the results
- Jupyter notebooks:
  - `Bug_Report_Classification_Bert.ipynb`
  - `NB+SVM+XG+DistilBERT.ipynb`

## 🧪 Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score, Log Loss, AUC
- Repeated runs and averaged results to ensure stability.

## 📎 Citation


---

**Author:** Yashmitha Ramesh  
**Course:** Intelligent Software Engineering, MSc Advanced Computer Science, University of Birmingham  
