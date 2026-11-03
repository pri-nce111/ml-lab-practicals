Machine Learning Lab Practicals

A set of 14 beginner-friendly machine learning practicals plus a Titanic case study, written in Python with NumPy, pandas, Matplotlib and scikit-learn.

Contents
File	Topic
1.py	Environment check and Python basics (variables, lists, dictionaries, functions)
2.py	Loading a CSV with pandas and NumPy (iris.csv)
3.py	Data preprocessing: missing values, Min-Max normalization, label encoding
4.py	Linear regression (study hours vs. marks)
5.py	Logistic regression (study hours vs. pass/fail)
6.py	K-Nearest Neighbors with a scaling pipeline and GridSearchCV (Iris)
7.py	Decision Tree vs. Random Forest (Iris)
8.py	Naive Bayes spam classifier with CountVectorizer
9.py	K-Means clustering with a scatter plot
10.py	Support Vector Machine with a linear kernel (Iris)
11.py	Principal Component Analysis (PCA)
12.py	Model evaluation: confusion matrix, accuracy, precision, recall, F1
13.py	Cross-validation comparing Logistic Regression, Decision Tree and SVM
14.py	Neural network (MLPClassifier) on Iris
casestudy.py	Titanic survival prediction with a preprocessing + Random Forest pipeline

Datasets: iris.csv and titanic.csv.

Setup
bash
git clone <your-repo-url>
cd <your-repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
Running

Scripts load the CSV files using relative paths, so run them from the repository root:

bash
python 1.py
python casestudy.py

9.py opens a Matplotlib window with the clustering plot.
