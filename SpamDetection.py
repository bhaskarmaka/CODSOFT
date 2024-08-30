import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(r'C:\Users\bhask\Downloads\sms.csv')
print(df.head())
print(df.columns)
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
df = df[['label', 'message']]
print(df.isnull().sum())
df.dropna(inplace=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
X = df['message']
y = df['label']
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=50)
log_reg = LogisticRegression()
naive_bayes = MultinomialNB()
svm = SVC()
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
print("Logistic Regression")
print(f'Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}')
print(classification_report(y_test, y_pred_lr))
print(confusion_matrix(y_test, y_pred_lr))
naive_bayes.fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
print("Naive Bayes")
print(f'Accuracy: {accuracy_score(y_test, y_pred_nb):.2f}')
print(classification_report(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("Support Vector Machine")
print(f'Accuracy: {accuracy_score(y_test, y_pred_svm):.2f}')
print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))