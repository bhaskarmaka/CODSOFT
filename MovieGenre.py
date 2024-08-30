import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

df_test = pd.read_csv(r'C:\Users\bhask\Downloads\test_data.csv', low_memory=False)
df_solution = pd.read_csv(r'C:\Users\bhask\Downloads\test_data_solution.csv', low_memory=False)
df_train = pd.read_csv(r'C:\Users\bhask\Downloads\train_data.csv', low_memory=False)
print("Test data columns:", df_test.columns)
print("Solution data columns:", df_solution.columns)
print("Train data columns:", df_train.columns)
df_test.columns = ['identifier', 'headline', 'summary']
df_solution.columns = ['identifier', 'headline', 'category', 'summary']
df_train.columns = ['identifier', 'headline', 'category', 'summary']
df_combined = pd.concat([df_test, df_solution, df_train], ignore_index=True)
print(df_combined.head())
summary_column = 'summary'
category_column = 'category'
if summary_column in df_combined.columns and category_column in df_combined.columns:
    df_combined[summary_column] = df_combined[summary_column].fillna('')
    df_combined = df_combined.dropna(subset=[category_column])
    label_encoder = LabelEncoder()
    df_combined['encoded_category'] = label_encoder.fit_transform(df_combined[category_column])
    df_train_set, df_test_set = train_test_split(df_combined, test_size=0.2, random_state=50)
    X_train_summaries = df_train_set[summary_column]
    y_train_categories = df_train_set['encoded_category']
    X_test_summaries = df_test_set[summary_column]
    y_test_categories = label_encoder.transform(df_test_set[category_column])
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_summaries)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_summaries)
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_tfidf, y_train_categories)
    y_pred = nb_classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test_categories, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test_categories, y_pred, target_names=label_encoder.classes_))
else:
    print(f"Columns '{summary_column}' and/or '{category_column}' not found in the combined dataset.")