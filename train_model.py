import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Note: Aapko Kaggle se Fake.csv aur True.csv download karke isi folder mein rakhna hoga
try:
    fake_df = pd.read_csv("archive/Fake.csv")
    true_df = pd.read_csv("archive/True.csv")
    
    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df], axis=0).sample(frac=1).reset_index(drop=True)
    df = df[['text', 'label']]

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        words = text.split()
        cleaned_words = [word for word in words if word not in stop_words]
        return " ".join(cleaned_words)

    df['text'] = df['text'].apply(clean_text)

    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.7)
    X_train_vec = tfidf_vectorizer.fit_transform(X_train)
    X_test_vec = tfidf_vectorizer.transform(X_test)

    model = PassiveAggressiveClassifier(max_iter=50)
    model.fit(X_train_vec, y_train)

    print(f"Model Accuracy: {model.score(X_test_vec, y_test) * 100:.2f}%")

    pickle.dump(model, open('fake_news_model.pkl', 'wb'))
    pickle.dump(tfidf_vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
    print("Model and Vectorizer saved successfully!")
    
except FileNotFoundError:
    print("Abhi data files nahi hain! Pehle folder mein Fake.csv aur True.csv daaliye.")