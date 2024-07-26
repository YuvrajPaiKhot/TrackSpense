#!/usr/bin/env python
# coding: utf-8

# In[2]:


# train_model.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv('transactions.csv')  # Ensure 'transactions.csv' has 'description' and 'category' columns

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
data['cleaned_description'] = data['description'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_description'])
y = data['category']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_val)
print('Accuracy:', accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# Save model and vectorizer
joblib.dump(model, 'models/expense_categorization_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')


# In[ ]:




