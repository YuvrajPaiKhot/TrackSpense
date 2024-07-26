#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# api.py

from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load('models/expense_categorization_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Preprocessing function
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/categorize', methods=['POST'])
def categorize():
    description = request.json['description']
    cleaned_description = preprocess_text(description)
    features = vectorizer.transform([cleaned_description])
    category = model.predict(features)[0]
    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=False)


# In[ ]:




