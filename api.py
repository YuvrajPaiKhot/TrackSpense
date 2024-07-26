from flask import Flask, request, jsonify
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)

# Load the pre-trained model and vectorizer
try:
    model = joblib.load('models/expense_categorization_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')  # Load the vectorizer if you're using one
    app.logger.info("Model and vectorizer loaded successfully")
except Exception as e:
    app.logger.error(f"Failed to load model or vectorizer: {str(e)}")
    model = None
    vectorizer = None

# Text preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(w.lower()) for w in word_tokens if w.lower() not in stop_words and w.isalnum()]
    return ' '.join(filtered_words)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model or vectorizer not loaded'}), 500

    try:
        data = request.get_json()
        app.logger.info(f"Received data: {data}")  # Log the received data
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text field'}), 400
        
        text = data['text']
        app.logger.info(f"Received text: {text}")

        preprocessed_text = preprocess_text(text)
        app.logger.info(f"Preprocessed text: {preprocessed_text}")

        # Transform the text using the vectorizer
        vectorized_text = vectorizer.transform([preprocessed_text])
        
        prediction = model.predict(vectorized_text)
        app.logger.info(f"Prediction: {prediction[0]}")

        return jsonify({'category': prediction[0]})

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
