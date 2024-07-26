from flask import Flask, request, jsonify
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import joblib

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('expense_classifier.pkl')

# Text preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_words = [lemmatizer.lemmatize(w.lower()) for w in word_tokens if w.lower() not in stop_words and w.isalnum()]
    return ' '.join(filtered_words)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    preprocessed_text = preprocess_text(text)
    prediction = model.predict([preprocessed_text])
    return jsonify({'category': prediction[0]})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
