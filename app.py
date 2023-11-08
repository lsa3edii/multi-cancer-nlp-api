# !pip install flask

import numpy as np
import tensorflow as tf
import sklearn
import nltk
import pickle
import re

from keras.models import load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing import sequence
from flask import Flask, request, jsonify

model = load_model('model(text).h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as handle:
    le = pickle.load(handle)

lemmatizer = WordNetLemmatizer()

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = word_tokenize(text)

    words = []
    for word in tokens:
        if word not in stop_words and len(word) > 2:
            word = lemmatizer.lemmatize(word, 'n')
            word = lemmatizer.lemmatize(word, 'v')
            word = lemmatizer.lemmatize(word, 'a')
            word = lemmatizer.lemmatize(word, 'r')
            words.append(word)

    text = ' '.join(words)
    return text

app = Flask(__name__)

@app.route('/predict-text', methods = ['POST'])
def classify_text():
    try:
        text = request.form['text']

        preprocessed_text = preprocess_text(text)
        seq = tokenizer.texts_to_sequences([preprocessed_text])
        text = sequence.pad_sequences(seq, maxlen = 1000, padding = 'post')

        prediction = model.predict(text)
        predicted_class = np.argmax(prediction)
        predicted_class_name = le.inverse_transform([predicted_class])[0]

        return jsonify(
            {
                'prediction': predicted_class_name,
                'accuracy': np.double(prediction[0][predicted_class]) * 100
            }
        )
    except Exception as ex:
        return jsonify(
            {
                'error': str(ex)
            }
        )

if __name__ == '__main__':
    app.run(debug = False, port = 5000)
