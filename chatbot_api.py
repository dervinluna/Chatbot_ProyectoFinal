import random
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json', encoding='utf-8').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "You must ask the right questions"

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    try:
        req_data = request.get_json()
        message = req_data.get('message')
        if message:
            tag = predict_class(message)
            response = get_response(tag, intents)
            return jsonify({"response": response}), 200, {'Content-Type': 'application/json; charset=utf-8'}
        return jsonify({"response": "No message provided"}), 400, {'Content-Type': 'application/json; charset=utf-8'}
    except Exception as e:
        return jsonify({"error": str(e)}), 500, {'Content-Type': 'application/json; charset=utf-8'}

if __name__ == '__main__':
    import sys
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    app.run(debug=True)
