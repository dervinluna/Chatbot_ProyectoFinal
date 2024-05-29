import random
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas

lemmatizer = WordNetLemmatizer()
DATA_FILE = 'intents.json'

# Cargar los datos iniciales
intents = json.loads(open(DATA_FILE, encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def read_data():
    with open(DATA_FILE, 'r', encoding='utf-8') as file:
        return json.load(file)

def write_data(data):
    with open(DATA_FILE, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

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

# Rutas CRUD para intents
# GET /intents: Obtener todas las intenciones
@app.route('/intents', methods=['GET'])
def get_intents():
    data = read_data()
    return jsonify(data)

# GET /intent/<tag>: Obtener una intención por su tag
@app.route('/intent/<string:tag>', methods=['GET'])
def get_intent(tag):
    data = read_data()
    intent = next((item for item in data['intents'] if item['tag'] == tag), None)
    if intent is None:
        return jsonify({"error": "Intent not found"}), 404
    return jsonify(intent)

# POST /intent: Crear una nueva intención
@app.route('/intent', methods=['POST'])
def create_intent():
    data = read_data()
    new_intent = request.get_json()

    if 'tag' not in new_intent or not new_intent['tag']:
        return jsonify({"error": "Tag is required"}), 400

    if any(item['tag'] == new_intent['tag'] for item in data['intents']):
        return jsonify({"error": "Intent with this tag already exists"}), 400
    
    data['intents'].append(new_intent)
    write_data(data)
    return jsonify(new_intent), 201

# PUT /intent/<tag>: Actualizar una intención por su tag
@app.route('/intent/<string:tag>', methods=['PUT'])
def update_intent(tag):
    data = read_data()
    intent = next((item for item in data['intents'] if item['tag'] == tag), None)
    if intent is None:
        return jsonify({"error": "Intent not found"}), 404
    update_data = request.get_json()
    intent.update(update_data)
    write_data(data)
    return jsonify(intent)

# DELETE /intent/<tag>: Eliminar una intención por su tag
@app.route('/intent/<string:tag>', methods=['DELETE'])
def delete_intent(tag):
    data = read_data()
    intent = next((item for item in data['intents'] if item['tag'] == tag), None)
    if intent is None:
        return jsonify({"error": "Intent not found"}), 404
    data['intents'].remove(intent)
    write_data(data)
    return jsonify({"message": "Intent deleted"})

if __name__ == '__main__':
    import sys
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    app.run(debug=True)
