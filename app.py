from flask import Flask, request, jsonify, render_template
import random
import json
import pickle
import numpy as np
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Inicialización del lematizador y carga de recursos
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw')

# Actualización de nltk
nltk.download('popular')

# Cargar archivos y modelo
try:
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model.h5')
except Exception as e:
    print("Error al cargar archivos y modelo:", e)
    exit()

# Inicializar la aplicación Flask
app = Flask(__name__)

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    if not ints:
        # Si no se identificó ninguna intención, se devuelve una respuesta predeterminada
        return "Lo siento, no entiendo tu mensaje. ¿Puedes reformularlo?"
    
    # Se obtiene la etiqueta de la intención identificada
    tag = ints[0]['intent']
    
    # Se busca la intención correspondiente en el JSON de intenciones
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            # Se selecciona una respuesta aleatoria de las respuestas disponibles para esa intención
            return random.choice(intent['responses'])
    
    # Si no se encuentra la intención, se devuelve una respuesta predeterminada
    return "Lo siento, no entiendo tu mensaje. ¿Puedes reformularlo?"



@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    data = request.get_json()
    message = data['message']
    ints = predict_class(message)
    res = get_response(ints, intents)
    return jsonify({'response': res})

# Ruta para servir la página web
@app.route('/')
def index():
    return render_template('index.html')

# Manejo de ruta no encontrada
@app.errorhandler(404)
def page_not_found(e):
    return "Página no encontrada", 404

if __name__ == '__main__':
    app.run(debug=True)
