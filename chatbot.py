import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json', encoding='utf-8').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bag_of_words(sentence):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                # if show_details:
                    # print ("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence):
    # filter out predictions below a threshold
    bow= bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes [max_index]
    return category

    #ERROR_THRESHOLD = 0.25
    #results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    #results.sort(key=lambda x: x[1], reverse=True)
    #return_list = []
    #for r in results:
       # return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    #return return_list

def get_Response(tag, intents_json):
    #tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result =""
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
        else:
            result = "You must ask the right questions"
    return result

while True:
    message= input("")
    ints = predict_class(message)
    res = get_Response(ints, intents)
    print(res)

#def chatbot_response(msg):
 #   ints = predict_class(msg, model)
  #  res = get_Response(ints, intents)
   # return res
