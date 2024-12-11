import random
import json
import pickle
import numpy as np
import nltk
import streamlit as st
from nltk.stem import WordNetLemmatizer
from keras.models import load_model  # type: ignore

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the intents JSON file
intents = json.loads(open('D:/chatbot/chatbot/intents.json').read())

# Load the words, classes, and model
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Function to clean up the sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Function to create a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class of the sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Function to get the response based on the predicted intent
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['intent'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Streamlit UI
st.title("Chatbot")
st.write("Type a message to interact with the chatbot:")

# Create a text input field for the user to type a message
user_input = st.text_input("You: ")

if user_input:
    # Predict the class of the message
    intents = predict_class(user_input)
    response = get_response(intents, intents)

    # Display the response in the Streamlit interface
    st.write(f"Bot: {response}")
