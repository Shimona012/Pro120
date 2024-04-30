#Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')

import os
if os.name == 'nt':
    os.system('color')
# words to be igonred/omitted while framing the dataset
ignore_words = ['?', '!',',','.', "'s", "'m"]

import json
import pickle

import numpy as np
import random

# Model Load Lib
import tensorflow
from data_preprocessing import get_stem_words

# load the model
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))


def preprocess_user_input(user_input):

    bag=[]
    bag_of_words = []

    # tokenize the user_input
    input_word_token_1 = nltk.word_tokenize(user_input)                       
    # convert the user input into its root words : stemming
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words)
    # Remove duplicacy and sort the user_input
    input_word_token_2 = sorted(list(set(input_word_token_2)))
    # Input data encoding : Create BOW for user_input
    for word in words:            
        if word in input_word_token_2:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0) 
    bag.append(bag_of_words)

    return np.array(bag)
    
def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
  
    prediction = model.predict(inp)
   
    predicted_class_label = np.argmax(prediction[0])
    
    return predicted_class_label


def bot_response(user_input):
   
   predicted_class_label =  bot_class_prediction(user_input)
 
   # extract the class from the predicted_class_label
   predicted_class = classes[predicted_class_label]

   # now we have the predicted tag, select a random response

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
       
       # choose a random bot response
        bot_response = random.choice(intent['responses'])
        
    
        return bot_response
    


print("Hi I am Stella, How Can I help you?")

while True:

    # take input from the user
    user_input = input('\n Type your message here : ')
    print("\n User Input: ", user_input)
   
    print("\n\033[0m ~ ~ ~ \033[91mP\033[93mL\033[92mE\033[94mA\033[95mS\033[0mE\033[95m \033[94mW\033[92mA\033[93mI\033[91mT\033[0m ~ ~ ~ \n")

    response = bot_response(user_input)
    print("\n Bot Response:  ", response)
