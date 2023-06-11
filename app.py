import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
import pickle
import streamlit as st
import random

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

intents_file = 'intents.json'
model_file = 'chatbot_model.h5'
words_file = 'words.pickle'
classes_file = 'classes.pickle'
conversation_history_file = 'conversation_history.json'

# Load the trained model
model = load_model(model_file)

with open(intents_file) as file:
    intents_data = json.load(file)

# Load words and classes
with open(words_file, 'rb') as f:
    words = pickle.load(f)

with open(classes_file, 'rb') as f:
    classes = pickle.load(f)

ignore_words = ['?', '!']

intents = intents_data['intents']

# Preprocess input
def preprocess_input(user_input):
    tokens = word_tokenize(user_input.lower())
    tokens = sorted(set([lemmatizer.lemmatize(token) for token in tokens if token not in ignore_words]))
    return tokens

# Predict intent
def predict_intent(user_input):
    processed_input = preprocess_input(user_input)
    bag = [1 if word in processed_input else 0 for word in words]
    input_data = np.array([bag])
    result = model.predict(input_data)[0]
    predicted_intent_index = np.argmax(result)
    predicted_intent_tag = classes[predicted_intent_index]
    return predicted_intent_tag

# Get response
def get_chatbot_response(intent_tag):
    for intent in intents:
        if intent['tag'] == intent_tag:
            responses = intent['responses']
            return random.choice(responses)
    return "I'm sorry, but I don't understand."

# Load conversation history from file
def load_conversation_history():
    try:
        with open(conversation_history_file, 'r') as f:
            conversation_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        conversation_history = []
    return conversation_history

# Save conversation history to file
def save_conversation_history(conversation_history):
    with open(conversation_history_file, 'w') as f:
        json.dump(conversation_history, f)
        
##def del_entry(val):
##    with open(conversation_history_file, 'r') as f:
##        data = json.load(f)
##        data=list(data)
##        del data[val]
##        
##    with open(conversation_history_file, 'w') as f:
##        json.dump(data, f)
        

    
# Streamlit UI
def main():
    # Set page config
    st.set_page_config(page_title="ChatBot")

    # ChatBot header
    st.title("ChatBot")

    # Create a container for the chatbox
    chatbox_container = st.container()
    chatbox_expander = chatbox_container.expander("ChatBot")

    # Load conversation history
    conversation_history = load_conversation_history()

    with chatbox_expander:
        st.markdown("Welcome! Start a conversation with the ChatBot.")

        # User input
        user_input = st.text_input("You:")

        # Process user input and get response
        if user_input:
            predicted_intent_tag = predict_intent(user_input)
            response = get_chatbot_response(predicted_intent_tag)

            # Update conversation history
            conversation_history.append(('User', user_input))
            conversation_history.append(('ChatBot', response))

            # Save conversation history
            save_conversation_history(conversation_history)

    # Display conversation history
    if len(conversation_history) > 0:
        st.markdown("---")
        st.markdown("**Chat History**")

        # Clear conversation history
        if st.button("Clear History"):
            conversation_history = []
            save_conversation_history(conversation_history)

        with st.expander("Conversation History", expanded=True):
            for i, (role, message) in enumerate(conversation_history):
                st.text_area(role + ":", value=message, height=80, key=i, disabled=True)

##                # Delete individual message
##                if st.button("Delete", key=str(i)):
##                    del_entry(i)
##                    break

if __name__ == '__main__':
    main()
