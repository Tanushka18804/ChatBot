import os
import json
import datetime
import csv
import pickle
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL setup for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# File paths
BASE_DIR = os.path.abspath(".")
INTENTS_FILE = os.path.join(BASE_DIR, "intents.json")
MODEL_FILE = os.path.join(BASE_DIR, "chatbot_model.pkl")
VECTORIZER_FILE = os.path.join(BASE_DIR, "vectorizer.pkl")
CHAT_LOG_FILE = os.path.join(BASE_DIR, "chat_log.csv")

# Load intents
if os.path.exists(INTENTS_FILE):
    with open(INTENTS_FILE, "r") as file:
        intents = json.load(file)
else:
    st.error("Error: intents.json file not found!")
    st.stop()

# Check if model exists to avoid retraining
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    with open(MODEL_FILE, 'rb') as model_file, open(VECTORIZER_FILE, 'rb') as vectorizer_file:
        clf = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)
else:
    # Train model if not found
    tags, patterns = [], []
    for intent in intents:
        for pattern in intent['patterns']:
            tags.append(intent['tag'])
            patterns.append(pattern)

    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(patterns)
    clf = LogisticRegression(random_state=0, max_iter=10000)
    clf.fit(x, tags)

    # Save the model
    with open(MODEL_FILE, 'wb') as model_file, open(VECTORIZER_FILE, 'wb') as vectorizer_file:
        pickle.dump(clf, model_file)
        pickle.dump(vectorizer, vectorizer_file)

def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I'm not sure how to respond to that."

# Initialize Streamlit
st.title("Chatbot using NLP & Logistic Regression")
menu = ["Home", "Conversation History", "About"]
choice = st.sidebar.selectbox("Menu", menu)

# Create chat log file if it doesn't exist
if not os.path.exists(CHAT_LOG_FILE):
    with open(CHAT_LOG_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

if choice == "Home":
    st.write("Type a message and press Enter to chat.")

    if "counter" not in st.session_state:
        st.session_state.counter = 0
    st.session_state.counter += 1

    user_input = st.text_input("You:", key=f"user_input_{st.session_state.counter}")

    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{st.session_state.counter}")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(CHAT_LOG_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([user_input, response, timestamp])

        if response.lower() in ['goodbye', 'bye']:
            st.write("Thank you for chatting with me! 😊")
            st.stop()

elif choice == "Conversation History":
    st.header("Conversation History")
    try:
        with open(CHAT_LOG_FILE, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)  # Skip the header
            chat_data = list(csv_reader)
        if chat_data:
            st.dataframe(chat_data, columns=["User Input", "Chatbot Response", "Timestamp"], width=800)
        else:
            st.write("No conversation history available.")
    except FileNotFoundError:
        st.error("No chat history found!")

elif choice == "About":
    st.write("""
    ### About This Project
    - This chatbot is built using **Natural Language Processing (NLP)** and **Logistic Regression**.
    - The model is trained using a dataset of predefined intents and responses.
    - The chatbot interface is powered by **Streamlit**.
    
    ### How It Works
    1. The user inputs a message.
    2. The chatbot processes the message using **TF-IDF Vectorization** and predicts an intent using **Logistic Regression**.
    3. Based on the predicted intent, the chatbot generates a response.
    
    ### Future Enhancements
    - Improve intent recognition with **Deep Learning** (e.g., LSTMs, Transformers).
    - Add **context-based responses** for more natural conversations.
    - Use a **larger dataset** for better accuracy.
    
    🚀 Developed with ❤️ using Python.
    """)
