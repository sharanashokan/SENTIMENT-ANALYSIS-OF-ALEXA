import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import string
from flask import Flask, request, jsonify, render_template
import nltk
import joblib
import pickle
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import spacy

app = Flask(__name__)

# Define a function for data cleaning
def remove_tag(text):
    """Remove all HTML tags"""
    cleaned = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    return cleaned

def preprocess_data(sentence):
    # Convert to lowercase
    tokens = sentence.lower()

    # Remove HTML tags
    tokens = remove_tag(tokens)

    # Remove punctuations and numbers
    tokens = re.sub('[^a-zA-Z]', ' ', tokens)

    # Remove single characters
    tokens = re.sub(r'\b\w\b', '', tokens)

    # Remove multiple spaces
    tokens = re.sub(r'\s+', ' ', tokens)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens.split() if word not in stop_words]

    # Join the tokens into a single string
    cleaned_sentence = ' '.join(tokens)

    return cleaned_sentence


# Load the machine learning model
model = joblib.load(open('pipeline_model.joblib', 'rb'))

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']  # Get the user input from the HTML form

    # Clean the input data using the preprocess_data function
    cleaned_review = preprocess_data(review)

    # Make a prediction using the machine learning model
    prediction = model.predict([cleaned_review])

    # Map the prediction to a sentiment label
    if prediction == 1:
        sentiment = 'positive'
        print(sentiment)
    else:
        sentiment = 'negative'
        print(sentiment)


    # Render the result in the result.html template
    return render_template('result.html', review=review, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)

