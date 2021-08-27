import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
import sys, os
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer


model_name = 'rf_model.pk'
vectorizer_name = 'tfidf_vectorizer.pk'

model_path = os.path.join('/', model_name)
vect_path = os.path.join('/', vectorizer_name)

loaded_model = pickle.load(open('rf_model.pk', 'rb'))
loaded_vect = pickle.load(open('tfidf_vectorizer.pk', 'rb'))



stemmer = SnowballStemmer("english")
stop = stopwords.words('english')

def cleaner(review):
    word_list = nltk.word_tokenize(review)
    clean_list = []
    for word in word_list:
        if word.lower() not in stop:
            stemmed = stemmer.stem(word)
            clean_list.append(stemmed)
    return " ".join(clean_list)

def raw_test(review, model, vectorizer):
    # Clean Review
    review_c = cleaner(review)
    # Embed review using tf-idf vectorizer
    embedding = vectorizer.transform([review_c])
    # Predict using your model
    prediction = model.predict(embedding)
    # Return the Sentiment Prediction
    return "Positive" if prediction == 1 else "Negative"
	

def welcome():
    return 'welcome all'

# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    st.title('Amazon Food Review')
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;">Streamlit Amazon Food Review Classifier ML App </h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    your_review = st.text_input("Enter your review", "Type Here")
    result =""
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
    if st.button("Predict"):
        result = raw_test(your_review,loaded_model,loaded_vect)
    st.success('The output is {}'.format(result))


if __name__=='__main__':
    main()
