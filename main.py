import streamlit as st
import pandas as pd
import re
from nltk import word_tokenize
from nltk.corpus import stopwords
import joblib

def remove_redundant_char(text):
    """gets Text as input and removes the redundant single and double charcter words
       returns the cleaned text"""
    out_text = []
    for i in word_tokenize(text):
        if len(i) > 2:
            out_text.append(i)
    out_text = " ".join(out_text)
    return out_text

en_stopwords = stopwords.words('english')
def file_extraction_cleaning(html_content):
    """Gets the content obtained from read_html function
       returns cleaned Text"""
    words_text = ""
    columns = html_content[0].shape[1] 
    for col in range(columns):                                          #For each column in the content                                   
        content = list(html_content[0][col])                        
        for i in content:                                           
            if isinstance(i, str):                                      #removing nan
                for word in word_tokenize(i):                              
                    words = re.findall(r'\b[a-zA-Z]+\b', word)          #removing numbers
                    if len(words) != 0 and words not in en_stopwords:   #removing empty strings and stopwords
                        words_text += ' ' + ' '.join(words)             #framing the text
    words_text = words_text.lower()                                     #converting the text to lowecase
    words_text = remove_redundant_char(words_text)                      #removing redundant characters
    return [words_text]

file = st.file_uploader("Upload your html file")

if file:
    html_content = pd.read_html(file)                       #Reading the file
    words_text = file_extraction_cleaning(html_content)     #Extracting and cleaning the file

    model = joblib.load('svm_doc_classification.pkl')       #Loading the model
    vectorizer = joblib.load('TF_IDF_Vectorizer.pkl')        #Loading the vectorizer
    
    X_new = vectorizer.transform(words_text)                #Transform the new data using the loaded vectorizer
    
    predictions = model.predict(X_new)                      # Make predictions using the loaded model

    st.subheader(f":blue[Predicted Category: ] :green[{predictions[0]}]")  #Display the prediction

    



