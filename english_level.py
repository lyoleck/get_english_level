import streamlit as st
import pickle
import re
import numpy as np
import pandas as pd
import catboost as cb
from nltk.stem.porter import PorterStemmer

def main():
    st.title("Upload the subtitles file")
    st.write("It should be a *.srt file:")

    uploaded_file = st.file_uploader("Choose your file", type=["srt"])

    if uploaded_file is not None:
        st.write("File successfully uploaded!")
        # creating dataframe
        df = pd.DataFrame()
        subtitle_text = load_subtitles(uploaded_file)
        subtitle_text = clean_subtitles(subtitle_text)
        df['subtitle_text'] = [subtitle_text]
        # loading vocabularies
        a1_voc = pd.read_pickle('./data/a1.pkl')
        a2_voc = pd.read_pickle('./data/a2.pkl')
        b1_voc = pd.read_pickle('./data/b1.pkl')
        b2_voc = pd.read_pickle('./data/b2.pkl')
        c1_voc = pd.read_pickle('./data/c1.pkl')
        # loading model
        model = pd.read_pickle('./data/mymodel.pkl')
        # operations
        df = calculate_word_length(df)
        df['subtitle_text'] = df['subtitle_text'].apply(tokenizer_porter)
        df['A1_count'] = df['subtitle_text'].apply(lambda text: count_words(text, a1_voc))
        df['A2_count'] = df['subtitle_text'].apply(lambda text: count_words(text, a2_voc))
        df['B1_count'] = df['subtitle_text'].apply(lambda text: count_words(text, b1_voc))
        df['B2_count'] = df['subtitle_text'].apply(lambda text: count_words(text, b2_voc))
        df['C1_count'] = df['subtitle_text'].apply(lambda text: count_words(text, c1_voc))
        features = df[['A1_count', 'A2_count', 'B1_count', 'B2_count', 'C1_count', 'word_length']]
        input_data = features.iloc[0].values
        prediction = model.predict(input_data)
        value = prediction[0]
        st.write("## Your subtitles English level is: ", value)

def load_subtitles(file):
    try:
        content = file.read().decode("utf-8")
        return content
    except Exception as e:
        return ""

def clean_subtitles(text):
    cleaned_text = re.sub(r'<.*?>', '', text)
    cleaned_text = cleaned_text.replace('\n', ' ')
    pattern = r'[^a-zA-Z\s]'
    cleaned_text = re.sub(pattern, '', cleaned_text)
    return cleaned_text.lower()

def calculate_word_length(subtitles_df):
    def word_length(text):
        words = re.findall(r'\b\w+\b', text)  # Use regular expression to extract words
        word_lengths = [len(word) for word in words]
        return pd.Series(word_lengths).mean()
    subtitles_df['word_length'] = subtitles_df['subtitle_text'].apply(word_length)
    return subtitles_df

def tokenizer_porter(text):
    porter = PorterStemmer()
    return porter.stem(text)

def count_words(text, word_set):
    words_in_text = text.split()
    words_to_count = set(word_set['Words'])
    count = sum(word.lower() in words_to_count for word in words_in_text)
    return count

main()
