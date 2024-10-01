import streamlit as st
import urllib.request
from MyLibrary import RootCauseAnalysis

if __name__=="__main__":
    question = st.text_input("input here")
    response = None
    if st.button(label="submit"):
        question = f'question:{question}'
        api = RootCauseAnalysis()
        response = api.call(data=question)
        st.write(response)