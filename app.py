# Core Pkgs
import streamlit as st
import altair as alt

# EDA pkgs
import pandas as pd
import numpy as np
import base64

# Utils
import joblib
pipe_bag = joblib.load(open('emotion.pkl','rb'))

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-repeat: no-repeat;
        background-size: 1925px 1100px
        
    }}
    </style>
    """,
    unsafe_allow_html=True)
    
add_bg_from_local('1.jpg') 
# Functions

def predict_emotions(docx):
    results = pipe_bag.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_bag.predict_proba([docx])
    return results

emotions_emoji_dict = {"Negative":"😠","Neutral":"😊","Positive":"😃"}

def main():
    
    html_temp = """
    <div style = "text-color:#025246;padding:10px">
    <h1 style = "color:white;text-align:center;">Customer Sentiment Analyzer App</h1>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    st.subheader("Enter the Text")
    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("")
        submit_text = st.form_submit_button(label='Submit')
        
    if submit_text:
        col1,col2 = st.columns(2)
            
        # Apply function here
        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)
            
        with col1:
            st.success("Original Text")
            st.write(raw_text)
            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write("{}:{}".format(prediction,emoji_icon))
            st.write("Confidence:",format(np.max(probability)))
                
        with col2:
            st.success("Prediction probability")
            st.write(probability)
            proba_df = pd.DataFrame(probability,columns=pipe_bag.classes_)
            st.write(proba_df.T)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["Sentiment","probability"]                
            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Sentiment',y='probability',color='Sentiment')
            st.altair_chart(fig, use_container_width=True)

        
        
with st.sidebar:
    st.subheader("Mentor :- ")
    st.text("1.Varun Vennelaganti")
    st.text("2.Deepika P")
        
        
with st.sidebar:
    st.subheader("Made By :- ")
    st.text("1.Ashutosh Jha")
    st.text("2.Sameer Herkal")
    st.text("3.Suresh Solanki")
    st.text("4.Aditya Mhadye")
    st.text("5.Apeksha Singh")    
    st.text("6.More Murali Krishna")
    st.text("7.Vivek Naredla")

            
                 
    
if __name__ == "__main__":
    main()                
                
               