import streamlit as st
import os
import gdown
import zipfile
from transformers import pipeline, AutoModelForAudioClassification, AutoModel, AutoFeatureExtractor

# 1. Download and Extract Model from Drive
@st.cache_resource
def setup_models():
    # The folder ID is the part of the URL after "folders/"
    folder_id = '1Vw_CRVKAlsVikX-GQB1hvaxFnPuhWBKA'
    
    # 1. Download the entire folder structure
    if not os.path.exists("./Models"):
        # This creates a folder named 'Models' (or whatever it's called on Drive)
        # and downloads all nested files into it.
        gdown.download_folder(id=folder_id, output='./Models', quiet=False)
    
    # 2. Initialize Pipelines
    # Note: Ensure the folder names match exactly what was downloaded
    # (Sometimes gdown nests them, so check the path if this fails)
    base_path = "./Models" 
    
    from transformers import pipeline
    gender_pipe = pipeline("audio-classification", model=f"{base_path}/gender_model")
    age_pipe = pipeline("automatic-speech-recognition", model=f"{base_path}/age_model")
    emotion_pipe = pipeline("audio-classification", model=f"{base_path}/emotion_model")
    
    return gender_pipe, age_pipe, emotion_pipe

# 3. Streamlit Interface
st.title("Voice Age & Emotion Detector")
gender_pipe, age_pipe, emotion_pipe = setup_models()

uploaded_file = st.file_uploader("Upload a male voice note", type=["wav", "mp3"])

if uploaded_file:
    # Logic implementation
    # 1. Gender check
    gender_result = gender_pipe(uploaded_file)
    
    if gender_result[0]['label'] == 'female': # Update label based on your specific model
        st.error("Upload male voice.")
    else:
        # 2. Age Check
        age_result = age_pipe(uploaded_file)
        age = int(age_result['text']) # Assuming ASR model outputs age
        
        if age >= 60:
            # 3. Emotion Check
            emotion = emotion_pipe(uploaded_file)
            st.write(f"Age: {age} (Senior Citizen)")
            st.write(f"Emotion: {emotion[0]['label']}")
        else:
            st.write(f"Age: {age}")
