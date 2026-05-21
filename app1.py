import streamlit as st
import os
import gdown
import librosa
import numpy as np
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
    age_pipe = pipeline("audio-classification", model=f"{base_path}/age_model")
    emotion_pipe = pipeline("audio-classification", model=f"{base_path}/emotion_model")
    
    return gender_pipe, age_pipe, emotion_pipe

# --------------------------------------------------------

# 3. Streamlit Interface
st.title("Voice Age & Emotion Detector")
gender_pipe, age_pipe, emotion_pipe = setup_models()

uploaded_file = st.file_uploader("Upload a male voice note", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')

    # LOAD AUDIO PROPERLY
    # 1. Load the audio file into a numpy array (y) and sampling rate (sr)
    # librosa.load accepts the file-like object directly
    y, sr = librosa.load(uploaded_file, sr=16000)
    
    # 2. Pass 'y' (the numpy array) to the pipes instead of 'uploaded_file'
    gender_results = gender_pipe(y)
    gender = gender_results[0]['label'].lower()
    
    # Reset pointer for next model
    uploaded_file.seek(0)
    
    if 'female' in gender:
        st.error("Upload male voice.")
    else:
        # 2. Age Detection (Updated for Classification Model)
        age_results = age_pipe(y)
                
        # 2. Get the label with the highest confidence
        # The output is a list of dicts: [{'label': 'sixties', 'score': 0.8}, ...]
        top_result = max(age_results, key=lambda x: x['score'])
        detected_age_label = top_result['label']
        
        # Reset pointer for next model
        uploaded_file.seek(0)
        
        # 3. Logic to identify Senior Citizens
        senior_categories = ["sixties", "seventies", "eighties", "nineties"]

        if detected_age_label in senior_categories:
            st.success(f"Detected Age: {detected_age_label.capitalize()} (Senior Citizen)")
    
            # Trigger emotion detection
            emotion_results = emotion_pipe(y)
            best_emotion = max(emotion_results, key=lambda x: x['score'])
            st.info(f"Detected Emotion: {best_emotion['label']}")
        else:
            st.write(f"Detected Age: {detected_age_label.capitalize()}")
            st.write("Not a senior citizen.")
