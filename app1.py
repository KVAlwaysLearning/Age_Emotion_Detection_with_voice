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
    age_pipe = pipeline("automatic-speech-recognition", model=f"{base_path}/age_model")
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
        # Age Detection
        age_results = age_pipe(y)

        # Access the text field
        age_text = age_results.get('text', '')
        
        # Reset pointer for next model
        uploaded_file.seek(0)
        
        # Use regex to find ANY number in the text
        import re
        match = re.search(r'\d+', age_text)
        
        if match:
            age = int(match.group())
            st.success(f"Detected Age: {age}")
        else:
            # If the model didn't find a number, it might be outputting 
            # a label like "kid" or "senior" instead of a number.
            # Let's check for those keywords if no number is found
            if "kid" in age_text.lower():
                age = 10
            elif "senior" in age_text.lower():
                age = 70
            else:
                age = 0
                st.warning(f"Could not parse age from: {age_text}")
                
        # 3. Conditional Logic
        elif age > 60:
            # Detect Emotion for Senior Citizens
            emotion_results = emotion_pipe(uploaded_file)
            emotion = emotion_results[0]['label']
            st.success(f"Detected Age: {age} (Senior Citizen)")
            st.info(f"Detected Emotion: {emotion}")
        else:
            # Standard Age detection for others
            st.success(f"Detected Age: {age}")
