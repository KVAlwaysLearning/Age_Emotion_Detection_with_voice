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

        # Define the label map for the 'minato-ryan' model
        label_map = {
            "LABEL_0": "teens",
            "LABEL_1": "twenties",
            "LABEL_2": "thirties",
            "LABEL_3": "fourties",
            "LABEL_4": "fifties",
            "LABEL_5": "sixties",
            "LABEL_6": "seventies",
            "LABEL_7": "eighties",
            "LABEL_8": "nineties"
        }

        st.write("Raw Model Output:", age_results)
        
        # Get the highest confidence result
        best_result = max(age_results, key=lambda x: x['score'])
        raw_label = best_result['label']

        
                
        # 2. Get the label with the highest confidence
        # The output is a list of dicts: [{'label': 'sixties', 'score': 0.8}, ...]
        top_result = max(age_results, key=lambda x: x['score'])
        detected_age_label = top_result['label']
        
        # Reset pointer for next model
        uploaded_file.seek(0)
        
       # Map to human-readable string
        readable_label = label_map.get(raw_label, raw_label)
        
        st.success(f"Detected Age Group: {readable_label.capitalize()}")
        
        # 3. Conditional Logic for Senior Citizens
        # Define which groups are considered "Senior"
        senior_groups = ["sixties", "seventies", "eighties", "nineties"]
        
        if readable_label in senior_groups:
            st.info("Senior citizen detected. Proceeding to emotion analysis...")
    
           # Detect Emotion
            emotion_results = emotion_pipe(y)
            best_emotion = max(emotion_results, key=lambda x: x['score'])
            emotion = best_emotion['label']
            
            st.info(f"Detected Emotion: {emotion}")
        else:
            st.write(f"The person is in their {readable_label}. No further analysis required.")
