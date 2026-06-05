import streamlit as st
import os
import gdown
import librosa
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, AutoModel, pipeline

# --- 1. MODEL DEFINITIONS ---
class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)

class InferenceWrapper(nn.Module):
    def __init__(self, base, age_h, gender_h):
        super().__init__()
        self.wav2vec2 = base
        self.age = age_h
        self.gender = gender_h
    
    def forward(self, input_values):
        # Extract features from the base Wav2Vec2 model
        outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs.last_hidden_state, dim=1)
        age_logits = self.age(hidden_states)
        gender_logits = torch.softmax(self.gender(hidden_states), dim=1)
        return age_logits, gender_logits

# --- 2. SETUP & DOWNLOAD ---
@st.cache_resource
def setup_models():
    folder_id = '1Vw_CRVKAlsVikX-GQB1hvaxFnPuhWBKA'
    if not os.path.exists("./Models"):
        gdown.download_folder(id=folder_id, output='./Models', quiet=False)
    
    model_path = "./Models/age_model"
    
    # Load processor and base model forced to float32
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    base_model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
    
    # Initialize and cast heads to float32
    age_head = ModelHead(base_model.config, 1).to(torch.float32)
    gender_head = ModelHead(base_model.config, 3).to(torch.float32)
    
    # Wrap and set to evaluation mode
    age_model = InferenceWrapper(base_model, age_head, gender_head)
    age_model.eval()
    
    # Load separate pipelines
    gender_pipe = pipeline("audio-classification", model="./Models/gender_model")
    emotion_pipe = pipeline("audio-classification", model="./Models/emotion_model")
    
    return processor, age_model, gender_pipe, emotion_pipe

# --- 3. STREAMLIT INTERFACE ---
st.title("Voice Age & Emotion Detector")
processor, age_model, gender_pipe, emotion_pipe = setup_models()

uploaded_file = st.file_uploader("Upload voice note", type=["wav", "mp3"])

if uploaded_file:
    st.audio(uploaded_file, format='audio/wav')
    y, sr = librosa.load(uploaded_file, sr=16000)
    
    # 1. Gender check using pipeline
    gender_results = gender_pipe(y)
    gender_label = gender_results[0]['label'].lower()
    
    if 'female' in gender_label:
        st.error("Upload a male voice note.")
    else:
        # 2. Age Prediction using Custom Model
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(torch.float32)
        
        with torch.no_grad():
            logits_age, _ = age_model(input_values)
        
        age = int(logits_age.item() * 100)
        
        # 3. Logic
        if age <= 0:
            st.warning("Could not clearly detect age.")
        elif age > 60:
            emotion_results = emotion_pipe(y)
            emotion = emotion_results[0]['label']
            st.success(f"Detected Age: {age} (Senior Citizen)")
            st.info(f"Detected Emotion: {emotion}")
        else:
            st.success(f"Detected Age: {age}")
