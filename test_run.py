import streamlit as st
import os
import gdown
import librosa
import numpy as np
import torch
import torch.nn as nn
import re
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel

# --- 1. MODEL DEFINITIONS (Required for Age Model) ---
class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)

class AgeGenderModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender

# --- 2. SETUP & DOWNLOAD ---
@st.cache_resource
def setup_models():
    
    folder_id = '1AqPmnnIexWmEcp_IWBR1sCI4sGatxn-8'
    if not os.path.exists("./Models"):
        gdown.download_folder(id=folder_id, output='./Models', quiet=False)
    
    # Age Model Loading (Custom Architecture)
    model_path = "./Models/age_model" 
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    age_model = AgeGenderModel.from_pretrained(model_path)
    age_model.eval()
    
    # Gender & Emotion (Using standard pipelines)
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
    
    # 1. Gender Prediction
    gender_results = gender_pipe(y)
    gender = gender_results[0]['label'].lower()
    
    if 'female' in gender:
        st.error("Upload male voice.")
    else:
        # 2. Age Prediction (Using the Custom Model)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            _, logits_age, _ = age_model(inputs.input_values)
        
        age = int(logits_age.item() * 100)
        
        # 3. Conditional Logic
        if age == 0:
            st.warning("Could not clearly detect age.")
        elif age > 60:
            emotion_results = emotion_pipe(y)
            emotion = emotion_results[0]['label']
            st.success(f"Detected Age: {age} (Senior Citizen)")
            st.info(f"Detected Emotion: {emotion}")
        else:
            st.success(f"Detected Age: {age}")
