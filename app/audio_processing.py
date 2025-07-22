import whisper
import os
import streamlit as st

@st.cache_resource
def load_whisper_model(model_name="base"):
    
    st.info(f"Loading Whisper '{model_name}' model for transcription... This may take a moment.", icon="⏳")
   
    whisper_model = whisper.load_model(model_name)
    st.success(f"Whisper '{model_name}' model loaded successfully!")
    return whisper_model

whisper_model_instance = load_whisper_model("base")

def transcribe_audio(audio_path):
    
    try:
        result = whisper_model_instance.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error during Whisper transcription: {e}")
        return "" # Return empty string if transcription fails
