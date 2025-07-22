from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import streamlit as st 
import torch 

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import streamlit as st 
import torch
import time

@st.cache_resource
def load_emotion_model(model_name="cardiffnlp/twitter-roberta-base-emotion"):
    notify = st.empty()
    notify.info(f"Loading emotion detection model '{model_name}'...", icon="⏳")
    emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
    emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model_labels = [emotion_model.config.id2label[i] for i in range(len(emotion_model.config.id2label))]
    notify.success(f"Emotion model '{model_name}' loaded successfully! Detected labels: {', '.join(model_labels)}")
    time.sleep(2)
    notify.empty()
    return emotion_tokenizer, emotion_model, model_labels

emotion_tokenizer_instance, emotion_model_instance, emotion_labels = load_emotion_model("cardiffnlp/twitter-roberta-base-emotion")

def get_emotion(text):
    if not text:
        notify = st.empty()
        notify.warning("No text provided for emotion analysis.")
        time.sleep(2)
        notify.empty()
        return {}

    try:
        inputs = emotion_tokenizer_instance(text, return_tensors="pt", truncation=True, padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            emotion_model_instance.to('cuda')
        outputs = emotion_model_instance(**inputs)
        scores = softmax(outputs.logits.detach().cpu().numpy()[0])
        return dict(zip(emotion_labels, map(float, scores)))
    except Exception as e:
        notify = st.empty()
        notify.error(f"Error during emotion analysis: {e}")

# @st.cache_resource caches the tokenizer and model.
# This ensures they are loaded only once when the Streamlit app starts.
# @st.cache_resource
# def load_emotion_model(model_name="cardiffnlp/twitter-roberta-base-emotion"):
    
#     # st.info(f"Loading emotion detection model '{model_name}'...", icon="⏳")
#     emotion_tokenizer = AutoTokenizer.from_pretrained(model_name)
#     emotion_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
#     model_labels = [emotion_model.config.id2label[i] for i in range(len(emotion_model.config.id2label))]
    
#     # st.success(f"Emotion model '{model_name}' loaded successfully! Detected labels: {', '.join(model_labels)}")
#     return emotion_tokenizer, emotion_model, model_labels

# # Load the emotion models and labels globally within this module using the cached function.
# emotion_tokenizer_instance, emotion_model_instance, emotion_labels = load_emotion_model("cardiffnlp/twitter-roberta-base-emotion")

# def get_emotion(text):
    
#     if not text:
#         st.warning("No text provided for emotion analysis.")
#         return {}

#     try:
#         inputs = emotion_tokenizer_instance(text, return_tensors="pt", truncation=True, padding=True)
        
#         if torch.cuda.is_available():
#             inputs = {k: v.to('cuda') for k, v in inputs.items()}
#             emotion_model_instance.to('cuda') # Ensure model is also on GPU

#         outputs = emotion_model_instance(**inputs)
#         scores = softmax(outputs.logits.detach().cpu().numpy()[0]) # Move to CPU before numpy()
        
#         # Use the dynamically loaded labels
#         return dict(zip(emotion_labels, map(float, scores)))
    
#     except Exception as e:
#         st.error(f"Error during emotion analysis: {e}")
#         return {} # Return empty dict