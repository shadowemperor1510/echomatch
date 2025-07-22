import whisper
import os
import streamlit as st
import time

@st.cache_resource
def load_whisper_model(model_name="base"):
    # Create a placeholder for notifications
    notify = st.empty()
    notify.info(f"Loading Whisper '{model_name}' model for transcription... This may take a moment.", icon="⏳")
    whisper_model = whisper.load_model(model_name)
    notify.success(f"Whisper '{model_name}' model loaded successfully!")
    time.sleep(2)  # Show success for 2 seconds
    notify.empty() # Clear the notification
    return whisper_model

whisper_model_instance = load_whisper_model("base")

def transcribe_audio(audio_path):
    try:
        result = whisper_model_instance.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"Error during Whisper transcription: {e}")
        return "" # Return empty string if transcription fails
    

# @st.cache_resource
# def load_whisper_model(model_name="base"):
    
#     status_placeholder = st.empty()

#     with status_placeholder.status(f"Loading Whisper '{model_name}' model...", expanded=True) as status:
#         st.write("Initializing model download and setup...")
#         try:
#             whisper_model = whisper.load_model(model_name)
#             status.update(label=f"Whisper '{model_name}' model loaded successfully!", state="complete", expanded=False)
#             # st.session_state is available even in cached functions if they are part of the app's scope
#             st.session_state['whisper_model_loaded'] = True
#             return whisper_model
#         except Exception as e:
#             status.update(label=f"Error loading Whisper '{model_name}' model: {e}", state="error")
#             st.error("Model loading failed! Please check your internet connection or try again.")
#             st.session_state['whisper_model_loaded'] = False # Mark as failed
#             raise # Re-raise the exception to propagate it

# # Load the desired Whisper model globally within this module
# whisper_model_instance = load_whisper_model("base")

# def transcribe_audio(audio_path):
#     """
#     Transcribes the audio file.
#     """
#     try:
#         result = whisper_model_instance.transcribe(audio_path)
#         return result["text"]
#     except Exception as e:
#         st.error(f"Error during Whisper transcription: {e}")
#         return ""



# @st.cache_resource
# def load_whisper_model(model_name="base"):
    
#     # st.info(f"Loading Whisper '{model_name}' model for transcription... This may take a moment.", icon="⏳")
   
#     whisper_model = whisper.load_model(model_name)
#     # st.success(f"Whisper '{model_name}' model loaded successfully!")
#     return whisper_model

# whisper_model_instance = load_whisper_model("base")

# def transcribe_audio(audio_path):
    
#     try:
#         result = whisper_model_instance.transcribe(audio_path)
#         return result["text"]
#     except Exception as e:
#         st.error(f"Error during Whisper transcription: {e}")
#         return "" # Return empty string if transcription fails


# @st.cache_resource
# def load_whisper_model(model_name="base"):

#     with st.status(f"Loading Whisper '{model_name}' model...", expanded=True) as status:
#         st.write("Initializing model download and setup...")
#         try:
#             whisper_model = whisper.load_model(model_name)
#             # Update the status once complete
#             status.update(label=f"Whisper '{model_name}' model loaded successfully!", state="complete", expanded=False)
#             return whisper_model
#         except Exception as e:
#             # Update the status on error
#             status.update(label=f"Error loading Whisper '{model_name}' model: {e}", state="error")
#             st.error("Model loading failed! Please check your internet connection or try again.")
#             raise # Re-raise the exception to propagate it
            
# # Load the desired Whisper model globally within this module
# whisper_model_instance = load_whisper_model("base")

# def transcribe_audio(audio_path):

#     try:
#         result = whisper_model_instance.transcribe(audio_path)
#         return result["text"]
#     except Exception as e:
#         st.error(f"Error during Whisper transcription: {e}")
#         return ""