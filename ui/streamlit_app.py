import streamlit as st
import os
import tempfile
from pydub import AudioSegment # For robust audio format handling
from app.audio_processing import transcribe_audio
from app.emotion_text import get_emotion, emotion_labels 

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="EchoMatch: Emotion from Voice",
    page_icon="🎧",
    layout="centered" # or "wide" depending on preference
)

# --- App Title and Description ---
st.title("🎧 EchoMatch: Emotion from Voice")
st.markdown("Upload an audio file (MP3, WAV, M4A, FLAC, OGG) to get a text transcription and emotional analysis.")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Upload an audio file (max 15MB recommended)",
    type=["wav", "mp3", "m4a", "flac", "ogg"] # Supported audio types
)

# --- Main Processing Logic ---
if uploaded_file is None:
    st.info("Please upload an audio file to get started.")
else:
    # Check file size before processing
    if uploaded_file.size > 15 * 1024 * 1024: # 15 MB limit
        st.warning("File size exceeds the recommended 15MB limit. Processing may be slow or fail for larger files.")
        
    # Display the uploaded audio file
    st.audio(uploaded_file, format=uploaded_file.type)

    temp_audio_path = None 

    try:
        # Create a temporary file to save the uploaded audio. This is essential because Whisper and pydub need a file path.
        # Adding suffix based on original file name helps pydub/ffmpeg identify format.
        
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_audio_path = tmp_file.name # Store path for cleanup

        # Robust audio format conversion to WAV using pydub This helps ensure Whisper and other tools can process it reliably.
        if not temp_audio_path.lower().endswith(".wav"):
            st.info("Converting audio to WAV format for reliable processing...")
            try:
                audio = AudioSegment.from_file(temp_audio_path)
                wav_audio_path = temp_audio_path.replace(suffix, ".wav")
                audio.export(wav_audio_path, format="wav")
                os.remove(temp_audio_path) # Clean up original temp file
                temp_audio_path = wav_audio_path # Update path to the WAV file
            except Exception as e:
                st.error(f"Could not convert audio to WAV. Please check your ffmpeg installation or try another format. Error: {e}")
                st.stop() # Stop further processing if conversion fails

        # --- Transcribe Audio ---
        st.subheader("📜 Transcription")
        with st.spinner("Transcribing audio... This might take a while for longer files."):
            transcribed_text = transcribe_audio(temp_audio_path)

        if transcribed_text:
            st.success("Transcription complete!")
            st.markdown(f"**Transcribed Text:**\n\n```\n{transcribed_text}\n```")
        else:
            st.warning("Could not generate transcription for this audio file.")

        # --- Emotion Analysis ---
        st.subheader("😄 Detected Emotions")
        if transcribed_text: # Only run emotion analysis if transcription was successful
            with st.spinner("Analyzing emotions..."):
                emotion_scores = get_emotion(transcribed_text)
            
            if emotion_scores:
                st.markdown("Here are the detected emotion scores:")
                
                # Display scores in a nicer format (e.g., using columns or just markdown)
                cols = st.columns(len(emotion_labels))
                for i, label in enumerate(emotion_labels):
                    with cols[i]:
                        score_percent = f"{emotion_scores.get(label, 0.0) * 100:.2f}%"
                        st.metric(label.capitalize(), score_percent)
                
                # Optionally, display raw JSON for more detail
                if st.checkbox("Show raw emotion scores (JSON)"):
                    st.json(emotion_scores)
            else:
                st.info("Could not detect emotions from the transcription (it might be empty or too short).")
        else:
            st.info("Emotion analysis skipped as transcription was not available.")

    except Exception as e:
        st.error(f"An unexpected error occurred during processing: {e}")
        st.exception(e) # Display full traceback for debugging (remove or comment in production)
    finally:
        # Clean up the temporary audio file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            # st.info(f"Cleaned up temporary file: {temp_audio_path}") # For debugging

# --- Footer ---
st.markdown("---")
st.markdown("EchoMatch: Emotion from Voice. Powered by Whisper ASR and Hugging Face Transformers.")