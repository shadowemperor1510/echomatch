import os
import streamlit as st
import tempfile
from pydub import AudioSegment
from app.audio_processing import transcribe_audio
from app.emotion_text import get_emotion, emotion_labels

# Configure Streamlit app
st.set_page_config(
    page_title="EchoMatch: Emotion from Voice",
    page_icon="🎧",
    layout= "centered"
)

st.title("🎧 EchoMatch: Emotion from Voice")
st.markdown("Upload an audio file (MP3, WAV, M4A, FLAC, OGG) to get a text transcription and emotional analysis.")

uploaded_file = st.file_uploader(
    "Upload an audio file (max 15MB recommended)",
    type=["wav", "mp3", "m4a", "flac", "ogg"]
)

if uploaded_file is None:
    st.info("Please upload an audio file to get started.")
else:
    if uploaded_file.size > 15 * 1024 * 1024:
        st.warning("File size exceeds the recommended 15MB limit. Processing may be slow or fail for larger files.")
    
    st.audio(uploaded_file, format=uploaded_file.type)

    temp_audio_path = None

    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_audio_path = tmp_file.name

        if not temp_audio_path.lower().endswith(".wav"):
            st.info("Converting audio to WAV format for reliable processing...")
            try:
                audio = AudioSegment.from_file(temp_audio_path)
                wav_audio_path = temp_audio_path.replace(suffix, ".wav")
                audio.export(wav_audio_path, format="wav")
                os.remove(temp_audio_path)
                temp_audio_path = wav_audio_path
            except Exception as e:
                st.error(f"Could not convert audio to WAV. Please ensure ffmpeg is installed and correctly configured in your system's PATH. Error: {e}")
                st.stop()

        st.subheader("📜 Transcription")
        with st.spinner("Transcribing audio... This might take a while for longer files."):
            transcribed_text = transcribe_audio(temp_audio_path)

        if transcribed_text:
            st.success("Transcription complete!")
            st.markdown(f"**Transcribed Text:**\n\n```\n{transcribed_text}\n```")
        else:
            st.warning("Could not generate transcription for this audio file.")

        st.subheader("😄 Detected Emotions")
        if transcribed_text:
            with st.spinner("Analyzing emotions..."):
                emotion_scores = get_emotion(transcribed_text)
            
            if emotion_scores:
                st.markdown("Here are the detected emotion scores:")
                
                cols = st.columns(len(emotion_labels))
                for i, label in enumerate(emotion_labels):
                    with cols[i]:
                        score_percent = f"{emotion_scores.get(label, 0.0) * 100:.2f}%"
                        st.metric(label.capitalize(), score_percent)
                
                if st.checkbox("Show raw emotion scores (JSON)"):
                    st.json(emotion_scores)
            else:
                st.info("Could not detect emotions from the transcription (it might be empty or too short).")
        else:
            st.info("Emotion analysis skipped as transcription was not available.")

    except Exception as e:
        st.error(f"An unexpected error occurred during processing: {e}")
        st.exception(e)
    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

st.markdown("---")
st.markdown("EchoMatch: Emotion from Voice. Powered by Whisper ASR and Hugging Face Transformers.")