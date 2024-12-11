import streamlit as st
from transformers import pipeline
import torchaudio
from torchaudio.transforms import Resample
import tempfile
import os

# Initialize the Whisper pipeline
st.title("Whisper Twi Speech Recognition")
st.markdown("### Realtime demo for Twi speech recognition using a fine-tuned Whisper model.")

# Load your Whisper model
@st.cache_resource  # Cache the pipeline for better performance
def load_model():
    return pipeline(task="automatic-speech-recognition", model="Ibaahjnr/Twi_model_v1")

pipe = load_model()

# Function to resample audio to 16 kHz
def resample_audio(file_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sample_rate:
        resampler = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform, target_sample_rate

# File uploader for audio input
uploaded_audio = st.file_uploader("Upload an audio file:", type=["wav", "mp3", "ogg", "flac"])

# Transcription button
if uploaded_audio is not None:
    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_audio.read())
        temp_file_path = temp_file.name

    # Resample the audio to 16 kHz
    try:
        waveform, sample_rate = resample_audio(temp_file_path)
        # Save the resampled audio back to a temporary file
        resampled_path = temp_file_path.replace(".wav", "_resampled.wav")
        torchaudio.save(resampled_path, waveform, sample_rate)

        # Display the resampled audio
        st.audio(resampled_path, format="audio/wav")

        # Transcribe the audio
        with st.spinner("Transcribing audio..."):
            text = pipe(resampled_path)["text"]

        # Display the transcription
        st.write("### Transcription:")
        st.write(text)

        # Clean up temporary files
        os.remove(temp_file_path)
        os.remove(resampled_path)
    except Exception as e:
        st.error(f"Error processing the audio: {e}")
