import streamlit as st
from transformers import pipeline
import librosa
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

# Function to resample audio to 16 kHz using librosa
def resample_audio(file_path, target_sample_rate=16000):
    # Load the audio file with librosa (it automatically resamples to the target sample rate)
    waveform, sample_rate = librosa.load(file_path, sr=None)  # `sr=None` ensures original sample rate is preserved
    if sample_rate != target_sample_rate:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sample_rate)
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
        librosa.output.write_wav(resampled_path, waveform, sample_rate)

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
