import streamlit as st
from transformers import pipeline
import torchaudio
from torchaudio.transforms import Resample
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, ClientSettings
import numpy as np
import av

# Initialize the Whisper pipeline
st.title("Whisper Twi Speech Recognition")
st.markdown("### Realtime demo for Twi speech recognition using a fine-tuned Whisper model.")

# Load your Whisper model
@st.cache_resource
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

# Class to handle audio recording
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame: av.AudioFrame) -> None:
        self.frames.append(frame.to_ndarray().flatten())

    def save_audio(self, filepath: str):
        audio = np.concatenate(self.frames, axis=0).astype(np.float32)
        sample_rate = 48000  # Default sample rate from WebRTC
        torchaudio.save(filepath, torch.tensor([audio]), sample_rate)
        self.frames = []  # Clear frames after saving

# File uploader for audio input
uploaded_audio = st.file_uploader("Upload an audio file:", type=["wav", "mp3", "ogg", "flac"])

# Transcription button for uploaded audio
if uploaded_audio is not None:
    # Save the uploaded file to a temporary directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(uploaded_audio.read())
        temp_file_path = temp_file.name

    # Resample the audio to 16 kHz
    try:
        waveform, sample_rate = resample_audio(temp_file_path)
        resampled_path = temp_file_path.replace(".wav", "_resampled.wav")
        torchaudio.save(resampled_path, waveform, sample_rate)

        st.audio(resampled_path, format="audio/wav")

        with st.spinner("Transcribing audio..."):
            text = pipe(resampled_path)["text"]

        st.write("### Transcription:")
        st.write(text)

        os.remove(temp_file_path)
        os.remove(resampled_path)
    except Exception as e:
        st.error(f"Error processing the audio: {e}")

# Audio recording section
st.markdown("### Record Audio")
webrtc_ctx = webrtc_streamer(
    key="audio-recorder",
    client_settings=ClientSettings(
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"audio": True, "video": False},
    ),
    audio_processor_factory=AudioProcessor,
)

if webrtc_ctx and webrtc_ctx.audio_processor:
    audio_processor = webrtc_ctx.audio_processor
    if st.button("Save Recording"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
        audio_processor.save_audio(temp_file_path)

        # Resample the recorded audio to 16 kHz
        try:
            waveform, sample_rate = resample_audio(temp_file_path)
            resampled_path = temp_file_path.replace(".wav", "_resampled.wav")
            torchaudio.save(resampled_path, waveform, sample_rate)

            st.audio(resampled_path, format="audio/wav")

            with st.spinner("Transcribing recorded audio..."):
                text = pipe(resampled_path)["text"]

            st.write("### Transcription:")
            st.write(text)

            os.remove(temp_file_path)
            os.remove(resampled_path)
        except Exception as e:
            st.error(f"Error processing the recorded audio: {e}")
