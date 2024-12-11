import streamlit as st
from transformers import pipeline
import torchaudio
from torchaudio.transforms import Resample
import sounddevice as sd
import wave
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

# Function to record audio
def record_audio(duration=5, sample_rate=16000, channels=1):
    st.info("Recording... Speak now!")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
    sd.wait()  # Wait until the recording is complete
    st.success("Recording finished!")
    return audio_data, sample_rate

# Save recorded audio to a temporary file
def save_audio_to_file(audio_data, sample_rate, file_path):
    with wave.open(file_path, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 2 bytes per sample
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

# File uploader and recorder options
st.markdown("#### Choose an input method:")
option = st.radio("Select an input method:", ["Upload Audio File", "Record Audio"])

if option == "Upload Audio File":
    uploaded_audio = st.file_uploader("Upload an audio file:", type=["wav", "mp3", "ogg", "flac"])
    if uploaded_audio is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_audio.read())
            temp_file_path = temp_file.name

        # Resample the audio to 16 kHz
        try:
            waveform, sample_rate = resample_audio(temp_file_path)
            resampled_path = temp_file_path.replace(".wav", "_resampled.wav")
            torchaudio.save(resampled_path, waveform, sample_rate)

            # Display the resampled audio
            st.audio(resampled_path, format="audio/wav")

            # Transcribe the audio
            with st.spinner("Transcribing audio..."):
                text = pipe(resampled_path)["text"]

            st.write("### Transcription:")
            st.write(text)

            # Clean up temporary files
            os.remove(temp_file_path)
            os.remove(resampled_path)
        except Exception as e:
            st.error(f"Error processing the audio: {e}")

elif option == "Record Audio":
    duration = st.slider("Select recording duration (seconds):", min_value=1, max_value=10, value=5)
    if st.button("Record"):
        try:
            audio_data, sample_rate = record_audio(duration=duration)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                save_audio_to_file(audio_data, sample_rate, temp_file.name)
                temp_file_path = temp_file.name

            # Resample the audio to 16 kHz
            waveform, sample_rate = resample_audio(temp_file_path)
            resampled_path = temp_file_path.replace(".wav", "_resampled.wav")
            torchaudio.save(resampled_path, waveform, sample_rate)

            # Display the resampled audio
            st.audio(resampled_path, format="audio/wav")

            # Transcribe the audio
            with st.spinner("Transcribing audio..."):
                text = pipe(resampled_path)["text"]

            st.write("### Transcription:")
            st.write(text)

            # Clean up temporary files
            os.remove(temp_file_path)
            os.remove(resampled_path)
        except Exception as e:
            st.error(f"Error processing the recorded audio: {e}")
