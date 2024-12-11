from transformers import pipeline
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import torchaudio
from torchaudio.transforms import Resample
import tempfile

# Initialize the Whisper pipeline
@st.cache_resource
def load_model():
    return pipeline(task="automatic-speech-recognition", model="Ibaahjnr/Twi_model_v1")

pipe = load_model()

# Audio processor for recording
def save_audio(frame, filepath):
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(frame.to_ndarray().tobytes())

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_frames = []

    def recv(self, frame):
        self.audio_frames.append(frame)
        return frame

    def save_recorded_audio(self, filepath):
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"".join(self.audio_frames))

st.title("Whisper Twi Speech Recognition")

option = st.radio("Input Method", ["Upload Audio", "Record Audio"])
if option == "Upload Audio":
    uploaded_audio = st.file_uploader("Upload your audio file:", type=["wav", "mp3", "ogg", "flac"])
    if uploaded_audio:
        # Save audio temporarily and resample to 16 kHz
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(uploaded_audio.read())
            temp_audio_path = temp_audio.name

        audio, sr = torchaudio.load(temp_audio_path)
        if sr != 16000:
            resampler = Resample(sr, 16000)
            audio = resampler(audio)
        torchaudio.save(temp_audio_path, audio, 16000)

        # Transcribe the resampled audio
        with st.spinner("Transcribing..."):
            transcription = pipe(temp_audio_path)["text"]
        st.write("### Transcription:")
        st.write(transcription)

elif option == "Record Audio":
    st.write("Record your audio:")

    audio_processor = AudioProcessor()
    webrtc_ctx = webrtc_streamer(key="audio", audio_processor_factory=lambda: audio_processor)

    if st.button("Transcribe Recorded Audio"):
        if not audio_processor.audio_frames:
            st.error("No audio recorded. Please record some audio first.")
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                audio_processor.save_recorded_audio(temp_audio.name)
                temp_audio_path = temp_audio.name

            # Load and resample recorded audio
            audio, sr = torchaudio.load(temp_audio_path)
            if sr != 16000:
                resampler = Resample(sr, 16000)
                audio = resampler(audio)
            torchaudio.save(temp_audio_path, audio, 16000)

            # Transcribe the resampled audio
            with st.spinner("Transcribing..."):
                transcription = pipe(temp_audio_path)["text"]
            st.write("### Transcription:")
            st.write(transcription)
