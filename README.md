## ASR_Group12
A speech to text model that employs a fine-tuned Whisper model as well as a fine tuned BERT model

# Authors
- Issac Baah
- Emmanuel Nhyira Freduah-Agyemang
- Paa Kwesi Thompson Jnr.

# Summary
This project aimed to develop a speech recognition system for Asante Twi, a native Ghanaian language, using OpenAI Whisper fine-tuned on Asante Twi datasets. Subsequently, we attempted to use a BERT-based language model to refine the transcriptions through grammar error correction. The main objectives were to train an automatic speech recognition (ASR) model, evaluate its performance using held-out and newly compiled test datasets, and deploy the model with an accessible API.

We chose Whisper as our baseline model because initial testing with Twi audio recordings showed it could capture phonemes accurately but transcribe them in different languages. This observation led us to believe that fine-tuning Whisper with appropriate Twi datasets could help the model learn better representations of the Twi language and yield more accurate transcriptions.

To refine ASR predictions, we attempted to use a pre-trained BERT model. Predicted transcriptions from Whisper were fed into BERT for contextual corrections, particularly addressing grammatical errors and code-mixed ambiguities. This approach did not fully materialize due to time and resource constraints.
