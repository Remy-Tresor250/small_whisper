from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio

# Load fine-tuned KinyaWhisper model and processor from Hugging Face
model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("your_audio.wav")
inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt")

# Generate prediction
predicted_ids = model.generate(inputs["input_features"])
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("🗣️ Transcription:", transcription)