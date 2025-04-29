from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch

model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")

waveform, sample_rate = torchaudio.load("ikigori.mp3")

# Convert to mono if stereo
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Resample to 16kHz (the rate expected by Whisper)
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    sample_rate = 16000

# Convert to numpy array (as expected by the processor)
waveform_np = waveform.squeeze().numpy()

inputs = processor(waveform_np, sampling_rate=sample_rate, return_tensors="pt")

if hasattr(model.generation_config, "forced_decoder_ids"):
    model.generation_config.forced_decoder_ids = None

if "attention_mask" not in inputs:
    inputs["attention_mask"] = torch.ones_like(inputs["input_features"][:, :, 0])

predicted_ids = model.generate(
    inputs["input_features"],
    attention_mask=inputs.get("attention_mask", None)
)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

print("🗣️ Transcription:", transcription)