import os
import torch
import torchaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from TTS.api import TTS
import difflib

def load_whisper_model():
    """Load KinyaWhisper model and processor"""
    print("Loading KinyaWhisper model...")
    model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
    processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")
    return model, processor

def load_tts_model():
    """Load TTS model"""
    print("Loading TTS model...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())
    return tts

def preprocess_audio(audio_path):
    """Preprocess audio for Whisper model"""
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    
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
    
    return waveform_np, sample_rate

def transcribe_audio(model, processor, audio_path):
    """Transcribe audio using KinyaWhisper"""
    # Preprocess audio
    waveform_np, sample_rate = preprocess_audio(audio_path)
    
    # Process audio
    inputs = processor(waveform_np, sampling_rate=sample_rate, return_tensors="pt")
    
    # Clear forced_decoder_ids from generation config
    if hasattr(model.generation_config, "forced_decoder_ids"):
        model.generation_config.forced_decoder_ids = None
    
    # Add attention mask if needed
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_features"][:, :, 0])
    
    # Generate prediction
    predicted_ids = model.generate(
        inputs["input_features"],
        attention_mask=inputs.get("attention_mask", None)
    )
    
    # Decode transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    return transcription

def find_best_match(transcription, qa_dict, threshold=0.6):
    """Find the best matching question using fuzzy matching"""
    if not transcription:
        return None, None
    
    # Use difflib to find the best match
    matches = difflib.get_close_matches(transcription, qa_dict.keys(), n=1, cutoff=threshold)
    
    if matches:
        best_match = matches[0]
        return best_match, qa_dict[best_match]
    
    return None, None

def generate_speech(tts, text, output_path, reference_audio):
    """Generate speech from text using TTS"""
    if not text:
        return None
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate speech
    tts.tts_to_file(
        text=text,
        file_path=output_path,
        speaker_wav=reference_audio,
        language="multilingual"
    )
    
    return output_path

def record_and_save_audio(audio_data, sample_rate, output_path):
    """Save audio data to a file"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to PyTorch tensor if necessary
    if isinstance(audio_data, np.ndarray):
        audio_tensor = torch.from_numpy(audio_data).float()
    else:
        audio_tensor = audio_data
    
    # Ensure audio is 2D with shape [channels, samples]
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Save audio
    torchaudio.save(output_path, audio_tensor, sample_rate)
    
    return output_path

def create_sample_transcriptions(model, processor, audio_folder, transcription_folder):
    """Create transcriptions for sample audio files"""
    # Ensure transcription directory exists
    os.makedirs(transcription_folder, exist_ok=True)
    
    # Get all audio files in the folder
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith(('.wav', '.mp3'))]
    
    for audio_file in audio_files:
        # Full path to audio file
        audio_path = os.path.join(audio_folder, audio_file)
        
        # Transcribe audio
        transcription = transcribe_audio(model, processor, audio_path)
        
        # Save transcription to file
        base_name = os.path.splitext(audio_file)[0]
        transcription_path = os.path.join(transcription_folder, f"{base_name}.txt")
        
        with open(transcription_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        print(f"Transcribed {audio_file} -> {transcription_path}")
    
    return len(audio_files)