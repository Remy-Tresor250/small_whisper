import os
import torch
import torchaudio
import numpy as np
import gradio as gr
import difflib
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from TTS.api import TTS
import tempfile
import time
import shutil
import soundfile as sf
from pathlib import Path

# Add multiple TTS classes to safe globals for PyTorch 2.6 compatibility
try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig])
    # Add any other XTTS-related classes that might be needed
    try:
        from TTS.tts.models.xtts import Xtts
        torch.serialization.add_safe_globals([Xtts])
    except ImportError:
        pass
except ImportError:
    print("Warning: Could not import XttsConfig. Will attempt alternative loading method.")

class KinyarwandaQASystem:
    def __init__(self):
        print("Initializing Kinyarwanda Q&A System...")
        
        # Create directories if they don't exist
        os.makedirs("audio_samples", exist_ok=True)
        os.makedirs("transcriptions", exist_ok=True)
        os.makedirs("answers", exist_ok=True)
        
        # Load ASR model
        self.asr_model = WhisperForConditionalGeneration.from_pretrained("benax-rw/KinyaWhisper")
        self.processor = WhisperProcessor.from_pretrained("benax-rw/KinyaWhisper")
        
        # Configure ASR model
        if hasattr(self.asr_model.generation_config, "forced_decoder_ids"):
            self.asr_model.generation_config.forced_decoder_ids = None
        
        # Initialize TTS with robust error handling
        self.init_tts()
        
        # Define Q&A dictionary
        self.qa_pairs = {
            "Rwanda Coding Academy iherereye he?": "Iherereye mu Karere ka Nyabihu, mu Ntara y'Iburengerazuba.",
            "Umurwa mukuru w'u Rwanda ni uwuhe?": "Ni Kigali.",
            "Abana bakunda gukina?": "Yego! Akenshi bikinira umupira.",
            "Imbuto zigira akahe kamaro?": "Zongera intungamubiri.",
            "Ntuye mu gihugu cy'u Rwanda": "Waba utuye muri Kigali?",
            "Indirimbo y'igihugu": "Indirimbo y'igihugu yitwa Rwanda Nziza.",
            "Indimi za Leta mu Rwanda ni izihe?": "Indimi za Leta mu Rwanda ni Ikinyarwanda, Icyongereza n'Igifaransa.",
            "Igihugu cy'u Rwanda kigizwe n'ibihe bice?": "U Rwanda rugizwe n'intara enye n'umujyi wa Kigali."
        }
        
        # Save QA pairs to JSON
        with open("qa_pairs.json", "w", encoding="utf-8") as f:
            json.dump(self.qa_pairs, f, ensure_ascii=False, indent=4)
        
        print("System initialized and ready!")

    def init_tts(self):
        """Initialize TTS with robust error handling"""
        print("Loading TTS model...")
        try:
            # First try: Explicitly set weights_only=False to address PyTorch 2.6 change
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
            print("TTS model loaded successfully!")
        except Exception as e:
            print(f"Error loading TTS with standard approach: {str(e)}")
            try:
                # Try setting torch.serialization._UNSAFE_WEIGHTS_ONLY_DEFAULT = False
                # This is a more direct approach to handle PyTorch 2.6 changes
                import torch.serialization
                original_value = getattr(torch.serialization, '_UNSAFE_WEIGHTS_ONLY_DEFAULT', True)
                setattr(torch.serialization, '_UNSAFE_WEIGHTS_ONLY_DEFAULT', False)
                
                print("Attempting alternative TTS loading method with _UNSAFE_WEIGHTS_ONLY_DEFAULT=False...")
                self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
                
                # Restore original value
                setattr(torch.serialization, '_UNSAFE_WEIGHTS_ONLY_DEFAULT', original_value)
                print("TTS model loaded successfully with alternative method!")
            except Exception as inner_e:
                print(f"Failed to load TTS model with alternative approach: {str(inner_e)}")
                print("Will use text-only mode without speech synthesis.")
                self.tts = None

    def transcribe_audio(self, audio_path):
        """Transcribe audio file using KinyaWhisper"""
        print(f"Transcribing audio file: {audio_path}")
        
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
        
        # Process audio
        inputs = self.processor(waveform_np, sampling_rate=sample_rate, return_tensors="pt")
        
        # Add attention mask if needed
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_features"][:, :, 0])
        
        # Generate prediction
        predicted_ids = self.asr_model.generate(
            inputs["input_features"],
            attention_mask=inputs.get("attention_mask", None)
        )
        
        # Decode transcription
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Save transcription
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        transcription_path = f"transcriptions/{base_filename}.txt"
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(transcription)
        
        print(f"🗣️ Transcription: {transcription}")
        return transcription

    def find_best_match(self, query):
        """Find the closest match to the query in the QA pairs"""
        if not query:
            return None
            
        # Use difflib to find closest match
        matches = difflib.get_close_matches(query, self.qa_pairs.keys(), n=1, cutoff=0.6)
        
        if matches:
            best_match = matches[0]
            print(f"Matched query: '{best_match}'")
            return self.qa_pairs[best_match], best_match
        else:
            print("No match found in the QA database")
            default_response = "Mbabarira, sinumvise neza ikibazo cyawe."  # I'm sorry, I didn't understand your question
            return default_response, None

    def text_to_speech(self, text, output_path=None):
        """Convert text to speech using Coqui TTS and return the path to the audio file"""
        if not text:
            return None
            
        print(f"Converting to speech: '{text}'")
        
        # Create a temporary file for the speech if no output path provided
        if output_path is None:
            # Create a unique filename based on timestamp
            timestamp = int(time.time())
            output_path = f"answers/answer_{timestamp}.wav"
        
        # Generate speech if TTS is available
        if self.tts:
            try:
                # Make sure reference audio file exists
                reference_wav = "speaker_reference.wav"
                if not os.path.exists(reference_wav):
                    print(f"Warning: Reference voice file {reference_wav} not found. Using default voice.")
                    # Try to use TTS without speaker reference
                    self.tts.tts_to_file(text=text, 
                                      file_path=output_path,
                                      language="rw")  # Kinyarwanda language code
                else:
                    # Use with speaker reference
                    self.tts.tts_to_file(text=text, 
                                      file_path=output_path, 
                                      speaker_wav=reference_wav,
                                      language="rw")  # Kinyarwanda language code
                
                print(f"Speech saved to {output_path}")
                return output_path
            except Exception as e:
                print(f"TTS failed: {str(e)}")
                print("Falling back to text-only response")
                
                # Create an empty audio file to maintain interface compatibility
                sr = 22050  # Standard sample rate
                empty_audio = np.zeros(sr)  # 1 second of silence
                sf.write(output_path, empty_audio, sr)
                
                return output_path
        else:
            print("TTS not available, returning empty audio")
            # Create an empty audio file to maintain interface compatibility
            sr = 22050  # Standard sample rate
            empty_audio = np.zeros(sr)  # 1 second of silence
            sf.write(output_path, empty_audio, sr)
            
            return output_path

    def process_audio(self, audio_path):
        """Process audio file and return transcription, answer, and audio response"""
        # Save a copy of the input audio to the samples directory
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        sample_path = f"audio_samples/{base_filename}.wav"
        
        # Convert to WAV format if needed and save
        if audio_path != sample_path:
            data, samplerate = sf.read(audio_path)
            sf.write(sample_path, data, samplerate)
        
        # Transcribe audio to text
        transcription = self.transcribe_audio(audio_path)
        
        # Find best matching answer
        answer, matched_question = self.find_best_match(transcription)
        
        # Save the answer
        answer_text_path = f"answers/{base_filename}_answer.txt"
        with open(answer_text_path, "w", encoding="utf-8") as f:
            f.write(f"Question: {transcription}\n")
            f.write(f"Matched Question: {matched_question if matched_question else 'No match'}\n")
            f.write(f"Answer: {answer}")
        
        # Convert answer to speech
        answer_audio_path = f"answers/{base_filename}_answer.wav"
        self.text_to_speech(answer, answer_audio_path)
        
        return transcription, answer, answer_audio_path, matched_question

# Function for Gradio interface
def process_audio_file(audio_file):
    # Create temporary file for the audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        temp_audio_path = temp_audio.name
        # Copy audio data to temporary file
        shutil.copyfile(audio_file, temp_audio_path)
    
    # Process the audio
    transcription, answer, answer_audio, matched_question = qa_system.process_audio(temp_audio_path)
    
    # Return results
    matched_info = f"Matched to: '{matched_question}'" if matched_question else "No close match found"
    return transcription, matched_info, answer, answer_audio

def record_and_process(audio):
    # Save the recorded audio
    timestamp = int(time.time())
    audio_path = f"audio_samples/recording_{timestamp}.wav"
    sf.write(audio_path, audio[1], audio[0])
    
    # Process the audio
    transcription, answer, answer_audio, matched_question = qa_system.process_audio(audio_path)
    
    # Return results
    matched_info = f"Matched to: '{matched_question}'" if matched_question else "No close match found"
    return transcription, matched_info, answer, answer_audio

# Initialize QA system
qa_system = KinyarwandaQASystem()

# Create Gradio interface
with gr.Blocks(title="Kinyarwanda Speech Q&A System") as demo:
    gr.Markdown("# Kinyarwanda Speech Q&A System")
    gr.Markdown("""
    This system transcribes Kinyarwanda speech, matches it to a question in the database,
    and responds with a spoken answer. You can either upload an audio file or record directly.
    """)
    
    with gr.Tab("Upload Audio"):
        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File (Kinyarwanda)")
        
        submit_btn = gr.Button("Process Audio")
        
        with gr.Row():
            transcription_output = gr.Textbox(label="Transcription")
            matched_question_output = gr.Textbox(label="Matched Question")
        
        with gr.Row():
            answer_text_output = gr.Textbox(label="Answer Text")
            answer_audio_output = gr.Audio(label="Answer Audio")
        
        submit_btn.click(
            fn=process_audio_file,
            inputs=[audio_input],
            outputs=[transcription_output, matched_question_output, answer_text_output, answer_audio_output]
        )
    
    with gr.Tab("Record Audio"):
        with gr.Row():
            # Fixed: Use microphone=True instead of source="microphone"
            audio_recorder = gr.Audio(sources=["microphone"], type="numpy", label="Record Audio")
        
        record_btn = gr.Button("Process Recording")
        
        with gr.Row():
            rec_transcription_output = gr.Textbox(label="Transcription")
            rec_matched_question_output = gr.Textbox(label="Matched Question")
        
        with gr.Row():
            rec_answer_text_output = gr.Textbox(label="Answer Text")
            rec_answer_audio_output = gr.Audio(label="Answer Audio")
        
        record_btn.click(
            fn=record_and_process,
            inputs=[audio_recorder],
            outputs=[rec_transcription_output, rec_matched_question_output, rec_answer_text_output, rec_answer_audio_output]
        )
    
    gr.Markdown("""
    ## Available Questions
    
    The system can answer questions about:
    - Location of Rwanda Coding Academy
    - Capital city of Rwanda
    - Whether children like to play
    - Benefits of fruits
    - Living in Rwanda
    - Rwanda's national anthem
    - Official languages of Rwanda
    - Administrative divisions of Rwanda
    
    Feel free to ask these questions in Kinyarwanda!
    """)

if __name__ == "__main__":
    demo.launch()