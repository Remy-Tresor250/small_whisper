# Kinyarwanda Voice Assistant

This project is a voice assistant that can understand spoken Kinyarwanda questions, match them to predefined answers, and respond with synthesized speech.

## Features

- Speech recognition for Kinyarwanda using KinyaWhisper model
- Question-answer matching using fuzzy text matching
- Text-to-speech using Coqui TTS XTTS v2
- User-friendly Gradio web interface

## Requirements

- Python 3.8+
- PyTorch
- Torchaudio
- Transformers (Hugging Face)
- Coqui TTS
- Gradio
- NumPy

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/kinyarwanda-voice-assistant.git
cd kinyarwanda-voice-assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download a reference audio file for TTS voice synthesis (any clear speech sample):
```bash
# Place a WAV file named 'reference_audio.wav' in the project root
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open the provided URL in your web browser (typically http://127.0.0.1:7860)

3. Use the microphone to record a question in Kinyarwanda or upload an audio file

4. Click "Submit" to process the audio

5. View the transcription, matched question, and answer

6. Listen to the synthesized speech response

## Project Structure

- `app.py`: Main application script with ASR, QA matching, TTS, and Gradio interface
- `output/`: Directory for storing generated audio files
- `audio_samples/`: Sample audio recordings in Kinyarwanda
- `transcriptions/`: Text transcriptions of the audio samples

## Sample Questions and Answers

The system includes the following question-answer pairs in Kinyarwanda:

1. Q: "Rwanda Coding Academy iherereye he?"
   A: "Iherereye mu Karere ka Nyabihu, mu Ntara y'Iburengerazuba."

2. Q: "Umurwa mukuru w'u Rwanda ni uwuhe?"
   A: "Ni Kigali."

3. Q: "Abana bakunda gukina?"
   A: "Yego! Akenshi bikinira umupira."

4. Q: "Imbuto zigira akahe kamaro?"
   A: "Zongera intungamubiri."

5. Q: "Ntuye mu gihugu cy'u Rwanda"
   A: "Waba utuye muri Kigali?"

6. Q: "Indirimbo y'igihugu"
   A: "Indirimbo y'igihugu yitwa Rwanda Nziza."

7. Q: "Ikirere cyacu gikorera iki?"
   A: "Ikirere cya Afurika cyacu gikorera umutungo kamere."

8. Q: "Bene Afurika bafite icyizere?"
   A: "Afurika ifite icyizere cyinshi cyo gutera imbere."

9. Q: "Ni iki gifite akamaro mu buzima?"
   A: "Ubuzima bwiza bufite akamaro mu buzima bwa muntu."

10. Q: "Ururimi ruvugwa mu Rwanda"
    A: "Mu Rwanda havugwa Ikinyarwanda, Icyongereza, Igifaransa n'Igiswahili."

## Audio Samples

The repository includes 5 sample audio files in Kinyarwanda:

1. `audio_samples/question1.wav` - Question about Rwanda Coding Academy
2. `audio_samples/question2.wav` - Question about Kigali
3. `audio_samples/question3.wav` - Question about children playing
4. `audio_samples/question4.wav` - Question about fruits
5. `audio_samples/question5.wav` - Statement about living in Rwanda

## Transcriptions

Text transcriptions of the audio samples are provided in:

1. `transcriptions/question1.txt`
2. `transcriptions/question2.txt`
3. `transcriptions/question3.txt`
4. `transcriptions/question4.txt`
5. `transcriptions/question5.txt`

## Extending the System

To add more question-answer pairs, modify the `qa_pairs` dictionary in `app.py`:

```python
qa_pairs = {
    "Your question in Kinyarwanda": "Your answer in Kinyarwanda",
    # Add more pairs here
}
```

## License

[MIT License](LICENSE)

## Acknowledgments

- [KinyaWhisper model by benax-rw](https://huggingface.co/benax-rw/KinyaWhisper)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Gradio](https://github.com/gradio-app/gradio)