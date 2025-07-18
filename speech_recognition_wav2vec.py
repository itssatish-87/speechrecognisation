import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def transcribe_wav2vec(audio_path):
    # Check if the file exists
    if not os.path.exists(audio_path):
        return "Error: Audio file not found."

    # Load audio using torchaudio
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        print(f"Sample Rate: {sample_rate}, Channels: {waveform.shape[0]}")
    except Exception as e:
        return f"Error loading audio: {e}"

    # Convert stereo to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16000 Hz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Load pre-trained model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Prepare inputs
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)

    # Run inference
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode output
    transcription = processor.batch_decode(predicted_ids)[0]

    return transcription.lower()


# === RUN IT ===
if __name__ == "__main__":
    audio_file = "D:/python/speech_to_text_project/hello.wav"
  # ✅ Use your actual path if needed
    result = transcribe_wav2vec(audio_file)
    print("Transcribed Text:", result)
