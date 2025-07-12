
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def transcribe_with_wav2vec(audio_path):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    speech, sr = librosa.load(audio_path, sr=16000)
    input_values = processor(speech, return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    print("Transcription:", transcription)
    return transcription

# Example usage
if __name__ == "__main__":
    transcribe_with_wav2vec("example.wav")
