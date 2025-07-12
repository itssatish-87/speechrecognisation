
import speech_recognition as sr

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        print("Transcription:", text)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Request failed: {e}")

# Example usage
if __name__ == "__main__":
    transcribe_audio("hello.wav")
