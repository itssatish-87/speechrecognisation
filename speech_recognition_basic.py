import speech_recognition as sr

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)  # read entire file

    try:
        text = recognizer.recognize_google(audio)
        return f"Transcription: {text}"
    except sr.UnknownValueError:
        return "Could not understand the audio"
    except sr.RequestError as e:
        return f"API Error: {e}"

# Example usage
print(transcribe_audio("hello.wav"))
