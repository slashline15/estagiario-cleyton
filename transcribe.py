from pydub import AudioSegment
import whisper

# Passo 1: Converter o arquivo MP3 para WAV, se necessário
audio_path = "./files/audio.mp3"  # Substitua pelo caminho do seu arquivo
audio = AudioSegment.from_file(audio_path)
wav_path = "./files/audio.wav"
audio.export(wav_path, format="wav")
print(f"Áudio convertido e salvo como: {wav_path}")

# Passo 2: Carregar o modelo Whisper
model = whisper.load_model("base")  # Outros modelos: "tiny", "small", "medium", "large"

# Passo 3: Transcrever o áudio
result = model.transcribe(wav_path, fp16=False)
transcription = result["text"]

# Passo 4: Exibir a transcrição
print("Transcrição do áudio:")
print(transcription)
