from pydub import AudioSegment
import librosa
import numpy as np
filepath = "./dataset/processed/funspeech/muser1-093623a.wav"

audio = AudioSegment.from_file(file=filepath,frame_rate=16000,format="wav")
audio = audio.set_channels(1)
y = np.array(audio.get_array_of_samples()).astype(np.float32)
y = y / (1 << 8*2 - 1)
print(y)
signal, sr = librosa.load(filepath, sr=16000)
print(signal)