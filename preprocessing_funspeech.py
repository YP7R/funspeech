import os
import shutil
import glob
import pandas as pd
from pydub import AudioSegment, silence

# Dossier d'entrée et de sorties
input_directory = '.\\dataset\\raw\\4voices-funspeech\\'
output_directory = '.\\dataset\\processed\\funspeech\\'

if not os.path.exists(input_directory):
    exit(0)

# Creation du dossier de sortie
shutil.rmtree(output_directory, ignore_errors=True)
os.makedirs(output_directory)

# Recuperer les fichiers wav
wav_files = glob.glob(input_directory + '**/*.wav', recursive=True)
if not wav_files:
    exit(0)

category = "m"
dataframe = []
for file in wav_files:
    # Nom du son
    sound_id = os.path.basename(os.path.dirname(file))

    # Nom de l'utilisateur et id echantillon
    base_file, ext = os.path.splitext(os.path.basename(file))
    user_name, info = base_file.split("_")

    # Fichier final
    new_filename = f"{category}{user_name}-{info}{sound_id}{ext}"
    output_filename = f"{output_directory}{new_filename}"
    audio = AudioSegment.from_file(file, format='wav', frame_rate=16000)

    silences = silence.detect_silence(audio, min_silence_len=1000, silence_thresh=audio.dBFS - 16)
    # https://stackoverflow.com/questions/40896370/detecting-the-index-of-silence-from-a-given-audio-file-using-python
    # https://stackoverflow.com/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
    # https://programtalk.com/python-examples/pydub.silence.detect_silence/
    # https://fr.coredump.biz/questions/29547218/remove-silence-at-the-beginning-and-at-the-end-of-wave-files-with-pydub
    # https://gist.github.com/sotelo/be57571a1d582d44f3896710b56bc60d

    # sil = [((start / 1000), (stop / 1000)) for start, stop in silences]  # in sec
    if len(silences) > 2:
        print(f"{file} doit être vérifié manuellement")
        continue

    start_audio, end_audio = 0, 0
    start_audio, end_audio = silences[0][1], silences[1][0]

    audio_clipped = audio[start_audio:end_audio]
    audio_clipped.export(output_filename, format="wav")

    # Informations nécessaires
    length = len(audio_clipped)
    dataframe.append((sound_id, category, base_file, length))
    print(sound_id, category, base_file, length)

df = pd.DataFrame.from_records(dataframe, columns=['sound_id', 'category', 'base_file', 'length'])
df.to_csv(f"{output_directory}_funspeech.csv", sep=",", index=False)
