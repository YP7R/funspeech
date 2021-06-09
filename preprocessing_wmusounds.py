import os
import shutil
import glob
import pandas as pd
from pydub import AudioSegment

# Dossier d'entrée et de sorties
input_directory = '.\\dataset\\raw\\sound-timedata\\'
output_directory = '.\\dataset\\processed\\wmusounds\\'
dat_file = f"{input_directory}timedata.dat"

if not os.path.exists(input_directory):
    exit(0)

if not os.path.exists(dat_file):
    exit(0)

# Creation du dossier de sortie
shutil.rmtree(output_directory, ignore_errors=True)
os.makedirs(output_directory)

wav_files = glob.glob(input_directory + '**/*.wav', recursive=True)

path_hashmap = {'m': 'men', 'w': 'women', 'b': 'kids', 'g': 'kids'}
df = pd.read_csv(dat_file, sep=";", comment="#" or ' ')

print(wav_files)

dataframe = []
for index, row in df.iterrows():
    category = path_hashmap[row['File'][0]]
    filename = f"{input_directory}{category}\\{row['File']}.wav"
    output_filename = f"{output_directory}{row['File']}.wav"

    # Extraction de phoneme
    audio = AudioSegment.from_file(file=filename, format="wav", frame_rate=16000)
    audio_clipped = audio[int(row['Start']):row['End']]
    audio_clipped.export(output_filename, format="wav")

    # Informations nécessaires
    sound_id = row['File'][-2:]
    length = len(audio_clipped)
    base_file = row['File'][:-2]
    dataframe.append((sound_id, category, base_file, length))
    print(sound_id, category, base_file, length)

df = pd.DataFrame.from_records(dataframe, columns=['sound_id', 'category', 'base_file', 'length'])
df.to_csv(f"{output_directory}wmusounds.csv", sep=",", index=False)
