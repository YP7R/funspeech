import glob

# Fichiers de référence
csv_files = glob.glob(".\\dataset\\processed\\"+"**/*.csv", recursive=True)
print(csv_files)
