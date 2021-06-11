# Funspeech, *Bachelor Final work* 

## 0. Collect dataset : Western Michigan University or FunSpeech
* https://homepages.wmich.edu/~hillenbr/voweldata.html
## 1. Preprocessing
[funspeech processing](./preprocessing_funspeech.py), [wmu processing](./preprocessing_wmusounds.py)
* coupe les silences
* extrait les phonèmes
* génère des fichiers de référence `sound_id, category, base_file(username), length` 
---
En cas de [`RunTimeWarning: Couldn't find ffmpeg or avconv`](http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/)
En cas de ..., vérifier les fichiers à la main

## 2. Extraction des caractéristiques acoustiques
[extraction features](./extraction_features.py)
* (Sample FunSpeech / wmu)
* Bande de bases
* log banc de filtres
* Mel-Frequency Cepstral Coefficients
* MW-FCC
* LFCC
* ERBFCC

## 3. Classification...
[main](./main.py), [confusion matrix](./debug/confusion_matrix.py), [votting classifier](./debug/votting_clf.py)
* Barycentre
* KNN
* SVM
* Votting classifier

## 4. Youhou !!
* It's done





