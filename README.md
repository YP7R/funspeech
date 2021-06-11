# Funspeech, *Bachelor Final work* 

## 0. Collect dataset : Western Michigan University or FunSpeech
* https://homepages.wmich.edu/~hillenbr/voweldata.html
```
dataset\raw\{dataset_name}\[sounds.wav]

--- dataset
    |--- raw
        |--- {dataset_name}
            |--- [sounds.wav]
```

## 1. Preprocessing
[funspeech processing](./preprocessing_funspeech.py), [wmu processing](./preprocessing_wmusounds.py)
* cut silence
* extract phone/phonemes
* generate information files `sound_id, category, base_file(username), length`
```
dataset\processed\{dataset_name}\[{dataset_name}.csv, sounds_processed.wav]

--- dataset
    |--- processed
        |--- {dataset_name}
            |--- {dataset_name}.csv
            |--- [sounds_processed.wav]

```
---
In case of [`RunTimeWarning: Couldn't find ffmpeg or avconv`](http://blog.gregzaal.com/how-to-install-ffmpeg-on-windows/)  
In case of error message when processing, check files.  

## 2. Extract acoustic features on signal and a sample of signal
[extraction features](./extraction_features.py)
* generate numpy array files for features
* Sample of a signal / Signal
* Energies, Log Filterbank, M-FCC, MelWeighted-FCC, Linear-FCC & ERB-FCC

```
dataset\features\{dataset_name}\[{features_name}.npy, {features_name}_sample.npy]

--- dataset
    |--- features
        |--- {dataset_name}
            |--- [{features_name}.npy]
            |--- [{features_name}_sample.npy]
```

## 3. Classification ... train, test & validation sets
Classify datas and generate .png files
* BaryCentre
* KNN
* SVM
* Bagging
* Voting Classifier
## [main](./main.py)
Train on adults and predict on children
* Train, Test & Validation set
* Accuracy, NegLogLoss, Time 
```
dataset\results\{dataset_name}\[{dataset_name}_{features_name}_{classifier_name}.png, {dataset_name}_{features_name}_sample_{classifier_name}.png]

--- dataset
    |--- results
        |--- {dataset_name}
            |--- [{dataset_name}_{features_name}_{classifier_name}.png]
            |--- [{dataset_name}_{features_name}_sample_{classifier_name}.png]
```
## [main2](./main2.py)
Train & Test on the whole dataset
* Train & Test 
* Accuracy, Time 
```
dataset\results_2\{dataset_name}\[{dataset_name}_{features_name}_{classifier_name}.png, {dataset_name}_{features_name}_sample_{classifier_name}.png]

--- dataset
    |--- results_2
        |--- {dataset_name}
            |--- [{dataset_name}_{features_name}_{classifier_name}.png]
            |--- [{dataset_name}_{features_name}_sample_{classifier_name}.png]
```
## [confusion matrix](./debug/confusion_matrix.py)
Print a confusion matrix for the chosen classifier and chosen features
```
dataset\confusion_matrix\[{dataset_name}_{features_name}.png]

--- dataset
    |--- confusion_matrix
        |--- [{dataset_name}_{features_name}.png]
```
## [voting classifier](debug/voting_clf.py)
Execute Voting classifier for the chosen classifier and chosen features
```
dataset\voting_clf\[{dataset_name}_{features_name}_voting_classifier.png]

--- dataset
    |--- voting_clf
        |--- [{dataset_name}_{features_name}_voting_classifier.png]
```

## 4. Youhou !!
* It's done
* next ... confusion matrix for each feature and classifier
* next ... voting classifier for each feature 




