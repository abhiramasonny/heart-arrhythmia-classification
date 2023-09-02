import numpy as np
import pandas as pd
import os
import fnmatch
import librosa
import IPython.display as ipd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from joblib import dump, load 

music_path = "/Users/abhiramasonny/Developer/Python/Projects/heart/sthetacope app/data/set_a/"
sample_audio, sample_rate = librosa.load(music_path + "normal__201106221418.wav", duration=5)
ipd.Audio(sample_audio, rate=sample_rate)

def extract_features(audio_folders, columns, audio_types):
    features_list = []
    index = 0
    for folder in audio_folders:
        for audio_type in audio_types:
            files = fnmatch.filter(os.listdir(folder), audio_type)
            label = audio_type.split("*")[0]
            for file in files:
                x, sr = librosa.load(folder + file, duration=5, res_type='kaiser_fast')
                mfcc_features = [np.mean(x) for x in librosa.feature.mfcc(y=x, sr=sr)]
                features_list.append(mfcc_features)
                features_list[index].extend([
                    sum(librosa.zero_crossings(x)),
                    np.mean(librosa.feature.spectral_centroid(y=x)),
                    np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr)),
                    np.mean(librosa.feature.chroma_stft(y=x, sr=sr)),
                    label,
                    file
                ])
                index += 1
    return pd.DataFrame(features_list, columns=columns)

audio_folders = [
    "/Users/abhiramasonny/Developer/Python/Projects/heart/sthetacope app/data/set_a/",
    "/Users/abhiramasonny/Developer/Python/Projects/heart/sthetacope app/data/set_b/"
]
column_names = ["mfcc_" + str(i) for i in range(20)]
column_names.extend(["zero_crossings", "centroid", "rolloff", "chroma", "label", "filename"])
audio_types = ["normal*.wav", "artifact*.wav", "murmur*.wav", "extrahls*.wav"]

audio_df = extract_features(audio_folders, column_names, audio_types)
X = audio_df.iloc[:, 0:24]
y = audio_df["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.05, random_state=31)

rf = RandomForestClassifier(max_depth=8, max_features=5, min_samples_split=5, n_estimators=500)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
dump(rf, 'models/random_forest_model2.joblib')

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
y_pred_gnb = gnb.predict(X_test_scaled)
print("Gaussian Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_gnb))
dump(gnb, 'models/gaussian_naive_bayes_model2.joblib')

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
dump(svm, 'models/svm_model_poly.joblib')  # Save the model