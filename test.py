import numpy as np
import librosa
from joblib import load
from sklearn.preprocessing import StandardScaler
from collections import Counter

def extract_audio_features(file_path):
    # need to xtract MFCC
    x, sr = librosa.load(file_path, duration=5, res_type='kaiser_fast')
    mfcc_features = [np.mean(x) for x in librosa.feature.mfcc(y=x, sr=sr)]
    other_features = [
        sum(librosa.zero_crossings(x)),
        np.mean(librosa.feature.spectral_centroid(y=x)),
        np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr)),
        np.mean(librosa.feature.chroma_stft(y=x, sr=sr))
    ]
    return mfcc_features + other_features

rf = load('models/random_forest_model.joblib')
gnb = load('models/gaussian_naive_bayes_model.joblib')
svm = load('models/svm_model.joblib')
file_path = input("Please provide the path to the audio file: ")
features = np.array(extract_audio_features(file_path)).reshape(1, -1)

scaler = StandardScaler().fit(features)
scaled_features = scaler.transform(features)

prediction_rf = rf.predict(features)[0]
prediction_gnb = gnb.predict(scaled_features)[0]
prediction_svm = svm.predict(features)[0]

labels = ["artifact", "murmur", "normal"]
rfpred = labels[prediction_rf]
gnbpred = labels[prediction_gnb]
svmpred = labels[prediction_svm]

print(f"Random Forest Prediction: {rfpred}")
print(f"Gaussian Naive Bayes Prediction: {gnbpred}")
print(f"SVM Prediction: {svmpred}")

predictions = [rfpred, gnbpred, svmpred]
counter = Counter(predictions)
if len(counter) == len(predictions):
    most_common_prediction = rfpred
else:
    most_common_prediction = counter.most_common(1)[0][0]

print(f"Final Prediction: {most_common_prediction}")