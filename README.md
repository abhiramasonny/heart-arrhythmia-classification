# heart-arrhythmia-classification

## Overview

I coded a model which extracts data from the `.wav` files and then uses 3 ai models and produces 3 models.

## Features Extracted

1. Mel-frequency cepstral coefficients (MFCC)
2. Zero crossings
3. Spectral centroid
4. Spectral rolloff
5. Chroma

### Machine Learning Classifiers

1. Random Forest
2. Gaussian Naive Bayes
3. Support Vector Machines with a polynomial kernel

## Usage

Run `test.py` and get a heartbeat file as input.