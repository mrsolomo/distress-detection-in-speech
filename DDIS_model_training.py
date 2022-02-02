# By: SAYSON, Mary Raullette
#       1155156595@link.cuhk.edu.hk

import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from lib_features import lib_features_extract

DATASET_PATH = "D:/School/CUHK/Research/Dataset/RAVDESS/"

# we allow only these emotions
AVAILABLE_EMOTIONS = {
    "neutral",
    "angry"
}

# all emotions on RAVDESS dataset
int2emotion_dict = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def load_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob(DATASET_PATH + "Actor_*/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        emotion = int2emotion_dict[basename.split("-")[2]]
        # we allow only AVAILABLE_EMOTIONS we set
        if emotion not in AVAILABLE_EMOTIONS:
            continue
        # print(file)
        # extract speech features
        features = lib_features_extract(file, \
            mfcc=True, chroma=False, mel=False, contrast=False, tonnetz=False,\
            rms=False, zcr=False )
        # add to data
        X.append(features)
        y.append(emotion)
    
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)


if __name__ == "__main__":
    # Load Dataset
    X_train, X_test, y_train, y_test = load_data(test_size=0.25)
    
    # number of samples in training data
    print("[+] Number of training samples:", X_train.shape[0])
    # number of samples in testing data
    print("[+] Number of testing samples:", X_test.shape[0])
    # number of features used
    # this is a vector of features extracted 
    # using utils.extract_features() method
    print("[+] Number of features:", X_train.shape[1])

    # Load Model
    # model = KNeighborsClassifier(n_neighbors=3)
    model = svm.SVC(kernel='linear', C = 1.0)

    # Train the model
    print("[*] Training the model...")
    model.fit(X_train, y_train)

    # Test the model - predict 25% of data to measure how good we are
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Accuracy: {:.2f}%".format(accuracy*100))

    # Deploy the model - make result directory if doesn't exist yet
    if not os.path.isdir("result"):
        os.mkdir("result")

    pickle.dump(model, open("model/linear_svc.model", "wb"))
