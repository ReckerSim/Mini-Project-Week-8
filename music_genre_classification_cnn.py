import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import math
import librosa
import os

def preprocess(dataset_path, num_mfcc=40, n_fft=2048, hop_length=512, num_segment=10):
    data = {"labels": [], "mfcc": []}
    sample_rate = 22050
    samples_per_segment = int(sample_rate*30/num_segment)

    for label_idx, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath == dataset_path:
            continue

        for f in sorted(filenames):
            if not f.endswith('.wav'):
                continue
            file_path = str(dirpath) + '\\' + str(f)

            try:
                y, sr = librosa.load(file_path, sr=sample_rate)
            except:
                print("run")
                continue

            for n in range(num_segment):
                mfcc = librosa.feature.mfcc(y[samples_per_segment*n: samples_per_segment*(n+1)], sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length) 
                mfcc = mfcc.T

                

                if len(mfcc) == math.ceil(samples_per_segment/hop_length):
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(label_idx-4)

    return data

def cnn_model():
    input_shape = X_train.shape[1:]
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation="relu", padding="valid", input_shape=input_shape))
    model.add(MaxPooling2D(2, padding="same"))
    
    model.add(Conv2D(128, (3,3), activation="relu", padding="valid", input_shape=input_shape))
    model.add(MaxPooling2D(2, padding="same"))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(128, (3,3), activation="relu", padding="valid", input_shape=input_shape))
    model.add(MaxPooling2D(2, padding="same"))
    model.add(Dropout(0.3))
    
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    
    # optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

mfcc_data = preprocess(dataset_path="C:\\Users\\SIMHO\\OneDrive\\Desktop\\Applied Data Science Class\\Mini Project W8")

X = np.array(mfcc_data["mfcc"])
Y = np.array(mfcc_data["labels"])

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
Y = to_categorical(Y, num_classes=10)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

Y_train[Y_train==10] = 9
Y_val[Y_val==10] = 9
Y_test[Y_test==10] = 9

cn_model = cnn_model()
cn_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=32, verbose=2)

cn_model.save("music_CNN.h5")

validation_accuracy = cn_model.evaluate(X_val, Y_val)
training_accuracy = cn_model.evaluate(X_train, Y_train)
testing_accuracy = cn_model.evaluate(X_test, Y_test)

print("\nTraining accuracy: %f\t Validation accuracy: %f\t Testing Accuracy: %f" % (training_accuracy[1], validation_accuracy[1], testing_accuracy[1]))
print("\nTraining loss: %f    \t Validation loss: %f    \t Testing Loss: %f \n" % (training_accuracy[0], validation_accuracy[0], testing_accuracy[0]))
