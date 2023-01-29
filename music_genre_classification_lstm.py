import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def baseline_model():
    input_shape = X_train.shape[1:]
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(64))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    
    # optimizer = Adam(lr=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

music_three_sec_data = pd.read_csv('Data/features_3_sec.csv')
print(music_three_sec_data.head(10))
print(music_three_sec_data.info())

X = music_three_sec_data.iloc[:,1:-1]
Y = music_three_sec_data.label

X_np = X.to_numpy()
print(X_np.shape)

X_ts = []
Y_ts = []
window_size = 50

for i in range(X_np.shape[0] - window_size):
    X_ts.append(X_np[i: window_size + i, :].tolist())
    Y_ts.append(Y[i])

X_ts_np = np.array(X_ts, dtype=object).astype('float32')
Y_ts_np = np.array(Y_ts, dtype=object)

print(X_ts_np.shape)
print(Y_ts_np.shape)

label_encoder = LabelEncoder()
Y_ts_np = label_encoder.fit_transform(Y_ts_np)

X_train, X_test, Y_train, Y_test = train_test_split(X_ts_np, Y_ts_np, test_size=0.2, stratify=Y_ts_np, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, stratify=Y_train, random_state=42)

bn_model = baseline_model()
bn_model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=60, batch_size=32, verbose=2)

bn_model.save("music_LSTM.h5")

validation_accuracy = bn_model.evaluate(X_val, Y_val)
training_accuracy = bn_model.evaluate(X_train, Y_train)
testing_accuracy = bn_model.evaluate(X_test, Y_test)

print("\nTraining accuracy: %f\t Validation accuracy: %f\t Testing Accuracy: %f" % (training_accuracy[1], validation_accuracy[1], testing_accuracy[1]))
print("\nTraining loss: %f    \t Validation loss: %f    \t Testing Loss: %f \n" % (training_accuracy[0], validation_accuracy[0], testing_accuracy[0]))
