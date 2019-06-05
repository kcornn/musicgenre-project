import numpy as np
import pandas as pd
import matplotlib
import sklearn
import sklearn.preprocessing
import sklearn.metrics
from sklearn.model_selection import GridSearchCV
import pickle
import keras
import seaborn as sns
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, Flatten, Conv2D, BatchNormalization, Lambda
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop
from keras import regularizers

#X_test_pickle = open("data/X_test.pickle","rb")
#X_test = pickle.load(X_test_pickle)

#y_test_pickle = open("data/y_test.pickle","rb")
#y_test = pickle.load(y_test_pickle)
#y_test.shape

X_val_pickle = open("data/X_val.pickle","rb")
X_val = pickle.load(X_val_pickle)

y_val_pickle = open("data/y_val.pickle","rb")
y_val = pickle.load(y_val_pickle)
len(y_val)

with open("data/X_train1.pickle","rb") as f:
    X_train = pickle.load(f)
with open("data/X_train2.pickle","rb") as f:
    X_train = np.concatenate([X_train, pickle.load(f)])
with open("data/X_train3.pickle","rb") as f:
    X_train = np.concatenate([X_train, pickle.load(f)])
with open("data/X_train4.pickle","rb") as f:
    X_train = np.concatenate([X_train, pickle.load(f)])
"""
with open("data/X_train5.pickle","rb") as f:
    X_train = np.concatenate([X_train, pickle.load(f)])
with open("data/X_train6.pickle","rb") as f:
    X_train = np.concatenate([X_train, pickle.load(f)])
with open("data/X_train7.pickle","rb") as f:
    X_train = np.concatenate([X_train, pickle.load(f)])
"""
y_train = []
with open("data/y_train1.pickle","rb") as f:
    y_train = pickle.load(f)
with open("data/y_train2.pickle", "rb") as f:
    y_train = np.concatenate([y_train, pickle.load(f)])
with open("data/y_train3.pickle", "rb") as f:
    y_train = np.concatenate([y_train, pickle.load(f)])
with open("data/y_train4.pickle", "rb") as f:
    y_train = np.concatenate([y_train, pickle.load(f)])
"""
with open("data/y_train5.pickle", "rb") as f:
    y_train = np.concatenate([y_train, pickle.load(f)])
with open("data/y_train6.pickle", "rb") as f:
    y_train = np.concatenate([y_train, pickle.load(f)])
with open("data/y_train7.pickle", "rb") as f:
    y_train = np.concatenate([y_train, pickle.load(f)])
"""
print(len(y_train))
print(X_train.shape)

le = sklearn.preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.fit_transform(y_val)
#y_test = le.fit_transform(y_test)
genres = le.inverse_transform([0,1,2,3,4,5,6,7])

#y_test = np.array(np.eye(8)[y_test.reshape(-1)])
#print(y_test.shape)
y_train = np.array(np.eye(8)[y_train.reshape(-1)])
print(y_train.shape)
y_val = np.array(np.eye(8)[y_val.reshape(-1)])
print(y_val.shape)

BATCH_SIZE = 32
#EPOCH_COUNT = 50
EPOCH_COUNT = 3

LAYERS = 3
FILTERS = 256
KERNEL_SIZE = 5
#DROPOUT = 0.3

LSTM_COUNT = 96

def build_nn(filters=256, dropout_rate=0.3):
    for i in range(LAYERS):
        model = Sequential([
            Conv1D(
                filters=filters,
                kernel_size=KERNEL_SIZE,
                name='convolution_' + str(i + 1)
            ),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling1D(2),
            Dropout(dropout_rate)
        ])
    model.add(LSTM(LSTM_COUNT, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(8, activation="softmax")) # For number of classes
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    #cb = ModelCheckpoint('crnn-best.h5', monitor='val_loss', mode='min', save_best_only=True)
    #history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT, validation_data=(X_val, y_val), verbose=1, callbacks=[cb])
    #print(model.summary())
    return model

#model = KerasClassifier(build_fn=build_nn, verbose=1, batch_size=32, epochs=2)
model = KerasClassifier(build_fn=build_nn, verbose=1)

filters = [64, 96, 128, 256]
batch_size = [32, 64]
epochs = [25, 50]
dropout_rate = [0.3, 0.4]
param_grid = dict(dropout_rate=dropout_rate, batch_size=batch_size, epochs=epochs)
param_grid = dict(filters=filters, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(means, params):
    print("%f with: %r" % (mean, param))






