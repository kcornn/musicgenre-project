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
from keras.models import load_model, Sequential, Model
from keras.layers import Input, Dense, LSTM, Dropout, Activation
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend
from keras.utils import np_utils, multi_gpu_model, plot_model
from keras.optimizers import Adam, RMSprop
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
#if tf.test.gpu_device_name():
#    print('Default GPU: {}'.format(tf.test.gpu_device_name()))
#else:
#    print('Failed to find default GPU.')
#    sys.exit(1)
#with tf.device('/device:GPU:0 '):
X_test_pickle = open("X_test.pickle","rb")
X_test = pickle.load(X_test_pickle)

y_test_pickle = open("y_test.pickle","rb")
y_test = pickle.load(y_test_pickle)
y_test.shape

X_val_pickle = open("X_val.pickle","rb")
X_val = pickle.load(X_val_pickle)

y_val_pickle = open("y_val.pickle","rb")
y_val = pickle.load(y_val_pickle)
len(y_val)

with open("X_train1.pickle","rb") as f:
    X_train = pickle.load(f)
with open("X_train2.pickle","rb") as f:
    X_train = np.concatenate([X_train, pickle.load(f)])
with open("X_train3.pickle","rb") as f:
    X_train = np.concatenate([X_train, pickle.load(f)])
with open("X_train4.pickle","rb") as f:
    X_train = np.concatenate([X_train, pickle.load(f)])
with open("X_train5.pickle","rb") as f:
    X_train = np.concatenate([X_train, pickle.load(f)])
with open("X_train6.pickle","rb") as f:
    X_train = np.concatenate([X_train, pickle.load(f)])
with open("X_train7.pickle","rb") as f:
    X_train = np.concatenate([X_train, pickle.load(f)])
y_train = []
with open("y_train1.pickle","rb") as f:
    y_train = pickle.load(f)
with open("y_train2.pickle", "rb") as f:
    y_train = np.concatenate([y_train, pickle.load(f)])
with open("y_train3.pickle", "rb") as f:
    y_train = np.concatenate([y_train, pickle.load(f)])
with open("y_train4.pickle", "rb") as f:
    y_train = np.concatenate([y_train, pickle.load(f)])
with open("y_train5.pickle", "rb") as f:
    y_train = np.concatenate([y_train, pickle.load(f)])
with open("y_train6.pickle", "rb") as f:
    y_train = np.concatenate([y_train, pickle.load(f)])
with open("y_train7.pickle", "rb") as f:
    y_train = np.concatenate([y_train, pickle.load(f)])
print(len(y_train))
print(X_train.shape)

le = sklearn.preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.fit_transform(y_val)
y_test = le.fit_transform(y_test)
genres = le.inverse_transform([0,1,2,3,4,5,6,7])

y_test = np.array(np.eye(8)[y_test.reshape(-1)])
print(y_test.shape)
y_train = np.array(np.eye(8)[y_train.reshape(-1)])
print(y_train.shape)
y_val = np.array(np.eye(8)[y_val.reshape(-1)])
print(y_val.shape)

BATCH_SIZE = 32
EPOCH_COUNT = 80

filters = 56
kernel_size = 3
dropout_rate = 0.3

lstm_units = 256
dense_units = 64

def build_crnn():
    input_shape = (X_train[0].shape)
    model_input = Input(input_shape, name='input')
    x = model_input
    for i in range(3):
        x = Conv1D(
                filters=filters,
                kernel_size=kernel_size,
                kernel_regularizer=regularizers.l2(0.001),
                name='convolution_' + str(i)
            )(x)
        x = BatchNormalization()(x) 
        x = Activation('relu')(x)
        x = MaxPooling1D(2)(x)
        x = Dropout(dropout_rate)(x)
    x = LSTM(lstm_units, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units, kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(8, activation="softmax")(x) # For number of classes
    model_output = x
    model = Model(model_input, model_output)
    opt = Adam(lr=0.001)
    parallel_model = multi_gpu_model(model, 4)
    parallel_model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    cb = ModelCheckpoint('crnn-best.h5', monitor='val_loss', mode='min', save_best_only=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_delta=0.01, verbose=1)
    history = parallel_model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT, \
                                 validation_data=(X_val, y_val), verbose=1, callbacks=[cb, rlr])
    #print(parallel_model.summary())
    return parallel_model, model, history
    

model, savemodel, history = build_crnn()
plot_model(savemodel, to_file='model-crnn.png')
saved_model = load_model('crnn-best.h5')
y_pred = saved_model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
y_test_pickle = open("y_test.pickle","rb")
y_test = pickle.load(y_test_pickle)
y_test = le.fit_transform(y_test)
print(sklearn.metrics.classification_report(y_test, y_pred, labels=[0,1,2,3,4,5,6,7]))
m = sklearn.metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(m.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=genres, yticklabels=genres)
plt.xlabel('Actual Genre')
plt.ylabel('Predicted Genre')
plt.savefig('cf-matrix-crnn.png')
plt.clf()

# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy-crnn.png')
plt.clf()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss-crnn.png')
plt.clf()


