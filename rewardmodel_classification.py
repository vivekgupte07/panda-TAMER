import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.saving.save import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from numpy import loadtxt
import numpy as np


def train_completely(X, y):
    sampleX = [[1,1,1,7,1,1,1], [0,0,0,1,0,0,0]]
    scalarX, scalarY = MinMaxScaler(),  LabelEncoder()
    scalarX.fit(sampleX)
    scalarY.fit(y.reshape(len(y), 1))
    X = scalarX.transform(X)
    np.round(X,2)
    y = scalarY.transform(y.reshape(len(y), 1))

    # create model
    model = Sequential()
    model.add(Dense(64, input_shape=(7,), kernel_initializer='normal',
                    activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    print(">>>UPDATING WEIGHTS...")
    model.fit(X, y, batch_size=50, epochs=2000, verbose=0)
    model.save('model_1.h5')
    #model.summary()
    score = model.evaluate(X, y, verbose=0)
    return model


def train_partially(X, y):
    sampleX = [[1,1,1,7,1,1,1], [0,0,0,1,0,0,0]]
    scalarX, scalarY = MinMaxScaler(), LabelEncoder()
    scalarX.fit(sampleX)
    scalarY.fit(y.reshape(len(y), 1))
    X = scalarX.transform(X)
    y = scalarY.transform(y.reshape(len(y), 1))

    # Load model
    if os.path.exists('model_1.h5'):
        model = load_model('model_1.h5')
    else:
        train_model(1)
        model = load_model('model_1.h5')
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    print(">>>UPDATING WEIGHTS...")
    model.fit(X, y, batch_size=50, epochs=2000, verbose=0)
    model.save('model_1.h5')
    # model.summary()
    score = model.evaluate(X, y, verbose=0)
    return model


def predict(x_new):
    model_path = 'model_1.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        train_model(2)
        model = load_model(model_path)
    X = [[1,1,1,7,1,1,1], [0,0,0,1,0,0,0]]
    scalarXnew = MinMaxScaler()
    scalarXnew.fit(X)
    x_new = scalarXnew.transform(x_new.reshape(1, -1))
    y_new = model.predict(x_new)
    return y_new

def train_model(method):
    X = loadtxt('features.csv')
    y = loadtxt('rewards.csv')

    if method == 1:
        model = train_completely(X, y)
    if method == 2:
        model = train_partially(X, y)
