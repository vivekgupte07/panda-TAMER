import tensorflow as tf
from keras.saving.save import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from numpy import loadtxt
import os


# define the model
def larger_model(X, y):
    # create model
    scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
    scalarX.fit(X)
    scalarY.fit(y.reshape(len(y), 1))
    X = scalarX.transform(X)
    y = scalarY.transform(y.reshape(len(y), 1))

    model = Sequential()
    model.add(Dense(64, input_shape=(7,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, batch_size=10, epochs=1000, verbose=1)
    model.save('model_1.h5')
    model.summary()
    score = model.evaluate(X, y, verbose=0)
    print(score)
    return model


def make_prediction(x_new):
    # load dataset
    X = loadtxt('artifacts/features.csv')
    y = loadtxt('artifacts/rewards.csv')
    if os.path.exists('artifacts/model_1.h5'):
        model = load_model('artifacts/model_1.h5')

    scalarXnew = MinMaxScaler()
    scalarXnew.fit(X)
    x_new = scalarXnew.transform(x_new.reshape(1, -1))
    y_new = model.predict(x_new)
    return y_new


def train_partially(X, y):
    # create model
    scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
    scalarX.fit(X)
    scalarY.fit(y.reshape(len(y), 1))
    X = scalarX.transform(X)
    y = scalarY.transform(y.reshape(len(y), 1))

    model = load_model('model_1.h5')
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, batch_size=1, epochs=100, verbose=1)
    model.save('artifacts/model_1.h5')
    return model


def train_model(method):
    X = loadtxt('artifacts/features.csv')
    y = loadtxt('artifacts/rewards.csv')
    if method == 'full':
        model = larger_model(X, y)
    if method == 'partial':
        model = train_partially(X, y)
