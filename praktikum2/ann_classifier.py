from keras.layers import Input, Dense
from keras.models import Sequential
from keras.regularizers import l2
from keras.optimizers import Adam, SGD

def annClassifier(input_shape=(30,), optimizer=Adam(0.001), regularizer=l2(0.0)):

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Dense(units=16, activation="relu", use_bias=True, kernel_regularizer = regularizer))
    model.add(Dense(units=16, activation="relu", use_bias=True, kernel_regularizer = regularizer))
    model.add(Dense(units=2, activation="softmax", use_bias=True, kernel_regularizer = regularizer))

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])

    return model