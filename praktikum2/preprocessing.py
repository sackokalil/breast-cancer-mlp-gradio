import numpy as np
from keras.utils import to_categorical

def preprocess(data, target):
    #Shufflen der Daten
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data, target = data[indices], target[indices]

    #Encoding der Labels
    target = to_categorical(target, 2)

    #Teilen der Daten  in Trainings-, Validierungs und Testdaten
    train_ratio = 0.7
    val_ratio = 0.15
    #test_ratio = 0.15

    n_sample = data.shape[0]

    train_size = int(train_ratio * n_sample)
    val_size = int(val_ratio * n_sample)
    #test_size = int(test_ratio * n_sample)

    XTrain = data[:train_size, :]
    YTrain = target[:train_size]
    XVal = data[train_size:train_size+val_size, :]
    YVal = target[train_size:train_size+val_size]
    XTest = data[train_size+val_size : , : ]
    YTest = target[train_size+val_size :]

    #Normalisierung : Min-Max normaisation über Trainings daten
    """train_min = np.min(XTrain)
    train_max = np.max(XTrain)

    XTrain_normalised = (XTrain - train_min)/(train_max - train_min)
    XVal_normalised = (XVal - train_min)/(train_max - train_min)
    XTest_normalised = (XTest - train_min)/(train_max - train_min)
    np.save("data/XTest_normalised", XTest_normalised)"""


    #Standardisierung
    mu = XTrain.mean(axis=0)
    sigma = XTrain.std(axis=0)
    sigma[sigma == 0] = 1; """Falls ein Feature konstant ist, dann Standardabweichung = 0; also: (x - µ) / 0 -> Division durch 0 = Problem"""

    XTrain_scaled = (XTrain - mu) / sigma
    XVal_scaled   = (XVal   - mu) / sigma
    XTest_scaled  = (XTest  - mu) / sigma
    np.save("data/XTest_scaled", XTest_scaled)
    np.save("data/YTest", YTest)
    


    #return XTrain_normalised, YTrain, XVal_normalised, YVal, XTest_normalised, YTest
    return XTrain_scaled, YTrain, XVal_scaled, YVal, XTest_scaled, YTest
    