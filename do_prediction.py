from praktikum2.ann_classifier import annClassifier
from praktikum2.predict import predict
from keras.regularizers import l2
from keras.optimizers import Adam, SGD
import pandas as pd
import numpy as np
import os

if __name__=="__main__":

    #Laden des standardisierten XTest
    XTest = np.load("data/XTest_scaled.npy")
    YTest = np.load("data/YTest.npy")
    
    #Wiedererstellung des Models und Laden  der besten gespeicherten Gewichte
    myModel = annClassifier((30,), optimizer= Adam(0.001), regularizer=l2(0))
    myModel.summary()
    myModel.load_weights("models/setting_1_best.weights.h5")
    
    #Vorhersage auf XTest (im skalierten Raum)
    y_pred = predict(myModel, XTest)

    _,test_acc = myModel.evaluate(XTest, YTest)
    print("\n\nTEST-ACCURACY : ", test_acc, "\n\n")
    
    print("*************** PREDICTED KLASSES FÛR XTest ********************\n\n")
    print(y_pred)
    
    
    #File von XTest Vorhersagen
    # yPredict = pd.DataFrame(
    #     {
    #         "Id" : np.arange(1, len(y_pred) + 1 ),
    #         "Predicted Class" : y_pred.astype(int)
    #     }
    # )
    
    
    # output_path="output"
    # os.makedirs(output_path, exist_ok=True)
    
    # yPredict.to_csv(
    #      os.path.join(output_path,"Predicted.csv") ,
    #     index= False,
    # )


