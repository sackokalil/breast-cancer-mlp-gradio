import os
import numpy as np

def predict(model, XTest):

    ypred = model.predict(XTest)
    ypred_classe = np.argmax(ypred, axis = 1); """Weil ypred enthält nur die Wharscheilichkeiten der jeweiligen Klassen"""
    
    return ypred_classe

    