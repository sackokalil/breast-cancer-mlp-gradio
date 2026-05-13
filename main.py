from sklearn.datasets import load_breast_cancer
import numpy as np
from praktikum2.preprocessing import preprocess
from praktikum2.ann_classifier import annClassifier
from praktikum2.train_model import train_model
from keras.regularizers import l2
from keras.optimizers import Adam, SGD


if __name__=="__main__":

    # data, target = load_breast_cancer(return_X_y=True, as_frame=False)
    # np.save("data/data", data)
    # np.save("data/target", target)
    # print(data.shape,"****", target.shape)
    
    data = np.load("data/data.npy")
    target = np.load("data/target.npy")
    
    np.random.seed(21) #21
    
    #Preprocessing der Daten
    XTrain, YTrain, XVal, YVal, XTest, YTest = preprocess(data, target)
    
    #Training des Modells mit 4 Verschiedenen Settings
    settings = [
        (Adam, 0.001, l2(0.0)),
        (Adam, 0.0005, l2(0.0001)),
        (SGD, 0.01, l2(0.0001)),
        (Adam, 0.001, l2(0.0001))
    ]
    
    for i, (opt_class, lr, regul) in zip(range(len(settings)), settings) : 
        optimizer = opt_class(learning_rate=lr)
        print(
            f"****************OPTIMIZEER: {optimizer.__class__.__name__}, " 
            f"LEARNING_RATE: {lr}, "
            f"L2={regul.l2}*********************"
        )
    
        model = annClassifier((30,), optimizer= optimizer, regularizer=regul)
        model.summary()
        history = train_model(model, XTrain, YTrain, XVal, YVal, num_setting = i+1 )
        test_loss, test_acc = model.evaluate(XTest, YTest)
        print("\n\nTEST-ACCURACY : ", test_acc, "\n\n")


