import numpy as np
import gradio as gr

from sklearn.datasets import load_breast_cancer
from keras.regularizers import l2
from keras.optimizers import Adam, SGD, RMSprop

from praktikum2.preprocessing import preprocess
from praktikum2.ann_classifier import annClassifier
from praktikum2.train_model import train_model
import os


# --------------------------
# Globale Variablen
# --------------------------

XTrain = None
YTrain = None
XVal = None
YVal = None
XTest = None
YTest = None

current_model = None  # wird nach dem Training gesetzt
list_weight = None  # wird durch ini_data() funktion initialisiert




# --------------------------
# Gespeicherte Gewichte laden 
# --------------------------

def get_weight():

    dir = "models"
    global list_weight

    if not os.path.exists("models"):
        list_weight = []
    else:
        list_weight = os.listdir(dir)


# --------------------------
# Daten laden & preprocessen
# --------------------------

def init_data():
    global XTrain, YTrain, XVal, YVal, XTest, YTest

    data = np.load("data/data.npy")
    target = np.load("data/target.npy")

    get_weight() 

    np.random.seed(21)
    XTrain, YTrain, XVal, YVal, XTest, YTest = preprocess(data, target)


# --------------------------
# Optimizer-Factory
# --------------------------

def get_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return Adam(learning_rate=lr)
    elif name == "sgd":
        return SGD(learning_rate=lr)
    elif name == "rmsprop":
        return RMSprop(learning_rate=lr)
    else:
        raise ValueError(f"Unbekannter Optimizer: {name}")


# --------------------------
# Gradio-Funktionen
# --------------------------

def train_gradio(l2_value: float, lr: float, optimizer_name: str):
    """
    Wird vom 'Trainieren'-Button in Gradio aufgerufen.
    Baut ein neues Modell mit den gegebenen Hyperparametern,
    trainiert es und gibt Val-Accuracy und Tes-Accuracy zurück.
    """
    global current_model

    if XTrain is None:
        init_data()

    # Optimizer & Regularizer bauen
    optimizer = get_optimizer(optimizer_name, lr)
    regularizer = l2(l2_value)

    # Modell erstellen
    input_shape = (XTrain.shape[1],)  
    model = annClassifier(input_shape=input_shape,
                          optimizer=optimizer,
                          regularizer=regularizer)

    # Modell trainieren 
    history = train_model(model, XTrain, YTrain, XVal, YVal, "gradio")

    # Nach train_model sollte man bereits die besten Weights
    # (EarlyStopping + restore_best_weights) im Modell haben.
    _ , val_acc = model.evaluate(XVal, YVal, verbose=0)
    _ , test_acc = model.evaluate(XTest, YTest, verbose=0)

    # Modell speichern, damit die Vorhersage-Funktion darauf zugreifen kann
    current_model = model

    msg = (
        f"Training abgeschlossen.\n\n"
        f"Optimizer: {optimizer_name}, Lernrate: {lr}, L2: {l2_value}\n"
        f"Validierungs-Accuracy: {val_acc:.4f}\n"
        f"Test-Accuracy: {test_acc:.4f}\n"
        
    )
    return msg


def predict_val_sample(index: int, weight: str ):
    """
    Wird von Gradio aufgerufen, um ein einzelnes Beispiel
    aus der Validierungsmenge vorherzusagen.
    """

    global current_model

    if XVal is None:
        init_data()

    #Modell mit derselben Architektur neu aufbauen und Gewichte laden
    if weight :

        weight_path = os.path.join("models", weight)
        if not os.path.exists(weight_path):
            return f"Gewichtsdatei '{weight_path}' nicht gefunden "
        
        model = annClassifier(input_shape=(XVal.shape[1],))
        model.load_weights(weight_path)
        current_model = model

    else:
        if current_model is None:
            return "Bitte zuerst ein Modell trainieren."
    

    # Index absichern
    idx = int(index)
    if idx < 0 or idx >= XVal.shape[0]:
        return f"Index {idx} ist außerhalb des gültigen Bereichs (0 - {XVal.shape[0]-1})."

    x = XVal[idx].reshape(1, -1)
    y_true = YVal[idx]

    # Vorhersage
    y_pred = current_model.predict(x, verbose=0)

    #2-Output-Softmax 
    p0 = float(y_pred[0, 0])
    p1 = float(y_pred[0, 1])
    y_hat = int(np.argmax(y_pred, axis=1)[0])


    type_model = weight or "Gerade Trainiertes Model"
    text = (
        f"Model : {type_model}\n"
        f"Index in Validierungsmenge: {idx}\n"
        f"Wahre Klasse:       {y_true}\n"
        f"Vorhergesagte Klasse: {y_hat}\n"
        f"Wahrscheinlichkeiten: P(0) = {p0:.3f}, P(1) = {p1:.3f}"
    )
    return text


# --------------------------
# Gradio-UI bauen
# --------------------------

def build_interface():
    init_data()  # einmalig beim Start vorbereiten

    with gr.Blocks() as demo:
        gr.Markdown("# Breast Cancer MLP – Hyperparameter-Interface")

        with gr.Tab("Training"):
            gr.Markdown("## Modell trainieren")

            with gr.Row():
                l2_input = gr.Slider(
                    minimum=0.0, maximum=0.01, step=0.0001,
                    value=0.0001, label="L2-Regularisierungswert"
                )
                lr_input = gr.Slider(
                    minimum=1e-5, maximum=1e-1, step=1e-4,
                    value=0.001, label="Lernrate"
                )
                opt_input = gr.Dropdown(
                    choices=["Adam", "SGD", "RMSprop"],
                    value="Adam",
                    label="Optimizer"
                )

            train_button = gr.Button("Modell trainieren")
            train_output = gr.Textbox(
                label="Trainingsergebnis",
                lines=6
            )

            train_button.click(
                fn=train_gradio,
                inputs=[l2_input, lr_input, opt_input],
                outputs=train_output
            )

        with gr.Tab("Vorhersage"):
            gr.Markdown("## Vorhersage auf einem Validierungsbeispiel")


            with gr.Column():

                gr.Markdown("****Index des Validierungsbeispiels****")
                index_input = gr.Slider(
                minimum=0,
                maximum=int(YVal.shape[0] - 1),
                step=1,
                value=0,
                label=""
                )

                if list_weight:
                    gr.Markdown("****Model auswählen****")
                    weight_input = gr.Dropdown(
                        choices=list_weight,
                        value=list_weight[0],
                        label=""
                    )
                else:
                    gr.Markdown("****Model auswählen****")
                    weight_input = gr.Dropdown(
                        choices=[],
                        label="Keine gespeicherten Modelle gefunden, " \
                        "Sie können trotzdem Vorhersage durchführen aber erst " \
                        "nachdem Sie das Model trainiert haben",
                        interactive=False
                    )
   
                
            predict_button = gr.Button("Beispiel vorhersagen")

            gr.Markdown("****Ergebnis des Vorhersagens****")
            pred_output = gr.Textbox(
                label=" Die Vorhersage",
                lines=6
            )

            
            predict_button.click(
            fn=predict_val_sample,
            inputs=[index_input, weight_input],
            outputs=pred_output
            )
            
            

            

        return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()