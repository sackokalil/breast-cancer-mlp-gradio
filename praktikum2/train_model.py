from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os



def train_model(model, X_train, Y_train, X_val, Y_val, num_setting):

    # EarlyStopping: 
    es = EarlyStopping(
        monitor='val_loss',
        patience=80,
        verbose=1,
        mode='min',
        restore_best_weights=True,
    )


    # ModelCheckpoint: beste Weights speichern
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_file = os.path.join(models_dir, f"setting_{num_setting}_best.weights.h5")
    
    cp = ModelCheckpoint(
        checkpoint_file,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        save_freq='epoch',
    )

    #Einige parameter festlegen
    callbacks = [es, cp]
    batch_size = 16
    epochs=200
    
    model = model
    history = model.fit(
            x=X_train,
            y=Y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(X_val, Y_val),
    )

    print(f"Beste Weights gespeichert unter: {checkpoint_file}")

    return history
    

