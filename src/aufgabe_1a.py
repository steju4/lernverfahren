"""
Aufgabe 1a: CNN mit Keras/TensorFlow zur Auto-Erkennung.
Trainiert ein einfaches CNN auf CIFAR-10 (binär: Auto vs. kein Auto).
"""
from pathlib import Path
from tensorflow import keras
from daten import lade_cifar10


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = lade_cifar10()

    # Modell aufbauen
    model = keras.Sequential([
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Training mit Early Stopping damit wir nicht zu lange trainieren
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=64,
        callbacks=[early_stop],
        verbose=2
    )

    # Auswertung auf Testdaten
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test-Accuracy: {acc:.4f}")

    # Speichern
    Path('models/task1a').mkdir(parents=True, exist_ok=True)
    model.save('models/task1a/car_cnn.keras')
    print("Gespeichert unter models/task1a/car_cnn.keras")


if __name__ == '__main__':
    main()
