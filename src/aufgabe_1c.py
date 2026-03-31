"""
Aufgabe 1c: Vortrainiertes CNN aus dem Internet.
MobileNetV2 (ImageNet) wird per Fine-Tuning auf Auto-Erkennung angepasst.
"""
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow import keras
from daten import lade_cifar10


def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = lade_cifar10()

    # Bilder auf 96x96 vergrößern (Teilmenge, sonst zu viel RAM)
    x_train_c = tf.image.resize(x_train[:15000], (96, 96)).numpy()
    y_train_c = y_train[:15000]
    x_val_c = tf.image.resize(x_val[:3000], (96, 96)).numpy()
    y_val_c = y_val[:3000]
    x_test_c = tf.image.resize(x_test[:5000], (96, 96)).numpy()
    y_test_c = y_test[:5000]

    # MobileNetV2 laden (ohne den Klassifikationskopf)
    base_model = keras.applications.MobileNetV2(input_shape=(96, 96, 3), include_top=False, weights='imagenet')

    # Die meisten Schichten einfrieren, nur die letzten 30 trainierbar lassen
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    # Eigenen Kopf draufsetzen
    inputs = keras.Input(shape=(96, 96, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # Training
    model.fit(
        x_train_c, y_train_c,
        validation_data=(x_val_c, y_val_c),
        epochs=5,
        batch_size=64,
        verbose=2
    )

    loss, acc = model.evaluate(x_test_c, y_test_c, verbose=0)
    print(f"Test-Accuracy: {acc:.4f}")

    Path('models/task1c').mkdir(parents=True, exist_ok=True)
    model.save('models/task1c/car_cnn_mobilenetv2.keras')
    print("Gespeichert unter models/task1c/car_cnn_mobilenetv2.keras")


if __name__ == '__main__':
    main()
