"""
Hilfsfunktionen zum Laden und Aufbereiten der CIFAR-10 Daten.
Wird von allen drei Teilaufgaben der Aufgabe 1 verwendet.
"""
import numpy as np
from tensorflow import keras


def lade_cifar10():
    """
    Lädt CIFAR-10 und gibt Trainings-, Validierungs- und Testdaten zurück.
    Labels werden binär gemacht: Auto (Klasse 1) vs. alles andere.
    Pixelwerte werden auf [0, 1] normiert.
    """
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
    y_train_full = y_train_full.reshape(-1)
    y_test = y_test.reshape(-1)

    # Binäre Labels: Klasse 1 = Auto
    y_train_full = (y_train_full == 1).astype(np.float32)
    y_test = (y_test == 1).astype(np.float32)

    # Normieren
    x_train_full = x_train_full.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # 90/10 Split
    val_size = int(0.1 * len(x_train_full))
    x_val = x_train_full[:val_size]
    y_val = y_train_full[:val_size]
    x_train = x_train_full[val_size:]
    y_train = y_train_full[val_size:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
