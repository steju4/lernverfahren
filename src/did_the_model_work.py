import numpy as np
from tensorflow import keras
from PIL import Image
from pathlib import Path

print("Hallo")

# Modell laden
filepath = Path("../../models/task1a/car_cnn.keras")
model = keras.models.load_model("C:\Users\KOY2BH\OneDrive - Bosch Group\PersonalDrive\T2000\Studium Projekte\Lernverfahren\lernverfahren\models\task1a\car_cnn.keras")

# Bild laden und vorbereiten
img = Image.open("auto.png").resize((32, 32))
img_array = np.array(img).astype('float32') / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Vorhersage
prob = model.predict(img_array, verbose=0)[0][0]
is_car = prob >= 0.5

# Ausgabe
if is_car:
    print(f"AUTO ({prob*100:.2f}%)")
else:
    print(f"Was anderes({(1-prob)*100:.2f}%)")