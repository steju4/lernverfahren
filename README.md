# Praxisprojekt Lernverfahren

Praxisprojekt für die Vorlesung Grundlagen maschineller Lernverfahren (TIS/TIK24).

### Bearbeiter

| Name | Studiengang |
|------|-------------|
| Johann Flögel | TIK24 |
| Chiara Herz | TIK24 |
| Yanneck Kolb | TIK24 |
| Lucius Lechner | TIK24 |
| Julian Stengele | TIK24 |

## Aufbau

```
notebooks/          Jupyter Notebooks (eins pro Aufgabe)
src/                Python-Dateien als Einzelscripte
docs/               Aufgabenstellung und Handouts
data/               (wird beim Ausführen erstellt – CIFAR-10 etc.)
models/             (wird beim Ausführen erstellt – trainierte Modelle)
```

## Aufgabe 1: CNN zur Auto-Erkennung

Das Notebook `notebooks/aufgabe_1.ipynb` enthält alle drei Teilaufgaben:

- **1a)** CNN mit Keras trainieren (CIFAR-10, Auto vs. kein Auto)
- **1b)** CNN komplett selbst in NumPy geschrieben, mit eigener Backpropagation
- **1c)** MobileNetV2 aus dem Internet laden und per Fine-Tuning anpassen

Die gleichen Scripte liegen auch unter `src/` als einzelne Python-Dateien:
- `src/aufgabe_1a.py`, `src/aufgabe_1b.py`, `src/aufgabe_1c.py`

### Ausführen

Am einfachsten das Notebook in Google Colab oder lokal in Jupyter öffnen und alle Zellen von oben nach unten ausführen. Beim ersten Mal wird CIFAR-10 und MobileNetV2 automatisch heruntergeladen (Internet nötig).

### Pakete installieren

```
pip install -r requirements.txt
```

### Ergebnisse

Nach dem Training werden die Modelle gespeichert:
- `models/task1a/car_cnn.keras`
- `models/task1b/car_cnn_scratch.npz`
- `models/task1c/car_cnn_mobilenetv2.keras`
