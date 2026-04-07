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

## Aufgaben

### Aufgabe 1 (A1): CNN zur Auto-Erkennung

> 
> **A1 - Umsetzung und Ausführung**
> 
> Das Notebook `notebooks/aufgabe_1.ipynb` enthält die drei Teilaufgaben:
> - **1a)** CNN mit Keras trainieren (CIFAR-10, Auto vs. kein Auto)
> - **1b)** CNN komplett selbst in NumPy geschrieben, mit eigener Backpropagation
> - **1c)** MobileNetV2 aus dem Internet laden und per Fine-Tuning anpassen
> 
> Die gleichen Skripte liegen unter `src/`:
> - `src/aufgabe_1a.py`
> - `src/aufgabe_1b.py`
> - `src/aufgabe_1c.py`
> 
> **Empfohlen:** Ausführung in **Google Colab mit `T4 GPU`**.
> 1. `notebooks/aufgabe_1.ipynb` in Colab öffnen.
> 2. Unter Runtime/Laufzeit den Beschleuniger auf `T4 GPU` stellen.
> 3. Alle Zellen von oben nach unten ausführen.
> 
> Beim ersten Lauf werden CIFAR-10 und MobileNetV2 automatisch heruntergeladen (Internet nötig).
> Lokal in Jupyter ist es auch möglich, aber langsamer (vor allem 1c).
> 
> **Pakete installieren:**
> ```
> pip install -r requirements.txt
> ```
> 
> **Ergebnisse (gespeicherte Modelle):**
> - `models/task1a/car_cnn.keras`
> - `models/task1b/car_cnn_scratch.npz`
> - `models/task1c/car_cnn_mobilenetv2.keras`

### Aufgabe 2 (A2)

> 
> **A2 - Platzhalter**
> 
> In Arbeit.
> 
> **Ausführen:** Folgt.

### Aufgabe 3 (A3)

> 
> **A3 - Platzhalter**
> 
> In Arbeit.
> 
> **Ausführen:** Folgt.
