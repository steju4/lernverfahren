"""
Aufgabe 1b: CNN ohne keras.models / keras.layers.
Eigene Implementierung von Conv2D, ReLU, MaxPooling, Dense, Sigmoid
mit Forward- und Backward-Pass in NumPy.
"""
import numpy as np
from pathlib import Path
from daten import lade_cifar10


# ---- Layer-Klassen ----

class Conv2D:
    """Eigene Faltungsschicht"""
    def __init__(self, in_ch, out_ch, k_size, scale=0.3):
        self.k = k_size
        self.w = np.random.randn(out_ch, k_size, k_size, in_ch).astype(np.float32) * scale
        self.b = np.zeros((out_ch,), dtype=np.float32)
        self.x = None

    def forward(self, x):
        self.x = x
        n, h, w, _ = x.shape
        oh = h - self.k + 1
        ow = w - self.k + 1
        out = np.zeros((n, oh, ow, self.w.shape[0]), dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                patch = x[:, i:i+self.k, j:j+self.k, :]
                for oc in range(self.w.shape[0]):
                    out[:, i, j, oc] = np.sum(patch * self.w[oc], axis=(1, 2, 3)) + self.b[oc]
        return out

    def backward(self, grad_out, lr):
        x = self.x
        n, h, w, _ = x.shape
        oh = h - self.k + 1
        ow = w - self.k + 1
        grad_x = np.zeros_like(x)
        grad_w = np.zeros_like(self.w)
        grad_b = np.zeros_like(self.b)
        for i in range(oh):
            for j in range(ow):
                patch = x[:, i:i+self.k, j:j+self.k, :]
                for oc in range(self.w.shape[0]):
                    g = grad_out[:, i, j, oc].reshape(n, 1, 1, 1)
                    grad_w[oc] += np.sum(patch * g, axis=0)
                    grad_x[:, i:i+self.k, j:j+self.k, :] += self.w[oc] * g
                    grad_b[oc] += np.sum(grad_out[:, i, j, oc])
        self.w -= lr * (grad_w / n)
        self.b -= lr * (grad_b / n)
        return grad_x


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad_out, lr):
        return grad_out * self.mask


class MaxPool2x2:
    def __init__(self):
        self.x = None
        self.argmax = None

    def forward(self, x):
        self.x = x
        n, h, w, c = x.shape
        oh, ow = h // 2, w // 2
        out = np.zeros((n, oh, ow, c), dtype=np.float32)
        self.argmax = np.zeros((n, oh, ow, c), dtype=np.int32)
        for i in range(oh):
            for j in range(ow):
                patch = x[:, 2*i:2*i+2, 2*j:2*j+2, :].reshape(n, 4, c)
                idx = np.argmax(patch, axis=1)
                self.argmax[:, i, j, :] = idx
                out[:, i, j, :] = np.take_along_axis(patch, idx[:, None, :], axis=1).squeeze(1)
        return out

    def backward(self, grad_out, lr):
        n, h, w, c = self.x.shape
        oh, ow = h // 2, w // 2
        grad_x = np.zeros_like(self.x)
        for i in range(oh):
            for j in range(ow):
                idx = self.argmax[:, i, j, :]
                g = grad_out[:, i, j, :]
                for ni in range(n):
                    for ci in range(c):
                        fi = idx[ni, ci]
                        r, col = fi // 2, fi % 2
                        grad_x[ni, 2*i+r, 2*j+col, ci] += g[ni, ci]
        return grad_x


class Flatten:
    def __init__(self):
        self.shape = None

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_out, lr):
        return grad_out.reshape(self.shape)


class Dense:
    def __init__(self, in_feat, out_feat, scale=0.1):
        self.w = np.random.randn(in_feat, out_feat).astype(np.float32) * scale
        self.b = np.zeros((1, out_feat), dtype=np.float32)
        self.x = None

    def forward(self, x):
        self.x = x
        return x @ self.w + self.b

    def backward(self, grad_out, lr):
        n = self.x.shape[0]
        grad_w = (self.x.T @ grad_out) / n
        grad_b = np.mean(grad_out, axis=0, keepdims=True)
        grad_x = grad_out @ self.w.T
        self.w -= lr * grad_w
        self.b -= lr * grad_b
        return grad_x


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, grad_out, lr):
        return grad_out * self.out * (1.0 - self.out)


# ---- CNN zusammensetzen ----

class ScratchCNN:
    """Unser eigenes CNN - ohne Keras"""
    def __init__(self):
        self.layers = [
            Conv2D(in_ch=3, out_ch=8, k_size=3),
            ReLU(),
            MaxPool2x2(),
            Flatten(),
            Dense(7 * 7 * 8, 64),   # nach conv(16->14) + pool(14->7) = 7x7x8
            ReLU(),
            Dense(64, 1),
            Sigmoid()
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def bce_loss(self, pred, true):
        eps = 1e-7
        pred = np.clip(pred, eps, 1 - eps)
        return float(np.mean(-(true * np.log(pred) + (1 - true) * np.log(1 - pred))))

    def bce_grad(self, pred, true):
        eps = 1e-7
        pred = np.clip(pred, eps, 1 - eps)
        return (-(true / pred) + ((1 - true) / (1 - pred))) / true.shape[0]

    def backward(self, grad, lr):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def predict(self, x):
        p = self.forward(x)
        return (p >= 0.5).astype(np.float32)

    def save(self, path):
        conv = self.layers[0]
        d1 = self.layers[4]
        d2 = self.layers[6]
        np.savez(path, conv_w=conv.w, conv_b=conv.b,
                 d1_w=d1.w, d1_b=d1.b, d2_w=d2.w, d2_b=d2.b)


# ---- Training ----

def main():
    (x_train, y_train), _, (x_test, y_test) = lade_cifar10()

    # Kleinerer Datensatz für die eigene Implementierung
    # Bilder runterskalieren auf 16x16 (jedes zweite Pixel)
    # Klassen ausbalancieren damit das Netz auch wirklich lernt
    np.random.seed(24)

    n_train_per_class = 1800
    n_test_per_class = 700

    auto_idx = np.where(y_train == 1)[0][:n_train_per_class]
    nicht_auto_idx = np.where(y_train == 0)[0][:n_train_per_class]
    idx_b = np.concatenate([auto_idx, nicht_auto_idx])
    np.random.shuffle(idx_b)

    x_train_b = x_train[idx_b][:, ::2, ::2, :]
    y_train_b = y_train[idx_b].reshape(-1, 1)

    auto_idx_t = np.where(y_test == 1)[0][:n_test_per_class]
    nicht_auto_idx_t = np.where(y_test == 0)[0][:n_test_per_class]
    idx_bt = np.concatenate([auto_idx_t, nicht_auto_idx_t])
    np.random.shuffle(idx_bt)
    x_test_b = x_test[idx_bt][:, ::2, ::2, :]
    y_test_b = y_test[idx_bt].reshape(-1, 1)

    print("Train (1b):", x_train_b.shape, y_train_b.shape, "| Anteil Autos:", f"{y_train_b.mean():.2f}")
    print("Test  (1b):", x_test_b.shape, y_test_b.shape, "| Anteil Autos:", f"{y_test_b.mean():.2f}")

    # Training
    model_b = ScratchCNN()
    epochs_b = 15
    lr_b = 0.1
    bs = 32

    for epoch in range(1, epochs_b + 1):
        # Daten mischen
        idx = np.random.permutation(len(x_train_b))
        losses = []
        for start in range(0, len(idx), bs):
            batch = idx[start:start+bs]
            xb, yb = x_train_b[batch], y_train_b[batch]
            pred = model_b.forward(xb)
            loss = model_b.bce_loss(pred, yb)
            grad = model_b.bce_grad(pred, yb)
            model_b.backward(grad, lr=lr_b)
            losses.append(loss)

        # Kurze Auswertung
        train_pred = model_b.predict(x_train_b[:400]).reshape(-1)
        train_acc = np.mean(train_pred == y_train_b[:400].reshape(-1))
        test_pred = model_b.predict(x_test_b).reshape(-1)
        test_acc = np.mean(test_pred == y_test_b.reshape(-1))
        print(f"Epoch {epoch}/{epochs_b} | loss={np.mean(losses):.4f} | train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

    # Speichern
    Path('models/task1b').mkdir(parents=True, exist_ok=True)
    model_b.save('models/task1b/car_cnn_scratch.npz')
    print("Gespeichert unter models/task1b/car_cnn_scratch.npz")


if __name__ == '__main__':
    main()
