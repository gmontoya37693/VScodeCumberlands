# regression_vs_classification_offline.py
# One file, no downloads: synthetic regression + synthetic classification.
# Prints [Regression] Test MAE: ... and [Classification] Test Accuracy: ...

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # keep TF logs minimal

import numpy as np
import tensorflow as tf

def run_regression():
    print("\n=== REGRESSION (synthetic) ===")
    rng = np.random.default_rng(7)
    n, d = 1200, 8
    X = rng.normal(size=(n, d)).astype("float32")
    true_w = rng.normal(size=(d, 1)).astype("float32")
    true_b = np.array([[2.0]], dtype="float32")
    y = X @ true_w + true_b + 0.1 * rng.normal(size=(n, 1)).astype("float32")

    # Train/val/test split
    idx = rng.permutation(n)
    X_train, y_train = X[idx[:900]],  y[idx[:900]]
    X_val,   y_val   = X[idx[900:1050]], y[idx[900:1050]]
    X_test,  y_test  = X[idx[1050:]],    y[idx[1050:]]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(d,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1)  # linear output
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.summary()

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=10, batch_size=32, verbose=1)

    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    print(f"[Regression] Test MAE: {test_mae:.4f}")

def run_classification():
    print("\n=== CLASSIFICATION (synthetic) ===")
    rng = np.random.default_rng(11)

    # Two-class problem in 2D, linearly separable with noise
    n_per_class = 2000
    mean_pos, mean_neg = np.array([1.5, 1.5]), np.array([-1.5, -1.5])
    cov = np.array([[0.6, 0.0], [0.0, 0.6]])

    X_pos = rng.multivariate_normal(mean_pos, cov, size=n_per_class).astype("float32")
    X_neg = rng.multivariate_normal(mean_neg, cov, size=n_per_class).astype("float32")
    X = np.vstack([X_pos, X_neg]).astype("float32")
    y = np.hstack([np.ones(n_per_class, dtype="int32"), np.zeros(n_per_class, dtype="int32")])

    # Shuffle and split
    idx = rng.permutation(X.shape[0])
    X, y = X[idx], y[idx]
    X_train, y_train = X[:3200], y[:3200]
    X_val,   y_val   = X[3200:3600], y[3200:3600]
    X_test,  y_test  = X[3600:], y[3600:]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")  # binary logistic
    ])
    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=5, batch_size=64, verbose=1)

    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[Classification] Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    # Reproducibility
    tf.keras.utils.set_random_seed(7)

    # If the environment enforces deterministic ops, keep performance decent
    try:
        tf.config.experimental.enable_op_determinism = False
    except Exception:
        pass

    run_regression()
    run_classification()