"""
MSDS-534-M40 Deep Learning (Spring 2026)
Residency Day 2 - Project 2
Progressive Network Optimization and Hyperparameter Tuning

Single-script workflow:
1) Set RUN_VERSION to one of: V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11
2) Run this file
3) Capture terminal screenshot for that run
4) Repeat for required versions

This script logs each run into mnist_results.json and prints a comparison table.
"""

import json
import math
import os

import keras
import numpy as np
import tensorflow as tf


# --------------------------- USER CONTROL ---------------------------
RUN_VERSION = "V11"
# -------------------------------------------------------------------


def get_config(version):
    base = {
        "N_HIDDEN": 128,
        "EPOCHS": 20,
        "BATCH_SIZE": 128,
        "VALIDATION_SPLIT": 0.20,
        "DROPOUT": None,
        "DEPTH": 1,
        "OPTIMIZER": "sgd",
        "DESCRIPTION": "Baseline - 1 hidden layer, SGD",
    }

    variants = {
        "V1": dict(base),
        "V2": dict(base, DEPTH=2, DESCRIPTION="Added depth - 2 hidden layers, SGD"),
        "V3": dict(base, DROPOUT=0.30, DESCRIPTION="Dropout regularization - rate 0.30"),
        "V4": dict(base, DEPTH=2, DROPOUT=0.30, DESCRIPTION="Deeper + Dropout(0.30)"),
        "V5": dict(base, DROPOUT=0.50, DESCRIPTION="Stronger dropout - rate 0.50"),
        "V6": dict(base, DROPOUT=0.50, EPOCHS=30, DESCRIPTION="V3 tuned: dropout 0.50, epochs 30"),
        "V7": dict(base, DROPOUT=0.30, OPTIMIZER="adam", DESCRIPTION="Optimizer comparison baseline - Adam"),
        "V8": dict(base, DROPOUT=0.30, OPTIMIZER="sgd", DESCRIPTION="Optimizer comparison - standard SGD"),
        "V9": dict(base, N_HIDDEN=2048, DROPOUT=0.30, OPTIMIZER="adam", DESCRIPTION="Capacity stress test - hidden 2048"),
        "V10": dict(base, DROPOUT=0.30, OPTIMIZER="rmsprop", BATCH_SIZE=256, DESCRIPTION="Optimizer/batch variant - RMSprop + larger batch"),
        "V11": dict(
            base,
            N_HIDDEN=256,
            DEPTH=2,
            DROPOUT=0.4,
            OPTIMIZER="adam",
            BATCH_SIZE=64,
            EPOCHS=25,
            DESCRIPTION="Custom best model target >=98% val accuracy",
        ),
    }

    if version not in variants:
        raise ValueError("RUN_VERSION must be one of V1..V11")
    return variants[version]


def build_model(cfg):
    layers = [keras.Input(shape=(28, 28)), keras.layers.Flatten()]

    for _ in range(cfg["DEPTH"]):
        layers.append(keras.layers.Dense(cfg["N_HIDDEN"], activation="relu"))

    if cfg["DROPOUT"] is not None:
        layers.append(keras.layers.Dropout(cfg["DROPOUT"]))

    layers.append(keras.layers.Dense(10, activation="softmax"))
    return keras.models.Sequential(layers)


def print_run_explanation(version, cfg, x_train):
    print("=" * 80)
    print("MNIST Progressive Optimization")
    print(f"Run version: {version} | {cfg['DESCRIPTION']}")
    print("=" * 80)

    print("\n[HYPERPARAMETERS]")
    print(f"N_HIDDEN={cfg['N_HIDDEN']}, DEPTH={cfg['DEPTH']}, DROPOUT={cfg['DROPOUT']}")
    print(f"OPTIMIZER={cfg['OPTIMIZER']}, EPOCHS={cfg['EPOCHS']}, BATCH_SIZE={cfg['BATCH_SIZE']}")
    print(f"VALIDATION_SPLIT={cfg['VALIDATION_SPLIT']}")

    print("\n[LOAD & NORMALIZATION]")
    print(f"Input images shape: {x_train.shape[1]}x{x_train.shape[2]} (grayscale)")
    print(f"Flatten operation: 28x28 -> 784 numeric features per image")
    print("Normalization: pixel values scaled from [0,255] to [0,1] by dividing by 255.0")

    print("\n[MODEL DETAILS]")
    print("Input: (28,28)")
    print("Flatten: outputs 784 numbers")
    print(f"Hidden Dense layers: {cfg['DEPTH']} layer(s), {cfg['N_HIDDEN']} units each, activation='relu'")
    if cfg["DROPOUT"] is not None:
        print(f"Dropout layer: rate={cfg['DROPOUT']}")
    print("Output Dense: 10 units, activation='softmax' (digit classes 0-9)")

    hidden_params_single = (784 * cfg["N_HIDDEN"]) + cfg["N_HIDDEN"]
    if cfg["DEPTH"] == 1:
        hidden_total = hidden_params_single
    else:
        hidden_total = hidden_params_single + ((cfg["N_HIDDEN"] * cfg["N_HIDDEN"]) + cfg["N_HIDDEN"])
    output_params = (cfg["N_HIDDEN"] * 10) + 10
    print("Parameter view (dense layers)")
    print(f"Hidden params approx: {hidden_total:,}")
    print(f"Output params: {output_params:,}")

    steps = math.ceil((x_train.shape[0] * (1 - cfg["VALIDATION_SPLIT"])) / cfg["BATCH_SIZE"])
    print("\n[COMPILE & TRAIN PLAN]")
    print(f"Loss: sparse_categorical_crossentropy | Metric: accuracy")
    print(f"Optimizer: {cfg['OPTIMIZER']}")
    print(f"Epochs: {cfg['EPOCHS']} | Steps per epoch: {steps}")


def print_table(results):
    if not results:
        return

    results = sorted(results, key=lambda r: int(r["version"].replace("V", "")))
    print("\n" + "=" * 118)
    print("COMPARISON TABLE (all runs logged so far)")
    print("=" * 118)
    hdr = (
        f"{'Ver':<5} {'Description':<44} {'Opt':<8} {'Drop':>6} {'Batch':>6} "
        f"{'Epochs':>6} {'TrainAcc':>9} {'ValAcc':>8} {'TestAcc':>8} {'TrainLoss':>10} {'ValLoss':>8} {'Params':>10}"
    )
    print(hdr)
    print("-" * 118)

    for r in results:
        drop = f"{r['dropout']:.2f}" if r["dropout"] is not None else "-"
        row = (
            f"{r['version']:<5} {r['description']:<44} {r['optimizer']:<8} {drop:>6} {r['batch_size']:>6} "
            f"{r['epochs']:>6} {r['train_acc']:>9.4f} {r['val_acc']:>8.4f} {r['test_acc']:>8.4f} "
            f"{r['train_loss']:>10.4f} {r['val_loss']:>8.4f} {r['total_params']:>10,}"
        )
        print(row)

    best = max(results, key=lambda r: r["val_acc"])
    print("-" * 118)
    print(f"Best validation accuracy so far: {best['version']} -> {best['val_acc']*100:.2f}%")
    print("=" * 118)


def main():
    # Fixed seeds for consistent comparisons across reruns in the same environment
    np.random.seed(42)
    tf.random.set_seed(42)

    cfg = get_config(RUN_VERSION)

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print_run_explanation(RUN_VERSION, cfg, x_train)

    model = build_model(cfg)
    print("\n[MODEL SUMMARY]")
    model.summary()

    model.compile(
        optimizer=cfg["OPTIMIZER"],
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n[TRAINING OUTPUT]")
    history = model.fit(
        x_train,
        y_train,
        epochs=cfg["EPOCHS"],
        batch_size=cfg["BATCH_SIZE"],
        validation_split=cfg["VALIDATION_SPLIT"],
        verbose=1,
    )

    print("\n[EVALUATION OUTPUT]")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    train_acc = history.history["accuracy"][-1]
    val_acc = history.history["val_accuracy"][-1]
    train_loss = history.history["loss"][-1]
    val_loss = history.history["val_loss"][-1]

    print(f"Final train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Final val   accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"Final test  accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Final train loss: {train_loss:.4f}")
    print(f"Final val   loss: {val_loss:.4f}")
    print(f"Final test  loss: {test_loss:.4f}")

    print("\nHow to see required outputs:")
    print("1) Training output is printed epoch-by-epoch under [TRAINING OUTPUT].")
    print("2) Final metrics are printed under [EVALUATION OUTPUT].")
    print("3) Use terminal screenshots of those sections for your report.")

    log_file = os.path.join(os.path.dirname(__file__), "mnist_results.json")
    entry = {
        "version": RUN_VERSION,
        "description": cfg["DESCRIPTION"],
        "n_hidden": cfg["N_HIDDEN"],
        "depth": cfg["DEPTH"],
        "epochs": cfg["EPOCHS"],
        "batch_size": cfg["BATCH_SIZE"],
        "optimizer": str(cfg["OPTIMIZER"]).upper(),
        "dropout": cfg["DROPOUT"],
        "train_acc": round(train_acc, 4),
        "val_acc": round(val_acc, 4),
        "test_acc": round(test_acc, 4),
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "total_params": int(model.count_params()),
    }

    results = []
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            results = json.load(f)

    results = [r for r in results if r["version"] != RUN_VERSION]
    results.append(entry)
    results = sorted(results, key=lambda r: int(r["version"].replace("V", "")))

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nRun logged to: {log_file}")
    print_table(results)


if __name__ == "__main__":
    main()
