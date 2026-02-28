"""
================================================================================
                         LeNet-5 MNIST CLASSIFIER
================================================================================

WHAT:
  Convolutional Neural Network (CNN) implementation using LeNet-5 architecture
  trained on the MNIST dataset for handwritten digit recognition.

WHO:
  Cumberlands University - Course Assignment
  2026 Spring - Deep Learning (MSDS-534-M40) - Full Term
  Date: February 2026

OBJECTIVES:
  1. Build a LeNet-5 CNN model (2 Conv layers with MaxPooling, 2 Dense layers)
  2. Train on MNIST dataset (60,000 training samples) using eager execution
  3. Evaluate on test set (10,000 samples) and report accuracy
  4. Visualize predictions on a grid of test samples
  5. Demonstrate TensorFlow eager mode training with manual gradient computation

INPUT:
  - MNIST dataset (automatically downloaded via tf.keras.datasets.mnist)
    ~ Training: 60,000 images of 28×28 pixels (grayscale)
    ~ Testing: 10,000 images of 28×28 pixels (grayscale)
    ~ Labels: 10 classes (digits 0-9)
  - No external files required; data is auto-cached

OUTPUT:
  - Console output:
    ~ Dependency versions at startup
    ~ Training metrics per epoch (loss, accuracy)
    ~ Validation metrics per epoch
    ~ Final test set accuracy
    - Files (generated during execution):
    ~ metrics.json: Training/validation/test accuracies
        ~ mnist_normalization_comparison.png: Before/after normalization view
        ~ mnist_learning_curves.png: Loss and accuracy across epochs
        ~ mnist_confusion_matrix.png: Class-wise prediction matrix
        ~ mnist_misclassifications.png: Common model mistakes with confidence
    ~ mnist_predictions_grid.png: 5×5 grid of predictions vs. true labels

NOTES:
  - Uses TensorFlow 2.x eager execution (avoids graph-mode crashes on ARM64)
  - Manual training loop with tf.GradientTape for gradient computation
  - No model.fit() or tf.data pipeline (known incompatibility on macOS ARM64)
  - Expected final test accuracy: ~99%

================================================================================
"""

# ====================
# DEPENDENCY CHECK
# ====================
import sys

required_packages = {
    'json': None,
    'random': None,
    'numpy': 'np',
    'tensorflow': 'tf',
    'matplotlib': None,
}

print("Checking dependencies...")
missing = []

for package, import_name in required_packages.items():
    try:
        if import_name:
            mod = __import__(package)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {package}: {version}")
        else:
            __import__(package)
            print(f"  ✓ {package}: installed")
    except ImportError:
        print(f"  ✗ {package}: NOT FOUND")
        missing.append(package)

if missing:
    print(f"\nERROR: Missing packages: {', '.join(missing)}")
    sys.exit(1)

print("\nAll dependencies OK.\n")

import json
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ====================
# STAGE 1: SETUP
# ====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"TensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")


# ====================
# STAGE 2: DATA
# ====================
(train_images_raw, y_train_raw), (test_images_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()

X_train = train_images_raw.astype("float32") / 255.0
X_test = test_images_raw.astype("float32") / 255.0

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

y_train = tf.keras.utils.to_categorical(y_train_raw, 10).astype("float32")
y_test = tf.keras.utils.to_categorical(y_test_raw, 10).astype("float32")

print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test:  {X_test.shape}, {y_test.shape}")


# ====================
# STAGE 2.1: NORMALIZATION VIEW
# ====================
norm_idx = 0

fig, ax = plt.subplots(2, 2, figsize=(8, 6))

ax[0, 0].imshow(train_images_raw[norm_idx], cmap="gray")
ax[0, 0].set_title(f"Raw image (label={y_train_raw[norm_idx]})")
ax[0, 0].axis("off")

ax[0, 1].imshow(X_train[norm_idx].squeeze(), cmap="gray", vmin=0.0, vmax=1.0)
ax[0, 1].set_title("Normalized image [0,1]")
ax[0, 1].axis("off")

ax[1, 0].hist(train_images_raw[norm_idx].ravel(), bins=30, color="steelblue")
ax[1, 0].set_title("Raw pixel histogram")
ax[1, 0].set_xlabel("Pixel value")
ax[1, 0].set_ylabel("Count")

ax[1, 1].hist(X_train[norm_idx].ravel(), bins=30, color="seagreen")
ax[1, 1].set_title("Normalized pixel histogram")
ax[1, 1].set_xlabel("Pixel value")
ax[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("mnist_normalization_comparison.png", dpi=130, bbox_inches="tight")
plt.show()
plt.close()


# ====================
# STAGE 3: MODEL (LeNet)
# ====================
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(20, (5, 5), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(50, (5, 5), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ],
    name="LeNet_MNIST",
)

print("LeNet created")


# ====================
# STAGE 4: TRAIN (EAGER LOOP)
# ====================
EPOCHS = 20
BATCH_SIZE = 128
VAL_SPLIT = 0.1

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()

val_size = int(len(X_train) * VAL_SPLIT)
X_val = X_train[-val_size:]
y_val = y_train[-val_size:]
X_tr = X_train[:-val_size]
y_tr = y_train[:-val_size]

history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

for epoch in range(EPOCHS):
    idx = np.random.permutation(len(X_tr))
    X_tr_shuf = X_tr[idx]
    y_tr_shuf = y_tr[idx]

    batch_losses = []
    batch_correct = 0
    batch_total = 0

    for start in range(0, len(X_tr_shuf), BATCH_SIZE):
        end = start + BATCH_SIZE
        x_batch = tf.convert_to_tensor(X_tr_shuf[start:end], dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_tr_shuf[start:end], dtype=tf.float32)

        with tf.GradientTape() as tape:
            pred = model(x_batch, training=True)
            loss = loss_fn(y_batch, pred)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        batch_losses.append(float(loss.numpy()))
        pred_labels = tf.argmax(pred, axis=1).numpy()
        true_labels = tf.argmax(y_batch, axis=1).numpy()
        batch_correct += int(np.sum(pred_labels == true_labels))
        batch_total += len(true_labels)

    train_loss = float(np.mean(batch_losses))
    train_acc = batch_correct / batch_total

    val_losses = []
    val_correct = 0
    val_total = 0
    for start in range(0, len(X_val), BATCH_SIZE):
        end = start + BATCH_SIZE
        x_batch = tf.convert_to_tensor(X_val[start:end], dtype=tf.float32)
        y_batch = tf.convert_to_tensor(y_val[start:end], dtype=tf.float32)

        pred = model(x_batch, training=False)
        loss = loss_fn(y_batch, pred)
        val_losses.append(float(loss.numpy()))

        pred_labels = tf.argmax(pred, axis=1).numpy()
        true_labels = tf.argmax(y_batch, axis=1).numpy()
        val_correct += int(np.sum(pred_labels == true_labels))
        val_total += len(true_labels)

    val_loss = float(np.mean(val_losses))
    val_acc = val_correct / val_total

    history["loss"].append(train_loss)
    history["accuracy"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"loss={train_loss:.4f} acc={train_acc:.4f} "
        f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
    )


# ====================
# STAGE 5: TEST EVAL
# ====================
test_losses = []
test_correct = 0
test_total = 0
all_test_probs = []
all_test_pred_labels = []
all_test_true_labels = []

for start in range(0, len(X_test), BATCH_SIZE):
    end = start + BATCH_SIZE
    x_batch = tf.convert_to_tensor(X_test[start:end], dtype=tf.float32)
    y_batch = tf.convert_to_tensor(y_test[start:end], dtype=tf.float32)

    pred = model(x_batch, training=False)
    pred_np = pred.numpy()
    loss = loss_fn(y_batch, pred)
    test_losses.append(float(loss.numpy()))

    pred_labels = np.argmax(pred_np, axis=1)
    true_labels = tf.argmax(y_batch, axis=1).numpy()
    test_correct += int(np.sum(pred_labels == true_labels))
    test_total += len(true_labels)

    all_test_probs.append(pred_np)
    all_test_pred_labels.append(pred_labels)
    all_test_true_labels.append(true_labels)

test_loss = float(np.mean(test_losses))
test_acc = test_correct / test_total

all_test_probs = np.concatenate(all_test_probs, axis=0)
all_test_pred_labels = np.concatenate(all_test_pred_labels, axis=0)
all_test_true_labels = np.concatenate(all_test_true_labels, axis=0)

metrics = {
    "final_train_accuracy": history["accuracy"][-1],
    "final_val_accuracy": history["val_accuracy"][-1],
    "test_loss": test_loss,
    "test_accuracy": test_acc,
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Test loss: {test_loss:.4f}")
print(f"Test acc:  {test_acc:.4f}")


# ====================
# STAGE 5.1: LEARNING CURVES
# ====================
epochs_axis = np.arange(1, EPOCHS + 1)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(epochs_axis, history["loss"], marker="o", label="Train Loss")
plt.plot(epochs_axis, history["val_loss"], marker="o", label="Val Loss")
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_axis, history["accuracy"], marker="o", label="Train Acc")
plt.plot(epochs_axis, history["val_accuracy"], marker="o", label="Val Acc")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("mnist_learning_curves.png", dpi=130, bbox_inches="tight")
plt.show()
plt.close()


# ====================
# STAGE 5.2: CONFUSION MATRIX
# ====================
conf_mat = np.zeros((10, 10), dtype=np.int32)
np.add.at(conf_mat, (all_test_true_labels, all_test_pred_labels), 1)

plt.figure(figsize=(7, 6))
plt.imshow(conf_mat, cmap="Blues")
plt.title("MNIST Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(np.arange(10))
plt.yticks(np.arange(10))
plt.colorbar()

for i in range(10):
    for j in range(10):
        text_color = "white" if conf_mat[i, j] > conf_mat.max() * 0.5 else "black"
        plt.text(j, i, str(conf_mat[i, j]), ha="center", va="center", fontsize=8, color=text_color)

plt.tight_layout()
plt.savefig("mnist_confusion_matrix.png", dpi=130, bbox_inches="tight")
plt.show()
plt.close()


# ====================
# STAGE 5.3: MISCLASSIFICATIONS
# ====================
mis_idx = np.where(all_test_pred_labels != all_test_true_labels)[0]
show_n = min(25, len(mis_idx))

if show_n > 0:
    selected = np.random.choice(mis_idx, size=show_n, replace=False)
    plt.figure(figsize=(8, 8))
    for i, idx in enumerate(selected):
        plt.subplot(5, 5, i + 1)
        plt.imshow(X_test[idx].squeeze(), cmap="gray")
        conf = float(np.max(all_test_probs[idx]))
        plt.title(
            f"P:{all_test_pred_labels[idx]} T:{all_test_true_labels[idx]}\nC:{conf:.2f}",
            fontsize=7,
            color="red",
        )
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("mnist_misclassifications.png", dpi=130, bbox_inches="tight")
    plt.show()
    plt.close()


# ====================
# STAGE 6: PRED GRID
# ====================
sample_idx = np.random.choice(len(X_test), size=25, replace=False)
sample_x = X_test[sample_idx]
sample_true = y_test_raw[sample_idx]

sample_pred = model(tf.convert_to_tensor(sample_x, dtype=tf.float32), training=False).numpy()
sample_pred_labels = np.argmax(sample_pred, axis=1)

plt.figure(figsize=(8, 8))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(sample_x[i].squeeze(), cmap="gray")
    color = "green" if sample_pred_labels[i] == sample_true[i] else "red"
    plt.title(f"P:{sample_pred_labels[i]} T:{sample_true[i]}", fontsize=8, color=color)
    plt.axis("off")

plt.tight_layout()
plt.savefig("mnist_predictions_grid.png", dpi=130, bbox_inches="tight")
plt.show()
plt.close()

print("Saved: metrics.json")
print("Saved: mnist_normalization_comparison.png")
print("Saved: mnist_learning_curves.png")
print("Saved: mnist_confusion_matrix.png")
if show_n > 0:
    print("Saved: mnist_misclassifications.png")
print("Saved: mnist_predictions_grid.png")
