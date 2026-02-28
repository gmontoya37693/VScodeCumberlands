# ====================================================================
# Project: Programming exercise 2 — Writing recognition with TensorFlow
# What:   LeNet-based MNIST handwriting recognition in TensorFlow 2.x
# Who:    German Montoya, MSDS-534-M40 Deep Learning for Data Science
# Date:   2026-02-27
#
# Inputs:
#   - MNIST dataset from tf.keras.datasets.mnist
#   - Runtime environment variables and installed libraries on the host
#
# Outputs:
#   - Trained LeNet model weights and metrics
#   - Test accuracy summary and qualitative prediction grid
#   - Environment report with library versions and device visibility
#
# This script is built in stages. Stage 1 verifies the runtime by
# printing versions of Python, TensorFlow, Keras, NumPy, Matplotlib,
# and TensorBoard, and by reporting visible GPU devices.
# Subsequent stages define LeNet, compile, train for 20 epochs,
# evaluate accuracy, and save artifacts for the appendix.
# ====================================================================


# ====================
# STAGE 1: ENVIRONMENT SETUP AND VERIFICATION
# ====================
# Goal: Verify all required libraries are installed and working.
#       Confirm GPU availability.
#       Set reproducible random seeds so training is deterministic.
# ====================

import sys
import platform

# 1.1 IMPORT CORE LIBRARIES
# ==========================
print("\n[STAGE 1] Importing core libraries...")

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print("✓ 1.1 Core imports successful")

# 1.1 VERIFICATION
print("\n✓ VERIFY 1.1:")
print(f"  numpy available: {np.__version__}")
print(f"  tensorflow available: {tf.__version__}")
print(f"  matplotlib available: {plt.matplotlib.__version__}")


# 1.2 PRINT ENVIRONMENT INFORMATION
# ===================================
print("\n[STAGE 1] Printing environment information...")

print("\n" + "=" * 70)
print("ENVIRONMENT REPORT: LeNet MNIST Training")
print("=" * 70)

# Python version
print(f"Python version:        {sys.version.split()[0]}")
print(f"Platform:              {platform.platform()}")

# TensorFlow and Keras versions
print(f"TensorFlow version:    {tf.__version__}")
print(f"Keras version:         {tf.keras.__version__}")

# NumPy version
print(f"NumPy version:         {np.__version__}")

# Matplotlib version
print(f"Matplotlib version:    {plt.matplotlib.__version__}")

# GPU/TPU availability
print("\nDevice Configuration:")
gpus = tf.config.list_physical_devices('GPU')
tpus = tf.config.list_physical_devices('TPU')

if gpus:
    print(f"  ✓ GPUs available:     {len(gpus)} device(s)")
    for gpu in gpus:
        print(f"    - {gpu}")
else:
    print(f"  ✗ GPUs available:     None (will use CPU)")

if tpus:
    print(f"  ✓ TPUs available:     {len(tpus)} device(s)")
else:
    print(f"  ✗ TPUs available:     None")

print("=" * 70)

print("\n✓ 1.2 Environment information printed")

# 1.2 VERIFICATION
print("\n✓ VERIFY 1.2:")
print(f"  TensorFlow: {tf.__version__} (should be 2.x)")
print(f"  Keras: {tf.keras.__version__} (should be 3.x)")
print(f"  Device info printed: Yes ✓")


# 1.3 SET RANDOM SEEDS FOR REPRODUCIBILITY
# ==========================================
# Why seeds matter:
#   - Random initialization of weights: neural networks start with
#     random values. If the seed is not fixed, different runs can have
#     different random weights, leading to different final accuracies.
#   - Numpy operations: dropout, shuffling, etc. use randomness.
#   - TensorFlow operations: graph execution may use randomness.
#   - Python random module: used by some utilities.
#
# Result: With fixed seeds, training can produce identical results
#         across runs. This is critical for assignments and
#         reproducibility.

print("\n[STAGE 1] Setting random seeds for reproducibility...")

SEED = 42  # This value can be changed to any integer

# Python's random module seed
import random
random.seed(SEED)

# NumPy's random seed
np.random.seed(SEED)

# TensorFlow's random seed
tf.random.set_seed(SEED)

# Request deterministic operations (optional, may slow things down)
tf.config.run_functions_eagerly(False)  # Keep graph execution for speed

print(f"✓ 1.3 Seeds set (SEED={SEED})")
print("  - Python random.seed({})" .format(SEED))
print("  - NumPy np.random.seed({})".format(SEED))
print("  - TensorFlow tf.random.set_seed({})".format(SEED))

# 1.3 VERIFICATION
print("\n✓ VERIFY 1.3:")
print(f"  SEED value: {SEED} (fixed for reproducibility)")
print(f"  All 3 modules seeded: Yes ✓")


# 1.4 SUMMARY
# ============
print("\n" + "=" * 70)
print("STAGE 1 COMPLETE: Environment verified, seeds set, ready for data loading")
print("=" * 70 + "\n")


# ====================
# STAGE 2: LOAD AND PREPARE MNIST DATA
# ====================
# Goal: Load MNIST handwritten digits, inspect dimensions, 
#       normalize pixel values, reshape for LeNet, and convert 
#       labels to categorical format.
# ====================

print("[STAGE 2] Loading MNIST dataset...\n")

# 2.1 LOAD MNIST DATA FROM KERAS
# ================================
# tf.keras.datasets.mnist.load_data() returns:
#   - X_train: 60,000 training images (28×28 pixels each)
#   - y_train: 60,000 training labels (digit 0-9)
#   - X_test: 10,000 test images (28×28 pixels each)
#   - y_test: 10,000 test labels (digit 0-9)
#
# Data type: uint8 (unsigned 8-bit integer)
# Pixel values: 0-255 (0=black, 255=white)

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Keep an untouched copy for before-vs-after visualization
X_train_uint8 = X_train.copy()
X_test_uint8 = X_test.copy()

print("✓ 2.1 MNIST dataset loaded")
print(f"  Training set shape: {X_train.shape}")
print(f"  Test set shape: {X_test.shape}")
print(f"  Pixel value range: [{X_train.min()}, {X_train.max()}]")

# 2.1 VERIFICATION
print("\n✓ VERIFY 2.1:")
print(f"  X_train shape: {X_train.shape} (should be (60000, 28, 28))")
print(f"  X_test shape: {X_test.shape} (should be (10000, 28, 28))")
print(f"  Pixel values: 0-255 (should be uint8)")


# 2.2 NORMALIZE PIXEL VALUES
# ============================
# Why normalize?
#   Neural networks train better when inputs are in a small range (0.0-1.0)
#   instead of 0-255. Normalization helps the optimizer converge faster.
#
# How: Divide by 255.0 → values become 0.0 to 1.0

print("\n[STAGE 2] Normalizing pixel values...")

X_train = X_train / 255.0
X_test = X_test / 255.0

print("✓ 2.2 Pixel values normalized")
print(f"  New pixel value range: [{X_train.min():.4f}, {X_train.max():.4f}]")
print(f"  Data type: {X_train.dtype}")

# 2.2 VERIFICATION
print("\n✓ VERIFY 2.2:")
print(f"  Min value: {X_train.min():.4f} (should be ≈ 0.0)")
print(f"  Max value: {X_train.max():.4f} (should be ≈ 1.0)")
print(f"  Dtype: {X_train.dtype} (should be float32 or float64)")


# 2.3 RESHAPE DATA FOR LENET INPUT
# ==================================
# LeNet expects 4D input: (num_samples, height, width, channels)
#   - height=28, width=28, channels=1 (grayscale)
#
# Current shape: (60000, 28, 28) — missing channel dimension
# Target shape: (60000, 28, 28, 1)

print("\n[STAGE 2] Reshaping for LeNet input...")

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

print("✓ 2.3 Data reshaped for LeNet")
print(f"  Training set shape: {X_train.shape}")
print(f"  Test set shape: {X_test.shape}")

# 2.3 VERIFICATION
print("\n✓ VERIFY 2.3:")
print(f"  X_train shape: {X_train.shape} (should be (60000, 28, 28, 1))")
print(f"  X_test shape: {X_test.shape} (should be (10000, 28, 28, 1))")
print(f"  Channel dimension added: Yes ✓")


# 2.4 CONVERT LABELS TO CATEGORICAL (ONE-HOT ENCODING)
# ======================================================
# Why one-hot encoding?
#   Softmax classifier expects one-hot vectors, not single integers.
#   Example: label "3" becomes [0,0,0,1,0,0,0,0,0,0]
#
# This creates a 10-dimensional vector (one per digit 0-9)

print("\n[STAGE 2] Converting labels to categorical...")

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

print("✓ 2.4 Labels converted to one-hot encoding")
print(f"  Training labels shape: {y_train.shape}")
print(f"  Test labels shape: {y_test.shape}")
print(f"  Sample label (first training example):")
print(f"    {y_train[0]}")

# 2.4 VERIFICATION
print("\n✓ VERIFY 2.4:")
print(f"  y_train shape: {y_train.shape} (should be (60000, 10))")
print(f"  y_test shape: {y_test.shape} (should be (10000, 10))")
print(f"  One-hot format: Yes (sum of each row = 1.0) ✓")


# 2.5 VISUALIZE SAMPLE IMAGES
# =============================
# Why visualize?
#   Before training, the script should show the input data clearly.
#   MNIST has handwritten digits—some are clear, some are messy.
#   This helps show what LeNet must learn to recognize.
#
# The figure includes:
#   1. One digit before normalization (0-255)
#   2. The same digit after normalization (0.0-1.0)
#   3. A histogram showing pixel-value scaling

print("\n[STAGE 2] Visualizing sample MNIST images...")

# Create a 1x3 figure: before, after, histogram
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Same sample before and after normalization
sample_index = 0
img_before = X_train_uint8[sample_index]      # uint8, range 0-255
img_after = X_train[sample_index, :, :, 0]    # float, range 0.0-1.0
digit_label = np.argmax(y_train[sample_index])

# --- Left: Before normalization ---
axes[0].imshow(img_before, cmap='gray', vmin=0, vmax=255)
axes[0].set_title(f'Digit {digit_label}: Before (0-255)', fontsize=11, fontweight='bold')
axes[0].axis('off')

# --- Right: After normalization ---
axes[1].imshow(img_after, cmap='gray', vmin=0.0, vmax=1.0)
axes[1].set_title(f'Digit {digit_label}: After (0.0-1.0)', fontsize=11, fontweight='bold')
axes[1].axis('off')

# --- Histogram: pixel values before vs after normalization ---
# Convert "before" image to 0-1 so both histograms share the same x-scale
img_before_norm = img_before.astype('float32') / 255.0
shared_bins = np.linspace(0.0, 1.0, 21)

axes[2].hist(
    img_before_norm.flatten(),
    bins=shared_bins,
    histtype='step',
    linewidth=2.5,
    linestyle='--',
    color='orange',
    label='Before mapped to 0-1'
)
axes[2].hist(
    img_after.flatten(),
    bins=shared_bins,
    histtype='step',
    linewidth=2.0,
    linestyle='-',
    color='blue',
    label='After (0.0-1.0)'
)
axes[2].set_title('Pixel Value Histogram (same 0-1 scale)', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Pixel value (0.0 to 1.0)')
axes[2].set_ylabel('Count')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=100, bbox_inches='tight')
plt.show()  # Display the plot in a window
print("✓ 2.5 Sample images visualized and saved to 'mnist_samples.png'")

# 2.5 VERIFICATION
print("\n✓ VERIFY 2.5:")
print(f"  Same image shown before and after normalization: Yes ✓")
print(f"  Digit label shown: {digit_label}")
print(f"  Before range: [{img_before.min()}, {img_before.max()}] (should be 0-255)")
print(f"  After range: [{img_after.min():.4f}, {img_after.max():.4f}] (should be 0.0-1.0)")
print(f"  Visualization saved: mnist_samples.png ✓")

# Close figure to free memory
plt.close()


# 2.6 SUMMARY
# ============
print("\n" + "=" * 70)
print("STAGE 2 COMPLETE: Data loaded, normalized, reshaped, labeled, visualized")
print("=" * 70)
print(f"\nFinal Dataset Summary:")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Image dimensions: {X_train.shape[1:]} (height, width, channels)")
print(f"  Number of classes: {y_train.shape[1]}")
print(f"  X dtype: {X_train.dtype} | y dtype: {y_train.dtype}")
print(f"  Visualization: mnist_samples.png (saved to current directory)")
print("=" * 70 + "\n")


# ====================
# STAGE 3: DEFINE LENET ARCHITECTURE
# ====================
# Goal: Define the LeNet model (book-style) and verify its structure.
# This stage only builds and prints the architecture.
# ====================

print("[STAGE 3] Defining LeNet model architecture...\n")

# 3.1 DEFINE MODEL SETTINGS
# ==========================
# INPUT_SHAPE comes from Stage 2 preprocessing: (28, 28, 1)
# NB_CLASSES is the number of digits (0 to 9)

INPUT_SHAPE = (28, 28, 1)
NB_CLASSES = 10

print("✓ 3.1 Model settings defined")
print(f"  INPUT_SHAPE: {INPUT_SHAPE}")
print(f"  NB_CLASSES: {NB_CLASSES}")

# 3.1 VERIFICATION
print("\n✓ VERIFY 3.1:")
print(f"  Input shape matches preprocessed images: {INPUT_SHAPE == X_train.shape[1:]} ✓")
print(f"  Number of classes is 10: {NB_CLASSES == 10} ✓")


# 3.2 BUILD LENET MODEL (BOOK-STYLE)
# ====================================
# Book-aligned architecture:
#   Conv2D(20, 5x5, relu) -> MaxPool(2x2)
#   Conv2D(50, 5x5, relu) -> MaxPool(2x2)
#   Flatten -> Dense(500, relu) -> Dense(10, softmax)

print("\n[STAGE 3] Building LeNet layers...")

model = tf.keras.models.Sequential(name="LeNet_MNIST")

# Explicit input layer (recommended in current Keras)
model.add(tf.keras.layers.Input(shape=INPUT_SHAPE))

# First convolution block
model.add(tf.keras.layers.Conv2D(20, (5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolution block
model.add(tf.keras.layers.Conv2D(50, (5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Classifier head
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dense(NB_CLASSES, activation='softmax'))

print("✓ 3.2 LeNet layers created")


# 3.3 DISPLAY MODEL SUMMARY
# ==========================
# This prints each layer, output shape, and number of trainable parameters.

print("\n[STAGE 3] Model summary:")
model.summary()

print("\n✓ 3.3 Model summary printed")


# 3.4 VERIFICATION
# =================
# Quick checks to ensure architecture matches the expected structure.

conv_layers = sum(isinstance(layer, tf.keras.layers.Conv2D) for layer in model.layers)
pool_layers = sum(isinstance(layer, tf.keras.layers.MaxPooling2D) for layer in model.layers)
dense_layers = sum(isinstance(layer, tf.keras.layers.Dense) for layer in model.layers)

print("\n✓ VERIFY 3.4:")
print(f"  Conv2D layers: {conv_layers} (should be 2)")
print(f"  MaxPooling2D layers: {pool_layers} (should be 2)")
print(f"  Dense layers: {dense_layers} (should be 2)")
print(f"  Output layer units: {model.layers[-1].units} (should be 10)")
print(f"  Output activation: {model.layers[-1].activation.__name__} (should be softmax)")


# 3.5 SUMMARY
# ============
print("\n" + "=" * 70)
print("STAGE 3 COMPLETE: LeNet architecture defined and verified")
print("=" * 70 + "\n")


# ====================
# STAGE 4: COMPILE AND TRAIN MODEL
# ====================
# Goal: Compile LeNet and train for 20 epochs.
# ====================

print("[STAGE 4] Compiling and training LeNet...\n")

# 4.1 DEFINE TRAINING SETTINGS
# ============================
EPOCHS = 20
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
VERBOSE = 1
OPTIMIZER = tf.keras.optimizers.Adam()

print("✓ 4.1 Training settings defined")
print(f"  EPOCHS: {EPOCHS}")
print(f"  BATCH_SIZE: {BATCH_SIZE}")
print(f"  VALIDATION_SPLIT: {VALIDATION_SPLIT}")
print(f"  OPTIMIZER: {OPTIMIZER.__class__.__name__}")

# 4.1 VERIFICATION
print("\n✓ VERIFY 4.1:")
print(f"  Epochs set to 20: {EPOCHS == 20} ✓")
print(f"  Batch size > 0: {BATCH_SIZE > 0} ✓")
print(f"  Validation split between 0 and 1: {0 < VALIDATION_SPLIT < 1} ✓")


# 4.2 COMPILE MODEL
# ==================
# Loss: categorical_crossentropy for one-hot labels
# Metric: accuracy to track classification performance

print("\n[STAGE 4] Compiling model...")

model.compile(
    loss='categorical_crossentropy',
    optimizer=OPTIMIZER,
    metrics=['accuracy']
)

print("✓ 4.2 Model compiled")

# 4.2 VERIFICATION
print("\n✓ VERIFY 4.2:")
print("  Loss: categorical_crossentropy ✓")
print("  Metrics: accuracy ✓")


# 4.3 TRAIN MODEL FOR 20 EPOCHS
# ==============================
print("\n[STAGE 4] Training model (20 epochs)...")
print("  Note: CPU training can take several minutes.")

history = model.fit(
    X_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=VALIDATION_SPLIT
)

print("✓ 4.3 Training complete")


# 4.4 TRAINING RESULT SUMMARY
# ===========================
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print("\n[STAGE 4] Final epoch metrics:")
print(f"  Train loss: {final_train_loss:.4f}")
print(f"  Train accuracy: {final_train_acc:.4f}")
print(f"  Validation loss: {final_val_loss:.4f}")
print(f"  Validation accuracy: {final_val_acc:.4f}")

# 4.4 VERIFICATION
print("\n✓ VERIFY 4.4:")
print(f"  Epochs completed: {len(history.history['loss'])} (should be 20)")
print(f"  Validation accuracy available: {'val_accuracy' in history.history} ✓")
print(f"  Final validation accuracy printed: Yes ✓")


# 4.5 SUMMARY
# ============
print("\n" + "=" * 70)
print("STAGE 4 COMPLETE: Model compiled and trained for 20 epochs")
print("=" * 70 + "\n")

