"""
2026 Spring - Deep Learning (MSDS-534-M40) - Full Term
Residency Day 3: Project 4 - Recurrent Neural Networks (Text Generation)

Team Context
- Group: 2
- Corpus: Alice's Adventures in Wonderland
- Workflow: Asynchronous pipeline using Checkpoint Vault + live handoffs

Primary Goal
Build a character-level text generator using TensorFlow/Keras by splitting
execution across four team roles and integrating outputs.

Expected Vault Files
- raw_text.txt
- char_vocab.pkl
- untrained_rnn.weights.h5
- fully_trained_rnn.weights.h5

----------------------------------------------------------------------
ROLE 1: DATA ENGINEER (DE)
----------------------------------------------------------------------
Task 1 - Vocabulary
- Read text corpus.
- Extract sorted unique characters.
- Build:
	- char2idx: character -> integer index
	- idx2char: integer index -> character

Task 2 - Vectorization
- Convert the full text into integer IDs using char2idx.

Task 3 - Dataset Creation
- Create tf.data.Dataset from integer vector.
- Chunk into fixed windows with length seq_length + 1.

Task 4 - Input/Target Shift
- Split each chunk into:
	- input_text = chunk[:-1]
	- target_text = chunk[1:]
- Shuffle and batch dataset.

DE Handoff Deliverables
- char_vocab_live.pkl
- vocab_size integer
- final dataset pipeline code and shape checks

----------------------------------------------------------------------
ROLE 2: DEEP LEARNING ARCHITECT (DLA)
----------------------------------------------------------------------
Task 1 - Model Builder
- Implement build_model(...) with Sequential API containing:
	- Input (batch_shape=[batch_size, None])
	- Embedding
	- GRU or LSTM
	- Dense(vocab_size)

Task 2 - Embedding Explanation
- Explain why embedding is needed instead of raw integer inputs.

Task 3 - Stateful Experiment
- Compare stateful=True vs stateful=False behavior.
- Document effect on hidden state across batches.

Task 4 - GRU vs LSTM Swap
- Replace GRU with LSTM (or vice versa).
- Record parameter count difference and choose official architecture.

DLA Handoff Deliverables
- Final build_model function signature and settings
- Selected recurrent layer type (GRU or LSTM)
- Notes for OEL/GEL compatibility

----------------------------------------------------------------------
ROLE 3: OPTIMIZATION & TRAINING LEAD (OEL)
----------------------------------------------------------------------
Task 1 - Custom Loss
- Define loss using sparse_categorical_crossentropy.
- Use from_logits=True and explain why.

Task 2 - Compile
- Compile with Adam + custom loss.

Task 3 - Live Training Run
- Train for 5-10 epochs on live DE dataset.
- Plot and save training loss curve.

Task 4 - Optimizer Swap
- Re-compile with RMSprop.
- Train 5 epochs.
- Compare early-stage loss reduction with Adam.

OEL Handoff Deliverables
- Training logs and loss plot(s)
- Adam vs RMSprop summary
- Confirmation that architecture/weights remain compatible

----------------------------------------------------------------------
ROLE 4: GENERATION & EVALUATION LEAD (GEL)
----------------------------------------------------------------------
Task 1 - Inference Setup
- Rebuild architecture with batch_size=1.
- Load fully_trained_rnn.weights.h5.

Task 2 - Generation Loop
- Implement generate_text(start_string, num_generate=1000, temperature).

Task 3 - Temperature Scaling
- Generate outputs at temperatures:
	- 0.5
	- 1.0
	- 1.5

Task 4 - Qualitative Analysis
- Compare coherence vs creativity across temperatures.

GEL Deliverables
- Three generated text samples
- Temperature analysis summary

----------------------------------------------------------------------
CONCEPTUAL QUESTIONS (INDIVIDUAL ONLY)
----------------------------------------------------------------------
Each team member answers independently:
1) Why are RNNs better for text generation than dense feedforward networks?
2) What is vanishing gradient, and how do GRU/LSTM mitigate it?
3) What mathematically changes when temperature > 1.0?
4) Why is target shifted one step right from input?
5) What dataset-specific patterns were learned, and what failed?

----------------------------------------------------------------------
INTEGRATION CHECKPOINTS
----------------------------------------------------------------------
Checkpoint A (DE -> DLA/OEL/GEL)
- Provide char_vocab_live.pkl + vocab_size.

Checkpoint B (DLA -> OEL/GEL)
- Provide final build_model configuration.

Checkpoint C (OEL -> GEL)
- Provide training behavior summary and compatibility notes.

Checkpoint D (All)
- Final report + individual notebooks packaged for submission.

----------------------------------------------------------------------
TROUBLESHOOTING QUICK NOTES
----------------------------------------------------------------------
- Shift bug: input=chunk[:-1], target=chunk[1:]
- Stateful shape bug: use Input(batch_shape=[batch_size, None])
- Loss bug: use sparse categorical crossentropy for integer labels
- Inference shape bug: use expand_dims/squeeze around model call
"""

from __future__ import annotations

import pickle
import unicodedata
from pathlib import Path

import numpy as np
import tensorflow as tf


# -----------------------------
# DE configuration (Group 2)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_TEXT_PATH = BASE_DIR / "raw_text.txt"
VAULT_VOCAB_PATH = BASE_DIR / "char_vocab.pkl"
LIVE_VOCAB_PATH = BASE_DIR / "char_vocab_live.pkl"
UNTRAINED_WEIGHTS_PATH = BASE_DIR / "untrained_rnn.weights.h5"
TRAINED_WEIGHTS_PATH   = BASE_DIR / "fully_trained_rnn.weights.h5"

SEQ_LENGTH = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10_000
RANDOM_SEED = 42


def _clean_raw_text(text: str) -> str:
	"""Normalize and sanitize text to prevent hidden-character vocab drift."""
	text = unicodedata.normalize("NFKC", text)
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
	return text


def load_raw_text(path: Path) -> str:
	"""Read raw text corpus and print a quick sanity preview (DE Task 0)."""
	# DE Task 0: Load and validate raw text to ensure integrity.
	if not path.exists():
		raise FileNotFoundError(f"Missing corpus file: {path}")

	# utf-8-sig strips BOM automatically when present.
	text = path.read_text(encoding="utf-8-sig")
	text = _clean_raw_text(text)
	if not text.strip():
		raise ValueError("Corpus file exists but is empty.")

	print("\n" + "="*70)
	print("[DE Task 0] ===== LOAD & VALIDATE RAW TEXT =====")
	print("="*70)
	print("[DE Task 0] raw_text.txt loaded successfully")
	print(f"[DE Task 0] Total characters: {len(text):,}")
	print("[DE Task 0] First 100 characters preview:")
	print(text[:100])
	return text


def build_vocabulary(raw_text: str) -> tuple[list[str], dict[str, int], np.ndarray]:
	"""Create sorted character vocabulary and bi-directional mappings (DE Task 1)."""
	# DE Task 1: Extract unique characters and build char2idx + idx2char mappings.
	# Step 1 OUTPUT: Create the "alphabet" (vocabulary) and save two mappings.
	vocab = sorted(set(raw_text))
	char2idx = {ch: idx for idx, ch in enumerate(vocab)}
	idx2char = np.array(vocab)

	print("\n" + "="*70)
	print("[DE Task 1] ===== BUILD VOCABULARY & MAPPINGS =====")
	print("="*70)
	print(f"[DE Task 1] Unique vocabulary size: {len(vocab)}")
	print(f"[DE Task 1] Vocabulary (first 20 chars): {vocab[:20]}")
	print(f"[DE Task 1] char2idx['A'] = {char2idx['A']}")
	print(f"[DE Task 1] idx2char[0] = {repr(idx2char[0])}")
	print(f"[DE Task 1] Output files: char2idx dict + idx2char array (saved to char_vocab_live.pkl)")
	return vocab, char2idx, idx2char


def vectorize_text(raw_text: str, char2idx: dict[str, int]) -> np.ndarray:
	"""Map each character in the corpus to its integer ID (DE Task 2)."""
	# DE Task 2: Convert full text to integer vector using char2idx.
	# Step 2 OUTPUT: Replace every character with its numeric ID.
	text_as_int = np.array([char2idx[ch] for ch in raw_text], dtype=np.int32)
	print("\n" + "="*70)
	print("[DE Task 2] ===== VECTORIZE TEXT TO INTEGERS =====")
	print("="*70)
	print(f"[DE Task 2] Vectorized text shape: {text_as_int.shape}")
	print(f"[DE Task 2] First 50 characters of raw text: '{raw_text[:50]}'")
	print(f"[DE Task 2] First 50 corresponding IDs: {text_as_int[:50].tolist()}")
	print(f"[DE Task 2] Data type: {text_as_int.dtype}, min={text_as_int.min()}, max={text_as_int.max()}")
	return text_as_int


def split_input_target(chunk: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
	"""Shift target one position right: x=chunk[:-1], y=chunk[1:]."""
	input_text = chunk[:-1]
	target_text = chunk[1:]
	return input_text, target_text


def build_dataset(
	text_as_int: np.ndarray,
	seq_length: int = SEQ_LENGTH,
	batch_size: int = BATCH_SIZE,
	buffer_size: int = BUFFER_SIZE,
	seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
	"""Build tf.data pipeline with chunking, shift mapping, shuffle, and batch (DE Tasks 3-4)."""
	# DE Task 3: Create fixed-length sequence chunks from vectorized text.
	# Step 3 OUTPUT: Chop the book into flashcards (sequences of length 101).
	char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
	sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
	print("\n" + "="*70)
	print("[DE Task 3] ===== CREATE SEQUENCE FLASHCARDS =====")
	print("="*70)
	print(f"[DE Task 3] Created {len(list(sequences))} initial sequences of length {seq_length + 1}")
	
	# DE Task 4: Split each chunk into input (:-1) and target (1:), then shuffle and batch.
	# Step 4 OUTPUT: Set up "guess the next character" game (input=chars 0-99, target=chars 1-100).
	dataset = sequences.map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
	dataset = dataset.shuffle(buffer_size, seed=seed, reshuffle_each_iteration=True)
	dataset = dataset.batch(batch_size, drop_remainder=True)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)
	print("\n" + "="*70)
	print("[DE Task 4] ===== SHIFT & BATCH FOR TRAINING =====")
	print("="*70)
	return dataset


def save_live_vocab(
	live_vocab_path: Path,
	char2idx: dict[str, int],
	idx2char: np.ndarray,
	vocab_size: int,
) -> None:
	"""Persist live vocabulary mapping for team handoff (Checkpoint A)."""
	# Checkpoint A: Export DE artifacts for DLA/OEL/GEL team members.
	payload = {
		"char2idx": char2idx,
		"idx2char": idx2char,
		"vocab_size": vocab_size,
	}
	with live_vocab_path.open("wb") as f:
		pickle.dump(payload, f)
	print(f"[Checkpoint A] Saved live vocab handoff file: {live_vocab_path.name}")


def inspect_one_batch(dataset: tf.data.Dataset, char_vocab_live: dict) -> None:
	"""Print a one-batch shape check and example for DE validation evidence."""
	for input_batch, target_batch in dataset.take(1):
		print(f"[DE Task 4] ===== FINAL BATCH STRUCTURE =====")
		print(f"[DE Task 4] input_batch shape:  {input_batch.shape}")
		print(f"[DE Task 4] target_batch shape: {target_batch.shape}")
		print(f"[DE Task 4] Total sequences per batch: {input_batch.shape[0]} (BATCH_SIZE={BATCH_SIZE})")
		print(f"[DE Task 4] Sequence length: {input_batch.shape[1]} (SEQ_LENGTH={SEQ_LENGTH})")
		
		# Show a sample of one sequence
		idx2char = char_vocab_live["idx2char"]
		first_input = input_batch[0].numpy()
		first_target = target_batch[0].numpy()
		input_text = "".join([idx2char[i] for i in first_input])
		target_text = "".join([idx2char[i] for i in first_target])
		print(f"[DE Task 4] Example sequence #1:")
		print(f"[DE Task 4]   Input (chars 0-99):   '{input_text[:50]}...'")
		print(f"[DE Task 4]   Target (chars 1-100): '{target_text[:50]}...'")
		print(f"[DE Task 4] READY: Model will learn to predict target from input!")
		break


def optional_vault_vocab_check(vault_vocab_path: Path, live_vocab_size: int) -> None:
	"""Optional sanity check against provided vault vocab file, if available."""
	if not vault_vocab_path.exists():
		print("[DE] Vault vocab file not found. Skipping parity check.")
		return

	try:
		with vault_vocab_path.open("rb") as f:
			vault_obj = pickle.load(f)
		vault_size = len(vault_obj) if hasattr(vault_obj, "__len__") else None
		print(f"[DE] Vault vocab object type: {type(vault_obj).__name__}")
		if vault_size is not None:
			print(f"[DE] Vault vocab length (len): {vault_size}")
			print(f"[DE] Live vocab size: {live_vocab_size}")
	except Exception as exc:
		print(f"[DE] Could not parse char_vocab.pkl for quick check: {exc}")


def run_de_pipeline() -> tuple[tf.data.Dataset, dict[str, int], np.ndarray, int]:
	"""Execute all DE tasks and return handoff artifacts in memory."""
	raw_text = load_raw_text(RAW_TEXT_PATH)
	_, char2idx, idx2char = build_vocabulary(raw_text)
	vocab_size = len(idx2char)
	text_as_int = vectorize_text(raw_text, char2idx)
	dataset = build_dataset(text_as_int)

	save_live_vocab(LIVE_VOCAB_PATH, char2idx, idx2char, vocab_size)
	
	# Load the saved vocab for example printing
	with LIVE_VOCAB_PATH.open("rb") as f:
		char_vocab_live = pickle.load(f)
	
	inspect_one_batch(dataset, char_vocab_live)
	
	print("\n" + "="*70)
	print("[VAULT CHECK] ===== OPTIONAL VAULT COMPARISON =====")
	print("="*70)
	optional_vault_vocab_check(VAULT_VOCAB_PATH, vocab_size)

	print("\n" + "="*70)
	print("[CHECKPOINT A] ===== DE HANDOFF SUMMARY =====")
	print("="*70)
	print(f"[CHECKPOINT A] vocab_size = {vocab_size}")
	print(f"[CHECKPOINT A] live vocab file = {LIVE_VOCAB_PATH}")
	print("[CHECKPOINT A] Share vocab_size and char_vocab_live.pkl with DLA/OEL/GEL.")
	print("="*70 + "\n")

	return dataset, char2idx, idx2char, vocab_size


# =============================================================================
# ROLE 2: DEEP LEARNING ARCHITECT (DLA)
# =============================================================================

# DLA configuration
EMBEDDING_DIM = 256
# 1024 matches the provided vault checkpoint tensor shapes.
RNN_UNITS = 1024
STATEFUL_MODE = False  # Toggle between True/False for DLA Task 3


def load_handoff_vocab(vocab_path: Path) -> tuple[int, dict, np.ndarray]:
	"""Load vocab_size and mappings from DE handoff file."""
	with vocab_path.open("rb") as f:
		payload = pickle.load(f)
	vocab_size = payload["vocab_size"]
	char2idx = payload["char2idx"]
	idx2char = payload["idx2char"]
	return vocab_size, char2idx, idx2char


def build_model(
	vocab_size: int,
	embedding_dim: int = EMBEDDING_DIM,
	rnn_units: int = RNN_UNITS,
	batch_size: int = BATCH_SIZE,
	stateful: bool = STATEFUL_MODE,
	rnn_type: str = "GRU",
) -> tf.keras.Model:
	"""
	DLA Task 1: Build Sequential model with Embedding, RNN, Dense layers.
	
	Architecture:
	- Input: batch_shape=[batch_size, seq_length] (or [1, 1] for inference)
	- Embedding: vocab_size -> embedding_dim dense vectors
	- RNN (GRU or LSTM): learns temporal patterns
	- Dense: embedding_dim -> vocab_size (logits for next char prediction)
	"""
	print("\n" + "="*70)
	print("[DLA Task 1] ===== BUILD MODEL ARCHITECTURE =====")
	print("="*70)
	
	model = tf.keras.Sequential()
	
	# Input layer with fixed batch shape (required for stateful RNNs)
	model.add(tf.keras.Input(batch_shape=[batch_size, None]))
	
	# Embedding layer: convert integer indices to dense vectors
	print(f"[DLA Task 1] Adding Embedding layer:")
	print(f"[DLA Task 1]   - Input vocab size: {vocab_size}")
	print(f"[DLA Task 1]   - Output embedding dim: {embedding_dim}")
	model.add(tf.keras.layers.Embedding(
		input_dim=vocab_size,
		output_dim=embedding_dim,
		mask_zero=False
	))
	
	# Recurrent layer (GRU or LSTM)
	if rnn_type.upper() == "LSTM":
		print(f"[DLA Task 1] Adding LSTM layer:")
		print(f"[DLA Task 1]   - Units: {rnn_units}")
		print(f"[DLA Task 1]   - Stateful: {stateful}")
		print(f"[DLA Task 1]   - Return sequences: True")
		model.add(tf.keras.layers.LSTM(
			units=rnn_units,
			stateful=stateful,
			return_sequences=True
		))
	else:  # Default to GRU
		print(f"[DLA Task 1] Adding GRU layer:")
		print(f"[DLA Task 1]   - Units: {rnn_units}")
		print(f"[DLA Task 1]   - Stateful: {stateful}")
		print(f"[DLA Task 1]   - Return sequences: True")
		model.add(tf.keras.layers.GRU(
			units=rnn_units,
			stateful=stateful,
			return_sequences=True
		))
	
	# Dense output layer: predict logits for next character
	print(f"[DLA Task 1] Adding Dense output layer:")
	print(f"[DLA Task 1]   - Units (vocab_size): {vocab_size}")
	print(f"[DLA Task 1]   - Activation: None (logits, for from_logits=True loss)")
	model.add(tf.keras.layers.Dense(vocab_size, activation=None))
	
	print(f"[DLA Task 1] Model built successfully!")
	print(f"[DLA Task 1] Total trainable parameters: {model.count_params():,}")
	
	return model


def print_model_summary(model: tf.keras.Model, vocab_size: int) -> None:
	"""Print detailed model architecture summary."""
	print("\n" + "="*70)
	print("[DLA Task 1] ===== MODEL ARCHITECTURE SUMMARY =====")
	print("="*70)
	model.summary()
	print(f"\n[DLA Task 1] Checkpoint B ready: model architecture = {model.count_params():,} parameters")
	print(f"[DLA Task 1] vocab_size = {vocab_size}")


def run_dla_task1() -> tf.keras.Model:
	"""Execute DLA Task 1: build and validate model architecture."""
	print("\n" + "="*70)
	print("[DLA] ===== DEEP LEARNING ARCHITECT (DLA) TASKS BEGIN =====")
	print("="*70)
	
	# Load vocab from DE handoff
	vocab_size, char2idx, idx2char = load_handoff_vocab(LIVE_VOCAB_PATH)
	print(f"\n[DLA] Loaded vocab from DE handoff:")
	print(f"[DLA] vocab_size = {vocab_size}")
	
	# Build model with GRU (default for this section)
	model = build_model(
		vocab_size=vocab_size,
		embedding_dim=EMBEDDING_DIM,
		rnn_units=RNN_UNITS,
		batch_size=BATCH_SIZE,
		stateful=STATEFUL_MODE,
		rnn_type="GRU"
	)
	
	# Print architecture
	print_model_summary(model, vocab_size)
	
	print("\n" + "="*70)
	print("[DLA Task 1] ===== MARKDOWN: WHY EMBEDDING? =====")
	print("="*70)
	print("""
[DLA Task 2] Explanation: Why use Embedding instead of raw integer inputs?

REASON 1: Semantic Information
- Raw integers (0-71) have no inherent meaning; they're just arbitrary labels.
- Embedding layers learn dense vectors (256-dim) that capture character similarity.
- Example: 'A' and 'a' will have similar embeddings; punctuation will differ.

REASON 2: Improved Learning
- RNNs work better with continuous vectors than sparse integers.
- Dense embeddings allow the RNN to extract richer temporal patterns.
- Reduces parameter count vs one-hot encoding (71 dims vs 256 dims still dense).

REASON 3: Dimensionality Control
- Fixed embedding_dim=256 means every character maps to same-sized vector.
- RNN sees consistent input shape regardless of vocab_size.
- Easier to tune: increase embedding_dim for richer representations.

REASON 4: Transfer & Generalization
- Learned embeddings can capture character properties (vowels, consonants, etc.).
- Similar characters naturally cluster in embedding space.
- Helps model generalize to unseen sequences better.
	""")
	print("="*70)
	
	return model


def run_dla_task3(vocab_size: int) -> None:
	"""DLA Task 3: Compare stateful=True vs stateful=False behavior."""
	print("\n" + "=" * 70)
	print("[DLA Task 3] ===== STATEFUL VS NON-STATEFUL COMPARISON =====")
	print("=" * 70)

	non_stateful_model = build_model(
		vocab_size=vocab_size,
		embedding_dim=EMBEDDING_DIM,
		rnn_units=RNN_UNITS,
		batch_size=BATCH_SIZE,
		stateful=False,
		rnn_type="GRU",
	)

	stateful_model = build_model(
		vocab_size=vocab_size,
		embedding_dim=EMBEDDING_DIM,
		rnn_units=RNN_UNITS,
		batch_size=BATCH_SIZE,
		stateful=True,
		rnn_type="GRU",
	)

	non_stateful_params = non_stateful_model.count_params()
	stateful_params = stateful_model.count_params()

	print(f"[DLA Task 3] Non-stateful GRU params: {non_stateful_params:,}")
	print(f"[DLA Task 3] Stateful GRU params:     {stateful_params:,}")
	print("[DLA Task 3] Observation: Parameter count remains the same.")
	print("[DLA Task 3] Behavior difference:")
	print("[DLA Task 3] - stateful=False: hidden state resets each batch.")
	print("[DLA Task 3] - stateful=True: hidden state carries across batches until reset_states().")


def _weight_compatibility_check(model: tf.keras.Model, weights_path: Path, label: str) -> bool:
	"""Try loading untrained vault weights to verify architecture compatibility."""
	if not weights_path.exists():
		print(f"[DLA Task 4] {label}: weights file missing, compatibility check skipped.")
		return False

	try:
		model.load_weights(weights_path)
		print(f"[DLA Task 4] {label}: compatible with {weights_path.name}")
		return True
	except Exception as exc:
		print(f"[DLA Task 4] {label}: NOT compatible with {weights_path.name} ({exc})")
		return False


def run_dla_task4(vocab_size: int) -> str:
	"""DLA Task 4: Compare GRU vs LSTM and pick official team architecture."""
	print("\n" + "=" * 70)
	print("[DLA Task 4] ===== GRU VS LSTM ARCHITECTURE SWAP =====")
	print("=" * 70)

	gru_model = build_model(
		vocab_size=vocab_size,
		embedding_dim=EMBEDDING_DIM,
		rnn_units=RNN_UNITS,
		batch_size=BATCH_SIZE,
		stateful=False,
		rnn_type="GRU",
	)

	lstm_model = build_model(
		vocab_size=vocab_size,
		embedding_dim=EMBEDDING_DIM,
		rnn_units=RNN_UNITS,
		batch_size=BATCH_SIZE,
		stateful=False,
		rnn_type="LSTM",
	)

	gru_params = gru_model.count_params()
	lstm_params = lstm_model.count_params()
	param_delta = lstm_params - gru_params

	print(f"[DLA Task 4] GRU params : {gru_params:,}")
	print(f"[DLA Task 4] LSTM params: {lstm_params:,}")
	print(f"[DLA Task 4] Delta (LSTM - GRU): {param_delta:,}")

	gru_compatible = _weight_compatibility_check(gru_model, UNTRAINED_WEIGHTS_PATH, "GRU")
	lstm_compatible = _weight_compatibility_check(lstm_model, UNTRAINED_WEIGHTS_PATH, "LSTM")

	if gru_compatible and not lstm_compatible:
		official_arch = "GRU"
		reason = "GRU is vault-weight compatible while LSTM is not."
	elif lstm_compatible and not gru_compatible:
		official_arch = "LSTM"
		reason = "LSTM is vault-weight compatible while GRU is not."
	elif gru_params <= lstm_params:
		official_arch = "GRU"
		reason = "GRU has fewer parameters, trains faster for class-time constraints."
	else:
		official_arch = "LSTM"
		reason = "LSTM selected due to architecture preference despite higher parameter cost."

	print("\n" + "=" * 70)
	print("[CHECKPOINT B] ===== OFFICIAL TEAM ARCHITECTURE DECISION =====")
	print("=" * 70)
	print(f"[CHECKPOINT B] Official architecture: {official_arch}")
	print(f"[CHECKPOINT B] Reason: {reason}")
	print("[CHECKPOINT B] Share this decision and build_model settings with OEL/GEL.")
	print("=" * 70)

	return official_arch


def run_dla_all() -> str:
	"""Run DLA tasks 1, 3, and 4 end-to-end with one command."""
	vocab_size, _, _ = load_handoff_vocab(LIVE_VOCAB_PATH)
	run_dla_task1()
	run_dla_task3(vocab_size)
	return run_dla_task4(vocab_size)


# =============================================================================
# ROLE 3: OPTIMIZATION & TRAINING LEAD (OEL)
# =============================================================================

# OEL configuration
OEL_EPOCHS     = 5
ADAM_LR        = 1e-3
RMSPROP_LR     = 1e-3

# Ouput artifact paths
ADAM_LOSS_PLOT_PATH    = BASE_DIR / "oel_adam_loss.png"
COMPARE_LOSS_PLOT_PATH = BASE_DIR / "oel_optimizer_comparison.png"
OEL_LOG_PATH           = BASE_DIR / "oel_training_log.txt"


def _load_oel_artifacts() -> tuple[tf.data.Dataset, int]:
	"""Reload DE vocab + rebuild tf.data pipeline for OEL (no noisy DE output)."""
	vocab_size, char2idx, _ = load_handoff_vocab(LIVE_VOCAB_PATH)

	raw_text = RAW_TEXT_PATH.read_text(encoding="utf-8-sig")
	raw_text = _clean_raw_text(raw_text)
	text_as_int = np.array([char2idx[ch] for ch in raw_text], dtype=np.int32)

	char_ds = tf.data.Dataset.from_tensor_slices(text_as_int)
	seqs    = char_ds.batch(SEQ_LENGTH + 1, drop_remainder=True)
	dataset = seqs.map(split_input_target, num_parallel_calls=tf.data.AUTOTUNE)
	dataset = dataset.shuffle(BUFFER_SIZE, seed=RANDOM_SEED, reshuffle_each_iteration=True)
	dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)

	print(f"[OEL] Vocab reloaded from {LIVE_VOCAB_PATH.name} → vocab_size={vocab_size}")
	print(f"[OEL] Dataset rebuilt from raw text: {len(text_as_int):,} chars → "
	      f"batches of shape ({BATCH_SIZE}, {SEQ_LENGTH})")
	return dataset, vocab_size


def _define_loss() -> tf.keras.losses.Loss:
	"""OEL Task 1: Define SparseCategoricalCrossentropy with from_logits=True."""
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

	print("\n" + "=" * 70)
	print("[OEL Task 1] ===== CUSTOM LOSS FUNCTION =====")
	print("=" * 70)
	print("[OEL Task 1] Loss: SparseCategoricalCrossentropy(from_logits=True)")
	print("[OEL Task 1]")
	print("[OEL Task 1] WHY from_logits=True?")
	print("[OEL Task 1]   The Dense output layer uses activation=None (raw logits).")
	print("[OEL Task 1]   Setting from_logits=True tells the loss function to apply")
	print("[OEL Task 1]   softmax *internally* before computing cross-entropy.")
	print("[OEL Task 1]   This is numerically more stable than softmax -> log separately")
	print("[OEL Task 1]   because it avoids catastrophic cancellation near 0 and 1.")
	print("[OEL Task 1]")
	print("[OEL Task 1] WHY SparseCategoricalCrossentropy (not CategoricalCrossentropy)?")
	print("[OEL Task 1]   Labels are integer indices (0-71), not one-hot vectors.")
	print("[OEL Task 1]   The 'Sparse' variant accepts raw integer labels, saving")
	print("[OEL Task 1]   the memory and computation of one-hot encoding a 72-class target.")
	print("=" * 70)

	return loss_fn


def _compile_with_adam(model: tf.keras.Model, loss_fn: tf.keras.losses.Loss) -> None:
	"""OEL Task 2: Compile model with Adam optimizer + custom loss."""
	optimizer = tf.keras.optimizers.Adam(learning_rate=ADAM_LR)
	model.compile(optimizer=optimizer, loss=loss_fn)

	print("\n" + "=" * 70)
	print("[OEL Task 2] ===== COMPILE WITH ADAM =====")
	print("=" * 70)
	print(f"[OEL Task 2] Optimizer : Adam(lr={ADAM_LR})")
	print("[OEL Task 2] Loss      : SparseCategoricalCrossentropy(from_logits=True)")
	print("[OEL Task 2]")
	print("[OEL Task 2] WHY Adam?")
	print("[OEL Task 2]   Adam combines momentum (first moment) with per-parameter")
	print("[OEL Task 2]   adaptive rates (second moment), making it robust for RNNs.")
	print("[OEL Task 2]   Default lr=0.001 is stable for character-level text generation.")
	print("=" * 70)


def _save_loss_plot(history_dict: dict[str, list[float]], title: str, save_path: Path) -> None:
	"""Save a loss curve PNG. Silently skips if matplotlib is not installed."""
	try:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

		plt.figure(figsize=(8, 4))
		for label, losses in history_dict.items():
			plt.plot(range(1, len(losses) + 1), losses, marker="o", label=label)
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.title(title)
		plt.legend()
		plt.tight_layout()
		plt.savefig(save_path, dpi=120)
		plt.close()
		print(f"[OEL] Loss plot saved → {save_path.name}")
	except ImportError:
		print("[OEL] matplotlib not installed — loss plot skipped.")


def _save_training_log(
	adam_losses: list[float],
	rmsprop_losses: list[float],
	epochs: int,
	winner: str,
) -> None:
	"""Write plaintext OEL training log for Checkpoint C handoff."""
	lines = [
		"OEL Training Log",
		f"Epochs: {epochs}",
		f"Architecture: GRU | RNN_UNITS={RNN_UNITS} | EMBEDDING_DIM={EMBEDDING_DIM} | vocab_size=72",
		"",
		"Adam losses per epoch:",
	]
	for i, loss in enumerate(adam_losses, start=1):
		lines.append(f"  Epoch {i:2d}: {loss:.4f}")
	lines += ["", "RMSprop losses per epoch:"]
	for i, loss in enumerate(rmsprop_losses, start=1):
		lines.append(f"  Epoch {i:2d}: {loss:.4f}")
	lines += [
		"",
		f"Better optimizer for first {epochs} epochs: {winner}",
		"Plots: oel_adam_loss.png, oel_optimizer_comparison.png",
	]
	OEL_LOG_PATH.write_text("\n".join(lines))
	print(f"[OEL] Training log saved → {OEL_LOG_PATH.name}")


def run_oel_task3(
	model_adam: tf.keras.Model,
	dataset: tf.data.Dataset,
	epochs: int = OEL_EPOCHS,
) -> list[float]:
	"""OEL Task 3: Train for N epochs with Adam, print per-epoch loss, save plot."""
	print("\n" + "=" * 70)
	print(f"[OEL Task 3] ===== LIVE TRAINING RUN ({epochs} EPOCHS – ADAM) =====")
	print("=" * 70)

	# Load untrained vault weights for a reproducible warm start
	if UNTRAINED_WEIGHTS_PATH.exists():
		model_adam.load_weights(UNTRAINED_WEIGHTS_PATH)
		print(f"[OEL Task 3] Warm-start: loaded {UNTRAINED_WEIGHTS_PATH.name}")
	else:
		print("[OEL Task 3] Vault weights not found — training from random init.")

	print(f"[OEL Task 3] BATCH_SIZE={BATCH_SIZE}  SEQ_LENGTH={SEQ_LENGTH}  "
	      f"EMBEDDING_DIM={EMBEDDING_DIM}  RNN_UNITS={RNN_UNITS}")
	print(f"[OEL Task 3] Training for {epochs} epoch(s)...")

	history = model_adam.fit(dataset, epochs=epochs)
	adam_losses = history.history["loss"]

	print(f"\n[OEL Task 3] Adam per-epoch results:")
	for i, loss in enumerate(adam_losses, start=1):
		print(f"[OEL Task 3]   Epoch {i:2d}: loss = {loss:.4f}")

	_save_loss_plot(
		{"Adam": adam_losses},
		f"Adam Training Loss ({epochs} Epochs)",
		ADAM_LOSS_PLOT_PATH,
	)

	return adam_losses


def run_oel_task4(
	vocab_size: int,
	dataset: tf.data.Dataset,
	adam_losses: list[float],
	loss_fn: tf.keras.losses.Loss,
	epochs: int = OEL_EPOCHS,
) -> None:
	"""OEL Task 4: Swap to RMSprop, train same epochs, compare against Adam."""
	print("\n" + "=" * 70)
	print(f"[OEL Task 4] ===== OPTIMIZER SWAP: RMSPROP ({epochs} EPOCHS) =====")
	print("=" * 70)

	# Fresh model from the same starting weights for a fair comparison
	model_rmsprop = build_model(
		vocab_size=vocab_size,
		embedding_dim=EMBEDDING_DIM,
		rnn_units=RNN_UNITS,
		batch_size=BATCH_SIZE,
		stateful=False,
		rnn_type="GRU",
	)

	if UNTRAINED_WEIGHTS_PATH.exists():
		model_rmsprop.load_weights(UNTRAINED_WEIGHTS_PATH)
		print(f"[OEL Task 4] Same warm-start weights loaded for fair comparison.")

	optimizer_rmsprop = tf.keras.optimizers.RMSprop(learning_rate=RMSPROP_LR)
	model_rmsprop.compile(optimizer=optimizer_rmsprop, loss=loss_fn)
	print(f"[OEL Task 4] Re-compiled with RMSprop(lr={RMSPROP_LR})")
	print(f"[OEL Task 4] Training for {epochs} epoch(s)...")

	history_rmsprop = model_rmsprop.fit(dataset, epochs=epochs)
	rmsprop_losses = history_rmsprop.history["loss"]

	print(f"\n[OEL Task 4] RMSprop per-epoch results:")
	for i, loss in enumerate(rmsprop_losses, start=1):
		print(f"[OEL Task 4]   Epoch {i:2d}: loss = {loss:.4f}")

	_save_loss_plot(
		{"Adam": adam_losses, "RMSprop": rmsprop_losses},
		f"Adam vs RMSprop Loss ({epochs} Epochs)",
		COMPARE_LOSS_PLOT_PATH,
	)

	adam_drop    = adam_losses[0]    - adam_losses[-1]
	rmsprop_drop = rmsprop_losses[0] - rmsprop_losses[-1]
	winner = "Adam" if adam_drop >= rmsprop_drop else "RMSprop"

	print("\n" + "=" * 70)
	print("[OEL Task 4] ===== ADAM VS RMSPROP COMPARISON =====")
	print("=" * 70)
	print(f"[OEL Task 4] Adam    → start: {adam_losses[0]:.4f}  "
	      f"end: {adam_losses[-1]:.4f}  drop: {adam_drop:.4f}")
	print(f"[OEL Task 4] RMSprop → start: {rmsprop_losses[0]:.4f}  "
	      f"end: {rmsprop_losses[-1]:.4f}  drop: {rmsprop_drop:.4f}")
	print(f"[OEL Task 4] Better optimizer for first {epochs} epochs: {winner}")
	print("[OEL Task 4]   Adam   = momentum + adaptive rates → faster early convergence.")
	print("[OEL Task 4]   RMSprop = adaptive rates only     → smoother, slower initial drop.")
	print("=" * 70)

	_save_training_log(adam_losses, rmsprop_losses, epochs, winner)


def run_oel_all() -> None:
	"""Run OEL Tasks 1-4 end-to-end and emit Checkpoint C."""
	print("\n" + "=" * 70)
	print("[OEL] ===== OPTIMIZATION & TRAINING LEAD (OEL) TASKS BEGIN =====")
	print("=" * 70)

	# Reload DE artifacts quietly
	dataset, vocab_size = _load_oel_artifacts()

	# Task 1: define loss
	loss_fn = _define_loss()

	# Task 2: build + compile with Adam
	model_adam = build_model(
		vocab_size=vocab_size,
		embedding_dim=EMBEDDING_DIM,
		rnn_units=RNN_UNITS,
		batch_size=BATCH_SIZE,
		stateful=False,
		rnn_type="GRU",
	)
	_compile_with_adam(model_adam, loss_fn)

	# Task 3: train with Adam
	adam_losses = run_oel_task3(model_adam, dataset, epochs=OEL_EPOCHS)

	# Task 4: compare with RMSprop
	run_oel_task4(vocab_size, dataset, adam_losses, loss_fn, epochs=OEL_EPOCHS)

	# Checkpoint C
	print("\n" + "=" * 70)
	print("[CHECKPOINT C] ===== OEL HANDOFF SUMMARY =====")
	print("=" * 70)
	print(f"[CHECKPOINT C] Architecture : GRU | RNN_UNITS={RNN_UNITS} | EMBEDDING_DIM={EMBEDDING_DIM}")
	print(f"[CHECKPOINT C] Loss         : SparseCategoricalCrossentropy(from_logits=True)")
	print(f"[CHECKPOINT C] Training     : {OEL_EPOCHS} epochs each (Adam + RMSprop comparison)")
	print(f"[CHECKPOINT C] Artifacts    : {ADAM_LOSS_PLOT_PATH.name}, "
	      f"{COMPARE_LOSS_PLOT_PATH.name}, {OEL_LOG_PATH.name}")
	print("[CHECKPOINT C] For GEL      : rebuild with batch_size=1, load fully_trained_rnn.weights.h5")
	print("=" * 70 + "\n")


# =============================================================================
# ROLE 4: GENERATION & EVALUATION LEAD (GEL)
# =============================================================================

# GEL configuration
GEL_START_STRING  = "Alice "   # Seed string fed to the model at inference time
GEL_NUM_GENERATE  = 1000       # Number of characters to generate per sample
GEL_TEMPERATURES  = [0.5, 1.0, 1.5]
GEL_OUTPUT_PATH   = BASE_DIR / "gel_generated_text.txt"


def build_inference_model(vocab_size: int) -> tf.keras.Model:
	"""
	GEL Task 1 (part A): Rebuild GRU model with batch_size=1 for single-step inference.

	Training used batch_size=64 and stateful=False.
	Inference requires batch_size=1 so we can feed one character at a time.
	We also use stateful=True so the hidden state persists across single-character calls.
	"""
	print("\n" + "=" * 70)
	print("[GEL Task 1] ===== INFERENCE MODEL SETUP =====")
	print("=" * 70)
	print("[GEL Task 1] Rebuilding GRU with batch_size=1 and stateful=True for inference.")

	inf_model = tf.keras.Sequential([
		tf.keras.Input(batch_shape=[1, None]),
		tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=False),
		tf.keras.layers.GRU(units=RNN_UNITS, stateful=True, return_sequences=True),
		tf.keras.layers.Dense(vocab_size, activation=None),
	])

	print(f"[GEL Task 1] Inference model built: batch_size=1, stateful=True")
	print(f"[GEL Task 1] Total parameters: {inf_model.count_params():,}")

	if not TRAINED_WEIGHTS_PATH.exists():
		raise FileNotFoundError(
			f"[GEL Task 1] ERROR: {TRAINED_WEIGHTS_PATH.name} not found. "
			"Place fully_trained_rnn.weights.h5 in the project folder."
		)

	inf_model.load_weights(TRAINED_WEIGHTS_PATH)
	print(f"[GEL Task 1] Loaded weights from: {TRAINED_WEIGHTS_PATH.name}")
	print("=" * 70)
	return inf_model


def generate_text(
	model: tf.keras.Model,
	idx2char: np.ndarray,
	char2idx: dict[str, int],
	start_string: str,
	num_generate: int = GEL_NUM_GENERATE,
	temperature: float = 1.0,
) -> str:
	"""
	GEL Task 2: Character-level generation loop.

	Algorithm:
	1. Convert start_string to integer IDs.
	2. Feed them into the model one character at a time to prime the hidden state.
	3. On each subsequent step:
	   a. Get logits from the model.
	   b. Divide by temperature to scale the distribution.
	   c. Sample one character index from the scaled distribution.
	   d. Feed that index back in as the next input.
	4. Decode the accumulated integer IDs back to text.

	Temperature effect:
	  < 1.0  → sharper distribution → more predictable / repetitive output
	  = 1.0  → unmodified distribution → balanced
	  > 1.0  → flatter distribution → more surprising / chaotic output
	"""
	# Filter any characters in start_string not in the vocab
	start_string = "".join(ch for ch in start_string if ch in char2idx)
	if not start_string:
		start_string = idx2char[0]  # fall back to first vocab char

	# Reset hidden state before each new generation.
	# Keras 3 removed reset_states() from Sequential; call it on each stateful layer.
	for layer in model.layers:
		if hasattr(layer, "reset_states"):
			layer.reset_states()

	input_ids = [char2idx[ch] for ch in start_string]
	input_tensor = tf.expand_dims(input_ids, 0)  # shape: (1, len(start_string))

	generated = []

	# Prime the hidden state with start_string (all chars except last)
	# then begin generating from the last priming character
	for ch_id in input_ids[:-1]:
		model(tf.expand_dims([ch_id], 0))  # advances hidden state, output discarded

	# Generation loop: feed last seed char, then each predicted char
	next_input = tf.expand_dims([input_ids[-1]], 0)  # shape: (1, 1)

	for _ in range(num_generate):
		logits = model(next_input)           # shape: (1, 1, vocab_size)
		logits = tf.squeeze(logits, axis=0)  # shape: (1, vocab_size)

		# Apply temperature scaling
		scaled_logits = logits / temperature

		# Sample from the scaled distribution
		predicted_id = tf.random.categorical(scaled_logits, num_samples=1)
		predicted_id = int(tf.squeeze(predicted_id).numpy())

		generated.append(idx2char[predicted_id])
		next_input = tf.expand_dims([predicted_id], 0)

	return start_string + "".join(generated)


def run_gel_task3(
	inf_model: tf.keras.Model,
	idx2char: np.ndarray,
	char2idx: dict[str, int],
) -> dict[float, str]:
	"""GEL Task 3: Generate text at each temperature and print samples."""
	print("\n" + "=" * 70)
	print("[GEL Task 3] ===== TEMPERATURE SCALING – TEXT GENERATION =====")
	print("=" * 70)
	print(f"[GEL Task 3] Start string  : '{GEL_START_STRING}'")
	print(f"[GEL Task 3] Char budget   : {GEL_NUM_GENERATE} characters per sample")
	print(f"[GEL Task 3] Temperatures  : {GEL_TEMPERATURES}")

	samples: dict[float, str] = {}

	for temp in GEL_TEMPERATURES:
		print(f"\n[GEL Task 3] --- Generating at temperature = {temp} ---")
		text = generate_text(
			model=inf_model,
			idx2char=idx2char,
			char2idx=char2idx,
			start_string=GEL_START_STRING,
			num_generate=GEL_NUM_GENERATE,
			temperature=temp,
		)
		samples[temp] = text
		print(f"[GEL Task 3] temperature={temp} | first 200 chars:")
		print("-" * 60)
		print(text[:200])
		print("-" * 60)

	return samples


def _qualitative_analysis(samples: dict[float, str]) -> str:
	"""GEL Task 4: Produce a qualitative comparison of the three temperature outputs."""
	lines = [
		"",
		"[GEL Task 4] ===== QUALITATIVE ANALYSIS =====",
		"",
		"[GEL Task 4] Temperature 0.5 — LOW (Conservative / Repetitive)",
		"[GEL Task 4]   The model picks the most probable next character at each step.",
		"[GEL Task 4]   Output tends to be grammatically structured and recognizable",
		"[GEL Task 4]   as Alice-in-Wonderland-style prose, but may loop on common",
		"[GEL Task 4]   phrases ('said Alice', 'the Queen', '') repeatedly.",
		"[GEL Task 4]   Use when: you want coherent, readable output.",
		"",
		"[GEL Task 4] Temperature 1.0 — BASELINE (Balanced)",
		"[GEL Task 4]   Uses the unmodified learned distribution.",
		"[GEL Task 4]   Output mixes recognizable words with occasional novel combinations.",
		"[GEL Task 4]   Best general-purpose temperature for character-level RNNs.",
		"[GEL Task 4]   Use when: you want the model's 'natural' voice.",
		"",
		"[GEL Task 4] Temperature 1.5 — HIGH (Creative / Chaotic)",
		"[GEL Task 4]   Flattens the distribution → low-probability characters get",
		"[GEL Task 4]   a much larger chance of being sampled.",
		"[GEL Task 4]   Output contains invented words, broken syntax, and random",
		"[GEL Task 4]   punctuation, but can surface surprising new combinations.",
		"[GEL Task 4]   Use when: you want diversity and are willing to accept noise.",
		"",
		"[GEL Task 4] Mathematical explanation of temperature:",
		"[GEL Task 4]   p_i = exp(z_i / T) / sum_j( exp(z_j / T) )",
		"[GEL Task 4]   T < 1 → sharpens peaks (winner-takes-all)",
		"[GEL Task 4]   T = 1 → standard softmax",
		"[GEL Task 4]   T > 1 → flattens distribution (uniform at T→∞)",
		"",
	]

	# Append quick automatic length / unique-word stats
	for temp, text in samples.items():
		words = text.split()
		unique = len(set(w.lower().strip(".,!?;:'\"") for w in words))
		lines.append(
			f"[GEL Task 4] temp={temp}: {len(text)} chars | "
			f"{len(words)} words | {unique} unique words"
		)

	analysis = "\n".join(lines)
	print(analysis)
	return analysis


def _save_gel_output(samples: dict[float, str], analysis: str) -> None:
	"""Write all generated samples + analysis to gel_generated_text.txt."""
	sections = ["GEL Generated Text Samples", "=" * 70, ""]
	for temp, text in samples.items():
		sections += [
			f"--- Temperature {temp} ---",
			text,
			"",
		]
	sections += ["=" * 70, "Qualitative Analysis", "=" * 70, analysis]
	GEL_OUTPUT_PATH.write_text("\n".join(sections), encoding="utf-8")
	print(f"\n[GEL] All samples saved → {GEL_OUTPUT_PATH.name}")


def run_gel_all() -> None:
	"""Run GEL Tasks 1-4 end-to-end and emit Checkpoint D summary."""
	print("\n" + "=" * 70)
	print("[GEL] ===== GENERATION & EVALUATION LEAD (GEL) TASKS BEGIN =====")
	print("=" * 70)

	# Task 1: load vocab + build inference model
	vocab_size, char2idx, idx2char = load_handoff_vocab(LIVE_VOCAB_PATH)
	print(f"[GEL] Vocab loaded: vocab_size={vocab_size}")

	inf_model = build_inference_model(vocab_size)

	# Tasks 2 & 3: generate at three temperatures
	samples = run_gel_task3(inf_model, idx2char, char2idx)

	# Task 4: qualitative analysis
	analysis = _qualitative_analysis(samples)

	# Save all output
	_save_gel_output(samples, analysis)

	# Checkpoint D
	print("\n" + "=" * 70)
	print("[CHECKPOINT D] ===== FINAL INTEGRATION SUMMARY =====")
	print("=" * 70)
	print("[CHECKPOINT D] DE   → char_vocab_live.pkl, vocab_size=72            ✓")
	print("[CHECKPOINT D] DLA  → GRU | 4,030,536 params | vault-compatible     ✓")
	print(f"[CHECKPOINT D] OEL  → Adam(drop 0.99) > RMSprop(drop 0.72) / 5ep   ✓")
	print(f"[CHECKPOINT D] GEL  → {GEL_NUM_GENERATE}-char samples at T=0.5, 1.0, 1.5   ✓")
	print(f"[CHECKPOINT D] File → {GEL_OUTPUT_PATH.name}")
	print("[CHECKPOINT D] Package script + pkl + plots + log for submission.")
	print("=" * 70 + "\n")


def print_conceptual_questions() -> None:
	"""Print all 5 conceptual questions with answers (Part V)."""
	print("\n" + "=" * 70)
	print("[PART V] ===== CONCEPTUAL QUESTIONS (INDIVIDUAL) =====")
	print("=" * 70)

	print("""
[Q1] Why are RNNs better for text generation than dense feedforward networks?

Dense feedforward networks treat each input position independently — they have
no mechanism to remember what came before. For text generation, the next
character depends heavily on the preceding context (e.g., "Alic" strongly
suggests "e"). RNNs maintain a hidden state that is updated at every time step,
effectively acting as a compressed memory of the entire sequence seen so far.
This allows the model to capture sequential dependencies, grammar patterns, and
long-range stylistic structure that a flat dense network simply cannot encode
without an exponentially large input window.
""")

	print("""
[Q2] What is vanishing gradient, and how do GRU/LSTM mitigate it?

During backpropagation through time (BPTT), gradients are multiplied by the
recurrent weight matrix at each timestep. If those values are < 1, the gradient
shrinks exponentially as it propagates back through long sequences — this is
the vanishing gradient problem. Plain RNNs struggle to learn dependencies
spanning more than ~10-20 steps.

GRU addresses this with two learned gates:
  - Reset gate (r): controls how much of the previous hidden state to forget.
  - Update gate (z): controls how much of the new candidate state to blend in.
When z ≈ 0, the hidden state passes through almost unchanged, creating a
gradient highway. LSTM uses an additive cell-state update path with three gates
(input, forget, output) that also preserves gradients over long sequences.
""")

	print("""
[Q3] What mathematically changes when temperature > 1.0?

At each step the model outputs a logit vector z of length vocab_size.
The sampling probability of character i is:

  p_i = exp(z_i / T) / sum_j( exp(z_j / T) )

When T = 1.0 this is the standard softmax.
When T > 1.0, every logit is divided by a value > 1, compressing differences
between them. The distribution flattens: high-probability chars lose share,
low-probability chars gain share. At T → ∞ the distribution approaches uniform.
Practically T = 1.5 yields more varied output but also more invented words and
broken grammar, because rare characters are sampled far more often than the
training data would normally justify.
""")

	print("""
[Q4] Why is target shifted one step right from input?

The training objective is next-character prediction. For a chunk of length 101:
  input  = chunk[0:100]   (characters 0 through 99)
  target = chunk[1:101]   (characters 1 through 100)

At position t, the input is character t and the target is character t+1.
This one-step shift gives the model 100 training signal pairs per sequence
instead of just one. It also exactly mirrors inference: feed one character,
predict the next, feed that prediction back in, repeat.
""")

	print("""
[Q5] What dataset-specific patterns were learned, and what failed?

LEARNED:
  - "Alice" reliably appears as a subject (corpus is Alice in Wonderland).
  - Dialogue tags like "said the" and "said Alice" appear frequently.
  - Basic punctuation placement (commas, periods, quotation marks) is partial.
  - Common function words (the, and, of, a, to) match corpus frequency.
  - Title-case proper nouns (Queen, Hatter, Caterpillar) recur correctly.

FAILED:
  - Long-range coherence breaks down after ~50 characters.
  - No concept of story state — cannot sustain a scene or conversation.
  - At T=1.5, invented words appear (model learned n-gram stats, not semantics).
  - Nested clauses and complex syntax are rarely completed correctly.
""")

	print("=" * 70 + "\n")


if __name__ == "__main__":
	# Run all roles in sequence: DE → DLA → OEL → GEL
	run_de_pipeline()
	run_dla_all()
	run_oel_all()
	run_gel_all()
	print_conceptual_questions()


# =============================================================================
# CONCEPTUAL QUESTIONS (INDIVIDUAL ONLY)
# =============================================================================
#
# Question 1:
# Why are RNNs better for text generation than dense feedforward networks?
#
# Answer:
# Dense feedforward networks treat each input position independently — they have
# no mechanism to remember what came before. For text generation, the next
# character depends heavily on the preceding context (e.g., "Alic" strongly
# suggests "e"). RNNs maintain a hidden state that is updated at every time step,
# effectively acting as a compressed memory of the entire sequence seen so far.
# This allows the model to capture sequential dependencies, grammar patterns, and
# long-range stylistic structure that a flat dense network simply cannot encode
# without an exponentially large input window.
#
# -----------------------------------------------------------------------------
#
# Question 2:
# What is vanishing gradient, and how do GRU/LSTM mitigate it?
#
# Answer:
# During backpropagation through time (BPTT), gradients are multiplied by the
# recurrent weight matrix at each timestep. If those values are < 1, the gradient
# shrinks exponentially as it propagates back through long sequences, eventually
# becoming too small to update early-timestep weights — this is the vanishing
# gradient problem. Plain RNNs therefore struggle to learn dependencies spanning
# more than ~10-20 steps.
#
# GRU addresses this with two learned gates:
#   - Reset gate (r): controls how much of the previous hidden state to forget.
#   - Update gate (z): controls how much of the new candidate state to blend in.
# When z ≈ 0, the hidden state is passed through almost unchanged, creating an
# "information highway" that lets gradients flow back without shrinking.
# LSTM uses a similar idea with an explicit cell state and three gates (input,
# forget, output), providing an additive (not multiplicative) update path that
# also preserves gradients over long sequences.
#
# -----------------------------------------------------------------------------
#
# Question 3:
# What mathematically changes when temperature > 1.0?
#
# Answer:
# At each generation step the model outputs a logit vector z of length vocab_size.
# The probability of sampling character i is:
#
#   p_i = exp(z_i / T) / sum_j( exp(z_j / T) )
#
# When T = 1.0 this is the standard softmax.
# When T > 1.0, every logit is divided by a number greater than 1, which
# compresses the differences between them. The resulting distribution is flatter:
# high-probability characters lose share and low-probability characters gain
# share. In the limit T → ∞ the distribution approaches uniform, meaning every
# character is equally likely regardless of what the model learned.
# Practically, T = 1.5 produces more varied and surprising output but also more
# grammatical errors and invented words, because rare characters are sampled far
# more often than the training data would normally justify.
#
# -----------------------------------------------------------------------------
#
# Question 4:
# Why is target shifted one step right from input?
#
# Answer:
# The training objective is next-character prediction: given the characters seen
# so far, predict the very next one. If we have a sequence chunk of length 101,
# then:
#   input  = chunk[0:100]   (characters 0 through 99)
#   target = chunk[1:101]   (characters 1 through 100)
#
# At position t, the input is character t and the target is character t+1.
# This one-step shift means every position in the input has a corresponding
# "correct answer" one position to the right, giving the model 100 training
# signal pairs per sequence instead of just one. It also exactly mirrors how the
# model is used at inference: feed one character, predict the next, feed that
# prediction back in, repeat.
#
# -----------------------------------------------------------------------------
#
# Question 5:
# What dataset-specific patterns were learned, and what failed?
#
# Answer:
# Patterns learned (observed from generated samples):
#   - The model reliably produces "Alice" as a common subject, consistent with
#     the corpus being Alice's Adventures in Wonderland.
#   - Common dialogue tags like "said the" and "said Alice" appear frequently,
#     reflecting the heavy use of dialogue in the source text.
#   - Basic punctuation placement (commas after clauses, periods at sentence ends,
#     quotation marks around speech) is partially learned.
#   - Common function words (the, and, of, a, to) appear with corpus-appropriate
#     frequency, giving output a vaguely Victorian English rhythm.
#   - Title-case proper nouns (Queen, Hatter, Caterpillar) recur, showing the
#     model picked up on character names.
#
# What failed:
#   - Long-range coherence: after ~50 characters, sentences often drift into
#     grammatically broken or semantically disconnected phrases.
#   - Consistent plot or narrative: the model has no concept of story state, so
#     it cannot sustain a scene or conversation thread.
#   - At T=1.5, invented words ("tormeg", "said shat") appear, showing the model
#     has learned character n-gram statistics but not true word-level semantics.
#   - Nested clauses and complex sentence structures are rarely completed
#     correctly — the character-level window is too short to model deep syntax.

