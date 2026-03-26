"""
2026 Spring - Deep Learning (MSDS-534-M40)
Programming Exercise 3: Unsupervised Pretraining

Student: German Montoya
University: University of the Cumberlands
Program Area: Information Technology Sciences
Instructor: Dr. Intisar Rizwan I. Haque
Date: 2026-03-26

OBJECTIVE:
- Implement the greedy layer-wise unsupervised pretraining protocol using Fashion-MNIST.
- Use a 3-layer encoder stack trained one stage at a time with reconstruction loss.
- Clearly distinguish between the encoded representation that is passed to the next stage
	and the reconstruction that is used only to train the current stage.
- Provide code comments and simple visualizations that explain how the input shape is
	reduced across stages and what each layer learns to reconstruct.

INPUTS:
- Fashion-MNIST training set as unlabeled raw input data.
- Images normalized to the range [0, 1].
- Flattened input vectors of size 784 (28 x 28).
- Three encoder stages, for example:
	1) 784 -> 256
	2) 256 -> 128
	3) 128 -> 64

OUTPUTS:
- A Python implementation of greedy layer-wise unsupervised pretraining.
- Encoded representations produced at each stage: h1, h2, and h3.
- Reconstructions used to compute loss at each stage.
- Brief explanatory comments or markdown-style notes describing the training logic.
- Simple plots that show:
	1) an original Fashion-MNIST image,
	2) a reconstruction,
	3) the encoded representation passed to the next stage.

NOTES:
- During pretraining, each stage is trained as a shallow autoencoder.
- The encoder output is passed forward to the next stage.
- The decoder reconstruction is not passed forward; it is used only for the
	reconstruction loss of the current stage.

UPDATED PSEUDOCODE (MATCHES THIS SCRIPT):
1) Load Fashion-MNIST, normalize to [0, 1], flatten each image to 784 features.
2) Stage 1 pretraining:
	- Train AE1 on x to reconstruct x.
	- Encoder1: 784 -> 256 (ReLU)
	- Decoder1: 256 -> 784 (sigmoid, because x is in [0, 1])
	- Keep h1 = Encoder1(x) as passed-forward representation.
3) Stage 2 pretraining:
	- Train AE2 on h1 to reconstruct h1.
	- Encoder2: 256 -> 128 (ReLU)
	- Decoder2: 128 -> 256 (linear, latent-space reconstruction)
	- Keep h2 = Encoder2(h1) as passed-forward representation.
4) Stage 3 pretraining:
	- Train AE3 on h2 to reconstruct h2.
	- Encoder3: 128 -> 64 (ReLU)
	- Decoder3: 64 -> 128 (linear, latent-space reconstruction)
	- Keep h3 = Encoder3(h2) as final representation.
5) Report what is passed vs reconstruction-only:
	- Passed forward: h1 -> h2 -> h3
	- Reconstruction only: x_hat, h1_hat, h2_hat
6) Analysis-only visualization (not training objective):
	- Project deeper codes to image space through chained decoders:
	  h2 -> h1_hat -> x_hat
	  h3 -> h2_hat -> h1_hat -> x_hat
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
	# Prefer standalone Keras to reduce editor import-resolution false positives.
	from keras.datasets import fashion_mnist
	from keras.layers import Dense, Input
	from keras.models import Model
	from keras.optimizers import Adam
except ImportError as import_error:
	try:
		import tensorflow as tf

		fashion_mnist = tf.keras.datasets.fashion_mnist
		Dense = tf.keras.layers.Dense
		Input = tf.keras.layers.Input
		Model = tf.keras.models.Model
		Adam = tf.keras.optimizers.Adam
	except ImportError:
		raise ImportError(
			"TensorFlow/Keras is required for this script. "
			"Install with: pip install tensorflow"
		) from import_error


def load_fashion_mnist_flattened():
	"""Load Fashion-MNIST, normalize to [0, 1], and flatten images to vectors."""
	(x_train, _), (x_test, _) = fashion_mnist.load_data()

	x_train = x_train.astype("float32") / 255.0
	x_test = x_test.astype("float32") / 255.0

	x_train_flat = x_train.reshape(len(x_train), 784)
	x_test_flat = x_test.reshape(len(x_test), 784)
	return x_train, x_test, x_train_flat, x_test_flat


def build_shallow_autoencoder(input_dim, latent_dim, stage_id, learning_rate=1e-3):
	"""
	Build one stage of greedy pretraining.

	This stage is an autoencoder used only for local reconstruction training:
	input -> encoder(latent) -> decoder(reconstruction of current input)
	"""
	inputs = Input(shape=(input_dim,), name=f"stage{stage_id}_input")
	encoded = Dense(latent_dim, activation="relu", name=f"stage{stage_id}_encoder")(inputs)

	# For stage 1 we reconstruct normalized pixels in [0, 1], so sigmoid is natural.
	# For deeper stages, linear keeps reconstruction flexible for hidden features.
	decoder_activation = "sigmoid" if stage_id == 1 else "linear"
	decoded = Dense(input_dim, activation=decoder_activation, name=f"stage{stage_id}_decoder")(encoded)

	autoencoder = Model(inputs=inputs, outputs=decoded, name=f"autoencoder_stage_{stage_id}")
	encoder = Model(inputs=inputs, outputs=encoded, name=f"encoder_stage_{stage_id}")

	autoencoder.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
	return autoencoder, encoder


def pretrain_one_stage(stage_id, x_train_current, x_test_current, latent_dim, epochs=5, batch_size=256):
	"""
	Train one greedy layer-wise stage and return representations plus reconstructions.

	Important distinction:
	- Encoder output is passed to the next stage.
	- Decoder reconstruction is used only for this stage's loss.
	"""
	input_dim = x_train_current.shape[1]
	autoencoder, encoder = build_shallow_autoencoder(input_dim, latent_dim, stage_id)

	history = autoencoder.fit(
		x_train_current,
		x_train_current,
		epochs=epochs,
		batch_size=batch_size,
		validation_split=0.1,
		verbose=1,
	)

	train_encoded = encoder.predict(x_train_current, verbose=0)
	test_encoded = encoder.predict(x_test_current, verbose=0)
	test_reconstructed = autoencoder.predict(x_test_current, verbose=0)

	return {
		"stage_id": stage_id,
		"input_dim": input_dim,
		"latent_dim": latent_dim,
		"autoencoder": autoencoder,
		"encoder": encoder,
		"history": history,
		"train_encoded": train_encoded,
		"test_encoded": test_encoded,
		"test_reconstructed": test_reconstructed,
	}


def plot_stage1_image_reconstruction(x_test_images, stage1_recon, output_path, show_plot=True):
	"""Plot one original image and the stage-1 reconstruction (pixel space)."""
	fig = plt.figure(figsize=(8, 3))

	plt.subplot(1, 2, 1)
	plt.imshow(x_test_images[0], cmap="gray")
	plt.title("Original x (28x28)")
	plt.axis("off")

	plt.subplot(1, 2, 2)
	plt.imshow(stage1_recon[0].reshape(28, 28), cmap="gray")
	plt.title("x_hat from Stage 1\n(reconstruction only)")
	plt.axis("off")

	plt.tight_layout()
	plt.savefig(output_path, dpi=140)
	if show_plot:
		plt.show()
	else:
		plt.close(fig)


def plot_passed_vs_reconstructed(stage1, stage2, stage3, output_path, show_plot=True):
	"""Show what is passed forward versus what is reconstructed for loss only."""
	fig, axes = plt.subplots(5, 1, figsize=(10, 7))

	axes[0].imshow(stage1["test_encoded"][0][None, :], aspect="auto", cmap="viridis")
	axes[0].set_title("h1 = Encoder1(x) -> passed to Stage 2")

	axes[1].imshow(stage2["test_reconstructed"][0][None, :], aspect="auto", cmap="magma")
	axes[1].set_title("h1_hat = Decoder2(h2) -> reconstruction for Stage 2 loss only")

	axes[2].imshow(stage2["test_encoded"][0][None, :], aspect="auto", cmap="viridis")
	axes[2].set_title("h2 = Encoder2(h1) -> passed to Stage 3")

	axes[3].imshow(stage3["test_reconstructed"][0][None, :], aspect="auto", cmap="magma")
	axes[3].set_title("h2_hat = Decoder3(h3) -> reconstruction for Stage 3 loss only")

	axes[4].imshow(stage3["test_encoded"][0][None, :], aspect="auto", cmap="viridis")
	axes[4].set_title("h3 = Encoder3(h2) -> final 3-layer representation")

	for axis in axes:
		axis.set_yticks([])
		axis.set_xticks([])

	plt.tight_layout()
	plt.savefig(output_path, dpi=140)
	if show_plot:
		plt.show()
	else:
		plt.close(fig)


def decode_latent_to_image_space(stage1, stage2, stage3, sample_index=0):
	"""
	Build intuitive pixel-space reconstructions from h1/h2/h3.

	These are analysis visualizations. During greedy pretraining, h2 and h3 are not
	directly trained to reconstruct x; they reconstruct h1 and h2 respectively.
	"""
	# Decoder from stage 1 maps h1 -> x_hat.
	decoder1 = stage1["autoencoder"].get_layer("stage1_decoder")
	# Decoder from stage 2 maps h2 -> h1_hat.
	decoder2 = stage2["autoencoder"].get_layer("stage2_decoder")
	# Decoder from stage 3 maps h3 -> h2_hat.
	decoder3 = stage3["autoencoder"].get_layer("stage3_decoder")

	h1 = stage1["test_encoded"][sample_index : sample_index + 1]
	h2 = stage2["test_encoded"][sample_index : sample_index + 1]
	h3 = stage3["test_encoded"][sample_index : sample_index + 1]

	# Stage-wise direct decode endpoints.
	x_from_h1 = decoder1(h1, training=False).numpy()[0]
	h1_from_h2 = decoder2(h2, training=False).numpy()[0]
	h2_from_h3 = decoder3(h3, training=False).numpy()[0]

	# Chain decoders to project deeper codes back to image space for intuition.
	x_from_h2 = decoder1(h1_from_h2[None, :], training=False).numpy()[0]
	h1_from_h3 = decoder2(h2_from_h3[None, :], training=False).numpy()[0]
	x_from_h3 = decoder1(h1_from_h3[None, :], training=False).numpy()[0]

	return {
		"x_from_h1": x_from_h1,
		"x_from_h2_via_chain": x_from_h2,
		"x_from_h3_via_chain": x_from_h3,
		"h1_from_h2": h1_from_h2,
		"h2_from_h3": h2_from_h3,
	}


def plot_intuitive_image_reconstruction_by_depth(x_test_images, stage1, stage2, stage3, output_path, show_plot=True):
	"""Plot image-space reconstructions from h1/h2/h3 to make representation depth intuitive."""
	decoded = decode_latent_to_image_space(stage1, stage2, stage3, sample_index=0)

	fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
	fig.suptitle("Image-space intuition from passed-forward codes (h1, h2, h3)", fontsize=11)

	axes[0].imshow(x_test_images[0], cmap="gray")
	axes[0].set_title("Original x")
	axes[0].axis("off")

	axes[1].imshow(decoded["x_from_h1"].reshape(28, 28), cmap="gray")
	axes[1].set_title("h1 passed forward\nDecode: h1 -> x_hat")
	axes[1].axis("off")

	axes[2].imshow(decoded["x_from_h2_via_chain"].reshape(28, 28), cmap="gray")
	axes[2].set_title("h2 passed forward\nDecode: h2 -> h1_hat -> x_hat")
	axes[2].axis("off")

	axes[3].imshow(decoded["x_from_h3_via_chain"].reshape(28, 28), cmap="gray")
	axes[3].set_title("h3 passed forward\nDecode: h3 -> h2_hat -> h1_hat -> x_hat")
	axes[3].axis("off")

	fig.text(
		0.5,
		0.01,
		"Note: h1_hat and h2_hat are reconstructions used only for stage loss; they are not passed to the next stage.",
		ha="center",
		fontsize=9,
	)

	plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.92])
	plt.savefig(output_path, dpi=140)
	if show_plot:
		plt.show()
	else:
		plt.close(fig)


def print_shape_reduction_summary(stage1, stage2, stage3):
	"""Print a compact shape table to make stage-wise reduction explicit."""
	print("\nShape reduction across greedy stages")
	print("Stage | Encoder input dim | Encoder output dim | Reconstruction target")
	print("1     | 784               | 256                | x (784)")
	print("2     | 256               | 128                | h1 (256)")
	print("3     | 128               | 64                 | h2 (128)")

	print("\nActual array shapes from this run")
	print(f"x_test_flat: {stage1['test_reconstructed'].shape[0]} samples x {stage1['input_dim']} dims")
	print(f"h1 (passed forward): {stage1['test_encoded'].shape}")
	print(f"h1_hat (reconstruction only): {stage2['test_reconstructed'].shape}")
	print(f"h2 (passed forward): {stage2['test_encoded'].shape}")
	print(f"h2_hat (reconstruction only): {stage3['test_reconstructed'].shape}")
	print(f"h3 (final representation): {stage3['test_encoded'].shape}")


def main():
	# Keep this practical for assignment demos while preserving the full protocol.
	epochs_per_stage = 5
	batch_size = 256
	train_subset = 20000
	show_plots = True

	output_dir = Path(__file__).resolve().parent
	recon_plot_path = output_dir / "pretrain_stage1_reconstruction.png"
	flow_plot_path = output_dir / "pretrain_passed_vs_reconstructed.png"
	depth_recon_plot_path = output_dir / "pretrain_reconstruction_by_depth.png"

	x_train_images, x_test_images, x_train_flat, x_test_flat = load_fashion_mnist_flattened()

	if train_subset is not None:
		x_train_flat = x_train_flat[:train_subset]

	print("Greedy layer-wise unsupervised pretraining started.")
	print(f"Training samples used: {x_train_flat.shape[0]}")

	# Stage 1: x(784) -> h1(256) -> x_hat(784)
	stage1 = pretrain_one_stage(
		stage_id=1,
		x_train_current=x_train_flat,
		x_test_current=x_test_flat,
		latent_dim=256,
		epochs=epochs_per_stage,
		batch_size=batch_size,
	)

	# Stage 2 uses h1 as both input and reconstruction target during training.
	stage2 = pretrain_one_stage(
		stage_id=2,
		x_train_current=stage1["train_encoded"],
		x_test_current=stage1["test_encoded"],
		latent_dim=128,
		epochs=epochs_per_stage,
		batch_size=batch_size,
	)

	# Stage 3 uses h2 as both input and reconstruction target during training.
	stage3 = pretrain_one_stage(
		stage_id=3,
		x_train_current=stage2["train_encoded"],
		x_test_current=stage2["test_encoded"],
		latent_dim=64,
		epochs=epochs_per_stage,
		batch_size=batch_size,
	)

	print_shape_reduction_summary(stage1, stage2, stage3)

	plot_stage1_image_reconstruction(
		x_test_images=x_test_images,
		stage1_recon=stage1["test_reconstructed"],
		output_path=recon_plot_path,
		show_plot=show_plots,
	)
	plot_passed_vs_reconstructed(stage1, stage2, stage3, flow_plot_path, show_plot=show_plots)
	plot_intuitive_image_reconstruction_by_depth(
		x_test_images=x_test_images,
		stage1=stage1,
		stage2=stage2,
		stage3=stage3,
		output_path=depth_recon_plot_path,
		show_plot=show_plots,
	)

	print("\nSaved plots:")
	print(f"- {recon_plot_path.name}")
	print(f"- {flow_plot_path.name}")
	print(f"- {depth_recon_plot_path.name}")
	print(f"\nDisplay mode: {'on-screen windows' if show_plots else 'save only'}")
	print("\nProtocol reminder:")
	print("- Passed to next stage: h1 -> h2 -> h3")
	print("- Used for stage loss only: x_hat, h1_hat, h2_hat")


if __name__ == "__main__":
	main()
