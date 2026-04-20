README.txt
==========

Programming Exercise 4 – RBM Image Reconstruction
Course: 2026 Spring – Deep Learning (MSDS-534-M40)
Student: German Montoya
Date: April 19, 2026


1. Purpose of the Assignment
----------------------------
This assignment demonstrates unsupervised representation learning using a
Restricted Boltzmann Machine (RBM). The RBM is trained on unlabeled MNIST images
to learn latent structure in the data and to reconstruct input images from
learned hidden representations. Reconstruction quality is used as qualitative
evidence that the model captures meaningful features rather than memorizing
noise.


2. Contents of This Submission
------------------------------
- rbm_reconstruction.py
  A self-contained TensorFlow 2 script that:
  * Implements an RBM with Bernoulli visible and hidden units
  * Trains the model using Contrastive Divergence (CD-1)
  * Reconstructs held-out MNIST test images
  * Plots original and reconstructed images for comparison

- Output artifacts:
  * Console logs showing training progress and reconstruction error
  * Image file: rbm_reconstruction_results.png (original vs reconstructed images)


3. Key Modeling Decisions
------------------------
- Dataset: MNIST with standard train/test split
- Preprocessing:
  * Pixel normalization to [0, 1]
  * Binarization to support Bernoulli visible units
- Architecture:
  * 784 visible units (28x28 flattened images)
  * 128 hidden units (latent representation)
- Training:
  * Contrastive Divergence with CD-1 (single Gibbs sampling step)
  * Learning rate = 0.01, batch size = 64, epochs = 30
- Initialization:
  * Small random weights near zero, biases initialized to zero

These choices follow standard RBM practice and prioritize training stability
and interpretability over aggressive tuning.


4. Interpretation of Results
----------------------------
- Reconstruction error decreased monotonically (~63% reduction), indicating
  stable and well-behaved training.
- Reconstructed digits remain clearly recognizable but appear smoother and
  slightly blurred.
- Loss of fine detail is expected for probabilistic RBMs trained with CD-1 and
  reflects averaging over likely configurations rather than model failure.
- Overall reconstruction quality indicates that the RBM learned global digit
  structure such as strokes and shapes.


5. Connection to Previous Assignments
-------------------------------------
This assignment completes a conceptual sequence across the course:

- Week 7 (CNN Writing Recognition):
  Supervised learning focused on classification accuracy, with representations
  learned implicitly.

- Week 12 (Greedy Unsupervised Pretraining):
  Deterministic autoencoders used reconstruction as a local learning signal to
  build hierarchical representations without labels.

- This Assignment (RBM Reconstruction):
  Introduces probabilistic, generative representation learning and shows how
  reconstruction arises from modeling the data distribution itself.

RBMs historically motivated greedy unsupervised pretraining and provide the
theoretical grounding for reconstruction-based representation learning used
later in the course.


6. Notes
--------
- The "1" in CD-1 refers to one Gibbs sampling step during training, not the
  number of hidden layers.
- An RBM always consists of one visible layer and one hidden layer; depth
  emerges only when stacking multiple RBMs (not done here).


7. References
-------------
Hinton, G. E. (2010). A Practical Guide to Training Restricted Boltzmann Machines.
Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
Gulli, A., Kapoor, A., & Pal, S. (2019). Deep Learning with TensorFlow 2 and Keras,
Chapter 10.