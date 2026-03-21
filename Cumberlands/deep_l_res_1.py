"""
2026 Spring - Deep Learning (MSDS-534-M40)
Residency Day 1 - Project 1: Distance Metrics and Regularization

Objective:
- Complete Project 1 by analyzing L1 and L2 norms in two contexts:
	distance metrics between data vectors and regularization penalties in deep learning.
- Demonstrate metric properties (positivity, symmetry, triangle inequality), compare
	Hamming vs L2 behavior on binary encodings, and evaluate the cosine-based measure
	d(x, y) = 1 - cos(x, y) using proof/counterexample reasoning.
- Synthesize how norm selection affects geometric closeness, outlier sensitivity,
	sparsity, and weight behavior under regularization.

Student Information:
- Student Name: German Montoya
- Student ID: 5037693
- Course: MSDS-534-M40 (Deep Learning)
- Instructor: Dr. Intisar Rizwan I Haque
- Term: Spring 2026
- Date: 2026-03-20

INPUTS:
- Manually chosen vectors (2D/3D and binary 4D examples).
- Notebook outputs from: L1 and L2 Regularization.ipynb
- Notes from required tutorials/resources.

OUTPUTS:
- A complete report (PDF/Word) with Sections 1-5.
- Mathematical derivations, proof sketches, and numeric examples.
- Figures/tables (or equivalent numeric summaries) from regularization experiments.
- Final synthesis connecting distance metrics and regularization.

HOW TO USE THIS FILE:
- This file is a guided working draft for calculations, notes, and code snippets.
- We will complete each section in small steps.
"""

import argparse
import math
from pathlib import Path


NOTEBOOK_RESULTS = {
    "l1_figure_label": "Lasso figure/table title from Colab",
    "l2_figure_label": "Ridge figure/table title from Colab",
    "l1_train_loss": None,
    "l1_val_loss": None,
    "l2_train_loss": None,
    "l2_val_loss": None,
    "l1_near_zero_weights": None,
    "l2_near_zero_weights": None,
}


def l1_distance(vector_x, vector_y):
    return sum(abs(value_x - value_y) for value_x, value_y in zip(vector_x, vector_y))


def l2_distance(vector_x, vector_y):
    return math.sqrt(sum((value_x - value_y) ** 2 for value_x, value_y in zip(vector_x, vector_y)))


def hamming_distance(vector_x, vector_y):
    return sum(value_x != value_y for value_x, value_y in zip(vector_x, vector_y))


def cosine_similarity(vector_x, vector_y):
    dot_product = sum(value_x * value_y for value_x, value_y in zip(vector_x, vector_y))
    norm_x = math.sqrt(sum(value ** 2 for value in vector_x))
    norm_y = math.sqrt(sum(value ** 2 for value in vector_y))
    if norm_x == 0 or norm_y == 0:
        raise ValueError("Cosine similarity is undefined for the zero vector.")
    return dot_product / (norm_x * norm_y)


def cosine_distance(vector_x, vector_y):
    return 1 - cosine_similarity(vector_x, vector_y)


def print_section_1_part_a():
    pair_1_x = (1, 3, 2)
    pair_1_y = (4, 1, 6)
    pair_2_x = (2, 5)
    pair_2_y = (-1, 9)

    pair_1_l1 = l1_distance(pair_1_x, pair_1_y)
    pair_1_l2 = l2_distance(pair_1_x, pair_1_y)
    pair_2_l1 = l1_distance(pair_2_x, pair_2_y)
    pair_2_l2 = l2_distance(pair_2_x, pair_2_y)

    print("SECTION 1 - PART A: Manual Distance Calculations")
    print()
    print("Pair 1 (3-dimensional): x = (1, 3, 2), y = (4, 1, 6)")
    print("L1 distance:")
    print("||x - y||_1 = |1 - 4| + |3 - 1| + |2 - 6|")
    print("           = |-3| + |2| + |-4|")
    print(f"           = 3 + 2 + 4 = {pair_1_l1}")
    print()
    print("L2 distance:")
    print("||x - y||_2 = sqrt((1 - 4)^2 + (3 - 1)^2 + (2 - 6)^2)")
    print("           = sqrt((-3)^2 + 2^2 + (-4)^2)")
    print(f"           = sqrt(9 + 4 + 16) = sqrt(29) = {pair_1_l2:.4f}")
    print()
    print("Pair 2 (2-dimensional): x = (2, 5), y = (-1, 9)")
    print("L1 distance:")
    print("||x - y||_1 = |2 - (-1)| + |5 - 9|")
    print("           = |3| + |-4|")
    print(f"           = 3 + 4 = {pair_2_l1}")
    print()
    print("L2 distance:")
    print("||x - y||_2 = sqrt((2 - (-1))^2 + (5 - 9)^2)")
    print("           = sqrt(3^2 + (-4)^2)")
    print(f"           = sqrt(9 + 16) = sqrt(25) = {pair_2_l2:.4f}")
    print()
    print("Observation:")
    print("For both vector pairs, the L1 distance is larger than the L2 distance.")
    print("L1 adds absolute component-wise differences directly, while L2 combines")
    print("squared differences and compresses the total using the square root.")


def print_section_1_part_b_placeholders(results):
    print("\nSECTION 1 - PART B: Regularization Impact via Colab")
    print()
    print(f"L1 evidence to include: {results['l1_figure_label']}")
    print(f"L2 evidence to include: {results['l2_figure_label']}")
    print("Use Colab outputs for train/validation loss and delta values.")


def print_section_2_numeric_check():
    # Reuse Section 1 vectors and add one extra 3D vector for triangle checks.
    x = (1, 3, 2)
    y = (4, 1, 6)
    z = (2, 0, 5)

    d1_xy = l1_distance(x, y)
    d1_yx = l1_distance(y, x)
    d1_xz = l1_distance(x, z)
    d1_yz = l1_distance(y, z)

    d2_xy = l2_distance(x, y)
    d2_yx = l2_distance(y, x)
    d2_xz = l2_distance(x, z)
    d2_yz = l2_distance(y, z)

    print("\nSECTION 2 - NUMERICAL CHECK (L1 and L2 Metric Properties)")
    print()
    print(f"Vectors used: x = {x}, y = {y}, z = {z}")

    print("\nPositivity and identity:")
    print(f"L1 d(x,y) = {d1_xy:.4f} >= 0 ? {d1_xy >= 0}")
    print(f"L2 d(x,y) = {d2_xy:.4f} >= 0 ? {d2_xy >= 0}")
    print(f"L1 d(x,x) = {l1_distance(x, x):.4f}")
    print(f"L2 d(x,x) = {l2_distance(x, x):.4f}")

    print("\nSymmetry:")
    print(
        f"L1 d(x,y) = {d1_xy:.4f}, d(y,x) = {d1_yx:.4f}, equal? {abs(d1_xy - d1_yx) < 1e-12}"
    )
    print(
        f"L2 d(x,y) = {d2_xy:.4f}, d(y,x) = {d2_yx:.4f}, equal? {abs(d2_xy - d2_yx) < 1e-12}"
    )

    print("\nTriangle inequality:")
    print(
        f"L1 d(x,z) = {d1_xz:.4f} <= d(x,y)+d(y,z) = {(d1_xy + d1_yz):.4f} ? {d1_xz <= d1_xy + d1_yz + 1e-12}"
    )
    print(
        f"L2 d(x,z) = {d2_xz:.4f} <= d(x,y)+d(y,z) = {(d2_xy + d2_yz):.4f} ? {d2_xz <= d2_xy + d2_yz + 1e-12}"
    )


def print_section_3_hamming_vs_l2():
    # Binary vectors for Section 3 Part B (4-dimensional as requested).
    x = (1, 0, 0, 0)
    y = (1, 1, 0, 0)
    z = (1, 1, 1, 0)

    pairs = [
        ("d(x,y)", x, y),
        ("d(x,z)", x, z),
        ("d(y,z)", y, z),
    ]

    print("\nSECTION 3 - HAMMING VS L2 ON BINARY VECTORS")
    print("\nPart A - Conceptual explanation:")
    print(
        "For binary vectors, each mismatched coordinate contributes 1 to Hamming distance."
    )
    print(
        "In L2, each mismatch contributes (1-0)^2 = 1 inside the sum, so L2 becomes sqrt(number of mismatches)."
    )
    print(
        "Therefore, although numeric values differ, the closer-vs-farther ordering can be preserved."
    )

    print("\nPart B - Concrete mathematical example:")
    print(f"x = {x}")
    print(f"y = {y}")
    print(f"z = {z}")
    print("\nPairwise distances:")

    values = {}
    for label, left_vector, right_vector in pairs:
        hamming = hamming_distance(left_vector, right_vector)
        l2 = l2_distance(left_vector, right_vector)
        values[label] = (hamming, l2)
        print(f"{label}: Hamming = {hamming}, L2 = sqrt({hamming}) = {l2:.4f}")

    print("\nOrdering check:")
    print(
        f"Under Hamming: d(x,y)={values['d(x,y)'][0]}, d(y,z)={values['d(y,z)'][0]}, d(x,z)={values['d(x,z)'][0]}"
    )
    print(
        f"Under L2: d(x,y)={values['d(x,y)'][1]:.4f}, d(y,z)={values['d(y,z)'][1]:.4f}, d(x,z)={values['d(x,z)'][1]:.4f}"
    )
    print(
        "Conclusion: the relative ordering is identical in this example because d(x,z) is largest under both metrics."
    )


def print_section_4_cosine_measure():
    # Example vectors for positivity/symmetry and triangle inequality checks.
    x = (1, 0)
    y = (2, 0)
    z = (1, 1)
    w = (0, 1)

    d_xy = cosine_distance(x, y)
    d_yx = cosine_distance(y, x)

    d_xw = cosine_distance(x, w)
    d_xz = cosine_distance(x, z)
    d_zw = cosine_distance(z, w)

    print("\nSECTION 4 - COSINE-BASED MEASURE: d(x,y) = 1 - cos(x,y)")
    print("\nPositivity check:")
    print(f"d(x,y) with x={x}, y={y} is {d_xy:.4f} (nonnegative)")

    print("\nIdentity counterexample (required for true metric positivity condition):")
    print(f"x = {x}, y = {y}, x != y")
    print(
        f"cos(x,y) = {cosine_similarity(x, y):.4f}, so d(x,y) = 1 - cos(x,y) = {d_xy:.4f}"
    )
    print("Because d(x,y)=0 while x!=y, the identity condition fails.")

    print("\nSymmetry check:")
    print(f"d(x,y) = {d_xy:.4f}, d(y,x) = {d_yx:.4f}, equal? {abs(d_xy - d_yx) < 1e-12}")

    print("\nTriangle inequality counterexample:")
    print(f"Using x={x}, z={z}, w={w}")
    print(f"d(x,w) = {d_xw:.4f}")
    print(f"d(x,z) = {d_xz:.4f}")
    print(f"d(z,w) = {d_zw:.4f}")
    print(f"d(x,z) + d(z,w) = {d_xz + d_zw:.4f}")
    print(f"Triangle inequality d(x,w) <= d(x,z)+d(z,w)? {d_xw <= d_xz + d_zw + 1e-12}")

    print("\nConclusion:")
    print("The measure d(x,y)=1-cos(x,y) is symmetric and nonnegative for nonzero vectors,")
    print("but it is NOT a true metric because it fails the identity condition and")
    print("fails triangle inequality (as shown above).")


# ============================================================
# SECTION 5 - Importance of Choosing a Good Norm (Draft Text)
# ============================================================
# Part A - Distance context:
# The choice of norm changes which points are considered close in feature space.
# L1 measures additive coordinate-wise difference, L2 measures Euclidean
# straight-line distance, and cosine-based distance emphasizes direction rather
# than magnitude. In sparse settings, L1 is often easier to interpret and aligns
# well with mismatch-style reasoning. L2 is geometrically natural in continuous
# spaces, but it emphasizes larger deviations through squaring. From earlier
# results: Section 1 showed that L1 and L2 can rank distances differently in
# magnitude, and Section 3 showed that Hamming and L2 can preserve ordering on
# binary encodings. Section 4 showed that 1 - cosine is useful for directional
# comparison but is not a true metric.
#
# Part B - Regularization context:
# In model training, the same norms are applied to weights rather than data-point
# differences. L1 regularization penalizes sum(|w_i|), encouraging sparsity and
# feature selection by pushing weaker coefficients toward zero. L2 regularization
# penalizes sum(w_i^2), shrinking all coefficients smoothly and supporting stable
# weight decay. Increasing alpha (or lambda) strengthens both effects: larger
# alpha makes L1 more sparse and L2 more strongly shrunk. This matches the Part B
# observations from Colab, where Lasso simplified more aggressively while Ridge
# produced smoother shrinkage.
#
# Final synthesis (2-3 sentences):
# Distance metrics and regularization are two applications of the same norm-based
# mathematics. In feature space, norms define closeness between observations; in
# deep learning, norms define how strongly large weights are penalized. Therefore,
# norm choice directly affects both geometric interpretation and generalization.


def generate_part_b_figures_and_table_local(output_dir="."):
    # Lazy imports prevent local environment issues from blocking Part A.
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    except Exception as import_error:
        print("Part B local generation failed during imports.")
        print("Reason:", import_error)
        print("Use Colab for Part B figures, or repair local sklearn/pyarrow dependencies.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    m = 120
    X = 6 * np.random.rand(m, 1) - 3
    y = 2 + X[:, 0] + 0.5 * X[:, 0] ** 2 + np.random.randn(m)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    ridge = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=10, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )

    lasso = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=10, include_bias=False)),
            ("scale", StandardScaler()),
            ("model", Lasso(alpha=0.01, max_iter=20000)),
        ]
    )

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    ridge_train = ridge.predict(X_train)
    ridge_val = ridge.predict(X_val)
    lasso_train = lasso.predict(X_train)
    lasso_val = lasso.predict(X_val)

    ridge_train_mse = mean_squared_error(y_train, ridge_train)
    ridge_val_mse = mean_squared_error(y_val, ridge_val)
    lasso_train_mse = mean_squared_error(y_train, lasso_train)
    lasso_val_mse = mean_squared_error(y_val, lasso_val)

    ridge_delta = ridge_val_mse - ridge_train_mse
    lasso_delta = lasso_val_mse - lasso_train_mse

    ridge_coef = ridge.named_steps["model"].coef_
    lasso_coef = lasso.named_steps["model"].coef_

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(lasso_coef)), lasso_coef)
    plt.title("Figure 1: L1 Regularization (Lasso) Coefficients")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Weight Value")
    plt.tight_layout()
    l1_file = output_path / "part_b_l1_lasso_coefficients.png"
    plt.savefig(l1_file, dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(range(len(ridge_coef)), ridge_coef)
    plt.title("Figure 2: L2 Regularization (Ridge) Coefficients")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Weight Value")
    plt.tight_layout()
    l2_file = output_path / "part_b_l2_ridge_coefficients.png"
    plt.savefig(l2_file, dpi=200)
    plt.close()

    summary = pd.DataFrame(
        {
            "Model": ["L1 (Lasso)", "L2 (Ridge)"],
            "Train Loss (MSE)": [lasso_train_mse, ridge_train_mse],
            "Validation Loss (MSE)": [lasso_val_mse, ridge_val_mse],
            "Delta (Val - Train)": [lasso_delta, ridge_delta],
            "Near-zero Weights (|w| < 1e-3)": [
                int(np.sum(np.abs(lasso_coef) < 1e-3)),
                int(np.sum(np.abs(ridge_coef) < 1e-3)),
            ],
        }
    )

    table_file = output_path / "part_b_summary_table.csv"
    summary.to_csv(table_file, index=False)

    print("\nPart B summary table:")
    print(summary.to_string(index=False))
    print("\nSaved files:")
    print(f"- {l1_file}")
    print(f"- {l2_file}")
    print(f"- {table_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate assignment support output for Deep Learning Residency Day 1."
    )
    parser.add_argument(
        "--part-b-local",
        action="store_true",
        help="Generate Part B figures and summary table locally and save files.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save Part B files when --part-b-local is used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_section_1_part_a()
    print_section_1_part_b_placeholders(NOTEBOOK_RESULTS)
    print_section_2_numeric_check()
    print_section_3_hamming_vs_l2()
    print_section_4_cosine_measure()
    if args.part_b_local:
        generate_part_b_figures_and_table_local(output_dir=args.output_dir)
