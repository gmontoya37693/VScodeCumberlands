"""
Germán Montoya (ID 5037693)
MSDS 630 B01 Natural Language Processing, Fall 2026 Second Bi-Term
Instructor: Dr. Eve Thullen
Exam 3: Bayes Sentiment Classifier

Verifies the Naive Bayes classification of the test sentence
"pretty enjoyable plot" using add-1 smoothing. Each step prints
its result so the hand calculation can be checked line by line.
"""

from fractions import Fraction
from math import log, exp
from collections import Counter

# Step 1: the training set and the test sentence, exactly as given
training = [
    ("-", "the movie has no plot"),
    ("-", "honestly pretty boring"),
    ("+", "pretty interesting movie"),
]
test = "pretty enjoyable plot"
classes = ["-", "+"]
print("Step 1. Training set")
for label, doc in training:
    print(f"        {label}  {doc}")
print(f"        Test sentence: {test}")

# Step 2: vocabulary = distinct words across ALL training documents
vocab = sorted({w for _, doc in training for w in doc.split()})
print("\nStep 2. Vocabulary:", vocab)
print("        |V| =", len(vocab))

# Step 3: priors = documents in class / total documents
print("\nStep 3. Priors")
prior = {}
for c in classes:
    n_docs = sum(1 for label, _ in training if label == c)
    prior[c] = Fraction(n_docs, len(training))
    print(f"        P({c}) = {n_docs}/{len(training)} = {prior[c]}")

# Step 4: merge each class's documents into one bag of words and count
print("\nStep 4. Word counts per class")
counts = {c: Counter() for c in classes}
for label, doc in training:
    counts[label].update(doc.split())
for c in classes:
    print(f"        class {c}: {sum(counts[c].values())} words -> {dict(counts[c])}")

# Step 5: add-1 likelihood = (count + 1) / (words in class + |V|)
def likelihood(w, c):
    return Fraction(counts[c][w] + 1, sum(counts[c].values()) + len(vocab))

print("\nStep 5. Likelihoods with add-1 smoothing")
for w in vocab:
    print(f"        {w:<12} P(w|-) = {likelihood(w, '-')}   P(w|+) = {likelihood(w, '+')}")

# Step 6: drop test words that are not in the vocabulary
kept = [w for w in test.split() if w in vocab]
dropped = [w for w in test.split() if w not in vocab]
print("\nStep 6. Test words kept:", kept, "| dropped:", dropped)

# Step 7: score each class: prior x likelihoods, then the same in log space
print("\nStep 7. Scores")
log_score = {}
for c in classes:
    score = prior[c]
    parts = [f"{prior[c]}"]
    for w in kept:
        score *= likelihood(w, c)
        parts.append(f"P({w}|{c})={likelihood(w, c)}")
    log_score[c] = log(prior[c]) + sum(log(likelihood(w, c)) for w in kept)
    print(f"        {c}: {' x '.join(parts)}")
    print(f"           = {score}  (decimal {float(score):.5f}, log {log_score[c]:.4f})")

# Step 8: argmax = the class with the highest score
winner = max(classes, key=lambda c: log_score[c])
ratio = exp(log_score["-"] - log_score["+"])
print(f"\nStep 8. Prediction (argmax): {winner}   negative/positive ratio = {ratio:.2f}")