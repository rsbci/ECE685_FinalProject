import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Sequence, Tuple, TypeVar
from sklearn.model_selection import train_test_split

import nltk
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from nltk.corpus import treebank

Q = TypeVar("Q")
V = TypeVar("V")


def viterbi(
    obs: Sequence[int],
    pi: np.ndarray[Tuple[V], np.dtype[np.float_]],
    A: np.ndarray[Tuple[Q, Q], np.dtype[np.float_]],
    B: np.ndarray[Tuple[Q, V], np.dtype[np.float_]],
) -> Tuple[list[int], float]:
    N = len(obs)
    Q, V = B.shape

    epsilon = 1e-10  # Small epsilon value to avoid divide by zero warnings

    log_d = [np.log(pi + epsilon) + np.log(B[:, obs[0]])]
    log_psi = [np.zeros((Q,))]

    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))

    log_ps = np.max(log_d[-1])
    qs = [-1] * N
    qs[-1] = int(np.argmax(log_d[-1]))
    for i in range(N - 2, -1, -1):
        qs[i] = log_psi[i + 1][qs[i + 1]]

    return qs, np.exp(log_ps)


def compute_initial_distribution(tagged_sents):
    tag_counts = defaultdict(int)
    for sent in tagged_sents:
        tag_counts[sent[0][1]] += 1

    total_sents = len(tagged_sents)
    for tag in tag_counts:
        tag_counts[tag] /= total_sents
    return tag_counts


def compute_transition_matrix(tagged_sents, tag_list):
    transitions = defaultdict(lambda: defaultdict(int))
    for sent in tagged_sents:
        for i in range(len(sent) - 1):
            transitions[sent[i][1]][sent[i + 1][1]] += 1

    for tag1 in transitions:
        total = sum(transitions[tag1].values())
        for tag2 in tag_list:
            transitions[tag1][tag2] = (transitions[tag1][tag2] + 1) / (total + len(tag_list))

    return transitions


def compute_observation_matrix(tagged_sents, tag_list):
    observations = defaultdict(lambda: defaultdict(int))
    word_set = set(["OOV/UNK"])
    for sent in tagged_sents:
        for word, tag in sent:
            word_set.add(word)
            observations[tag][word] += 1

    word_list = list(word_set)
    for tag in observations:
        total = sum(observations[tag].values())
        for word in word_list:
            observations[tag][word] = (observations[tag][word] + 1) / (total + len(word_list))

    return observations, word_list

nltk.download('treebank')

# Load the Penn Treebank dataset
tagged_sentences = treebank.tagged_sents()

# Split the data into training and testing sets
train_data, test_data = train_test_split(tagged_sentences, test_size=0.2, random_state=42)

# Define the tag list
tag_list = set(tag for sent in tagged_sentences for _, tag in sent)

# Compute initial distribution, transition matrix, and observation matrix using the training set
initial_distribution = compute_initial_distribution(train_data)
transition_matrix = compute_transition_matrix(train_data, tag_list)
observation_matrix, word_list = compute_observation_matrix(train_data, tag_list)

pi = np.array([initial_distribution[tag] for tag in tag_list])
A = np.array([[transition_matrix[tag1][tag2] for tag2 in tag_list] for tag1 in tag_list])
B = np.array([[observation_matrix[tag][word] for word in word_list] for tag in tag_list])

correct = 0
total = 0

epsilon = 1e-10

for sentence in test_data:
    obs_indices = [word_list.index(word) if word in word_list else word_list.index("OOV/UNK") for word, _ in sentence]
    true_states = [list(tag_list).index(tag) for _, tag in sentence]

    predicted_states, probability = viterbi(obs_indices, pi, A, B)

    total += len(true_states)
    correct += sum(1 for true, pred in zip(true_states, predicted_states) if true == pred)

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")

# Plot and save the initial distribution
plt.bar(list(tag_list), pi + epsilon)
plt.title('Initial Distribution')
plt.xlabel('Tags')
plt.ylabel('Probability')
plt.xticks(rotation=90, fontsize='small')
plt.savefig("plots/initial_distribution.png")
plt.show()

# Plot and save the transition matrix
plt.imshow(A, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar()
plt.title('Transition Matrix')
plt.xlabel('Next State')
plt.ylabel('Current State')
plt.savefig("plots/transition_matrix.png")
plt.show()

# Plot and save the observation matrix
plt.imshow(B, cmap='viridis', origin='lower', aspect='auto')
plt.colorbar()
plt.title('Observation Matrix')
plt.xlabel('Observation')
plt.ylabel('State')
plt.savefig("plots/observation_matrix.png")
plt.show()
