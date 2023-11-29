import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.corpus import treebank
from nltk import word_tokenize, pos_tag
from collections import Counter
import nltk

"""
Bidirectional RNN
"""

# Download the Penn Treebank dataset
nltk.download('treebank')

# Load the Penn Treebank dataset
tagged_sentences = treebank.tagged_sents()

# Split the data into training and testing sets
train_data, test_data = train_test_split(tagged_sentences, test_size=0.2, random_state=42)

# Flatten the list of tagged sentences to obtain a list of (word, tag) pairs
flat_train_data = [(word, tag) for sentence in train_data for word, tag in sentence]
flat_test_data = [(word, tag) for sentence in test_data for word, tag in sentence]

# Create vocabulary and POS tag sets
vocab = set(word for word, _ in flat_train_data)
tags = set(tag for _, tag in flat_train_data)

# Create word and tag indices
word_index = {word: idx + 1 for idx, word in enumerate(vocab)}
tag_index = {tag: idx for idx, tag in enumerate(tags)}

# Define a custom dataset class
class PosDataset(Dataset):
    def __init__(self, data, word_index, tag_index):
        self.data = data
        self.word_index = word_index
        self.tag_index = tag_index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        word, tag = self.data[idx]
        word_idx = self.word_index.get(word, 0)  # 0 represents the index for unknown words
        tag_idx = self.tag_index[tag]
        return word_idx, tag_idx

# Create datasets and data loaders
train_dataset = PosDataset(flat_train_data, word_index, tag_index)
test_dataset = PosDataset(flat_test_data, word_index, tag_index)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the bidirectional RNN model
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply by 2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output)
        return output

# Instantiate the model, loss function, and optimizer
vocab_size = len(word_index) + 1  # Add 1 for unknown words
embedding_dim = 50
hidden_size = 50
output_size = len(tag_index)
model = BiRNN(vocab_size, embedding_dim, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, output_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")

#Evaluation on the testing data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Fix the index here
        total += targets.numel()  # Fix the calculation of total
        correct += (predicted == targets.view(-1)).sum().item()

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")






"""
Hidden Markov Model Version
"""

from typing import Sequence, Tuple, TypeVar
import numpy as np
from collections import defaultdict

Q = TypeVar("Q")
V = TypeVar("V")


def viterbi(
    obs: Sequence[int],
    pi: np.ndarray[Tuple[V], np.dtype[np.float_]],
    A: np.ndarray[Tuple[Q, Q], np.dtype[np.float_]],
    B: np.ndarray[Tuple[Q, V], np.dtype[np.float_]],
) -> tuple[list[int], float]:
    """Infer most likely state sequence using the Viterbi algorithm.

    Args:
        obs: An iterable of ints representing observations.
        pi: A 1D numpy array of floats representing initial state probabilities.
        A: A 2D numpy array of floats representing state transition probabilities.
        B: A 2D numpy array of floats representing emission probabilities.

    Returns:
        A tuple of:
        * A 1D numpy array of ints representing the most likely state sequence.
        * A float representing the probability of the most likely state sequence.
    """
    N = len(obs)
    Q, V = B.shape  # num_states, num_observations

    # d_{ti} = max prob of being in state i at step t
    #   AKA viterbi
    # \psi_{ti} = most likely state preceeding state i at step t
    #   AKA backpointer

    # initialization
    log_d = [np.log(pi) + np.log(B[:, obs[0]])]
    log_psi = [np.zeros((Q,))]

    # recursion
    for z in obs[1:]:
        log_da = np.expand_dims(log_d[-1], axis=1) + np.log(A)
        log_d.append(np.max(log_da, axis=0) + np.log(B[:, z]))
        log_psi.append(np.argmax(log_da, axis=0))

    # termination
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
            transitions[sent[i][1]][sent[i+1][1]] += 1

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




# Define the tag list
tag_list = set(tag for sent in tagged_sentences for _, tag in sent)

# Compute initial distribution, transition matrix, and observation matrix using the training set
initial_distribution = compute_initial_distribution(train_data)
transition_matrix = compute_transition_matrix(train_data, tag_list)
observation_matrix, word_list = compute_observation_matrix(train_data, tag_list)

# Convert dictionaries to numpy arrays
pi = np.array([initial_distribution[tag] for tag in tag_list])
A = np.array([[transition_matrix[tag1][tag2] for tag2 in tag_list] for tag1 in tag_list])
B = np.array([[observation_matrix[tag][word] for word in word_list] for tag in tag_list])

# Example usage of Viterbi algorithm on the testing set
correct = 0
total = 0

for sentence in test_data:
    obs_indices = [word_list.index(word) if word in word_list else word_list.index("OOV/UNK") for word, _ in sentence]
    true_states = [list(tag_list).index(tag) for _, tag in sentence]

    predicted_states, probability = viterbi(obs_indices, pi, A, B)

    total += len(true_states)
    correct += sum(1 for true, pred in zip(true_states, predicted_states) if true == pred)

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f}")