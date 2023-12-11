import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.corpus import treebank
from nltk import word_tokenize, pos_tag
from collections import Counter
import nltk
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# """
# Bidirectional RNN
# """
# Download the Penn Treebank dataset
nltk.download('treebank')

# Load the Penn Treebank dataset
tagged_sentences = treebank.tagged_sents()

train_data, test_data = train_test_split(tagged_sentences, test_size=0.2, random_state=42)
flat_train_data = [(word, tag) for sentence in train_data for word, tag in sentence]
flat_test_data = [(word, tag) for sentence in test_data for word, tag in sentence]

vocab = set(word for word, tag in flat_train_data)
tags = set(tag for word, tag in flat_train_data)

word_index = {word: idx + 1 for idx, word in enumerate(vocab)}
tag_index = {tag: idx for idx, tag in enumerate(tags)}

class PosDataset(Dataset):
    '''
    Define a custom dataset class for part-of-speech tagging.
    '''
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

class BiRNN(nn.Module):
    ''' Define the bidirectional RNN model '''
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

# Function for training the model
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    train_losses = []
    for epoch in range(epochs):
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
        train_losses.append(average_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}")
    return train_losses

# Function for evaluation on the testing data
def evaluate_model(model, test_loader, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=-1)  # Use dim=-1 for the last dimension
            total += targets.numel()
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
# Function for visualization of word embeddings using t-SNE
def visualize_embeddings(model, word_index, n_words=100):
    model.eval()
    words = list(word_index.keys())[:n_words]
    word_vectors = []
    with torch.no_grad():
        for word in words:
            idx = torch.tensor(word_index.get(word, 0)).unsqueeze(0)
            embedded = model.embedding(idx)
            word_vectors.append(embedded.numpy())
    word_vectors = np.vstack(word_vectors)
    tsne = TSNE(n_components=2, random_state=42)
    embedded_words = tsne.fit_transform(word_vectors)

    plt.figure(figsize=(12, 8))
    plt.scatter(embedded_words[:, 0], embedded_words[:, 1], marker='o')
    for i, word in enumerate(words):
        plt.annotate(word, (embedded_words[i, 0], embedded_words[i, 1]))

    plt.title('t-SNE Visualization of Word Embeddings')
    plt.savefig("plots/tsne_word_embeddings.png")
    plt.show()

# Train the model
train_losses = train_model(model, train_loader, criterion, optimizer, epochs=10)

# Plot training loss over epochs
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.savefig("plots/training_loss_plot.png")
plt.show()

# Evaluate the model on the testing set and visualize word embeddings
evaluate_model(model, test_loader, criterion)
visualize_embeddings(model, word_index)

#
# """
# Hidden Markov Model Version moved to new script
# """
