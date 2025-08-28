import os
import numpy as np
import random
import time
import pickle
from collections import defaultdict, Counter
import re
import copy
from math import log, sqrt

class ImprovedRNNEncoder:
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        scale = 0.1
        self.embedding = np.random.normal(0, scale, (vocab_size, embed_dim)).astype(np.float32)
        self.W_ih = np.random.normal(0, scale, (hidden_dim, embed_dim)).astype(np.float32)
        self.W_hh = np.random.normal(0, scale, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)

    def _orthogonal_init(self, shape):
        """Create orthogonal matrix for RNN weights"""
        flat_shape = (shape[0], np.prod(shape[1:]))
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v
        return q.reshape(shape)

    def forward(self, sequence, training=True):
        if len(sequence) == 0:
            return np.zeros(self.hidden_dim, dtype=np.float32), [], [], []

        seq_len = len(sequence)
        h = np.zeros(self.hidden_dim, dtype=np.float32)
        hidden_states = np.zeros((seq_len, self.hidden_dim), dtype=np.float32)
        embeddings = np.zeros((seq_len, self.embed_dim), dtype=np.float32)

        valid_indices = [min(max(idx, 0), self.vocab_size-1) for idx in sequence]
        embeddings = self.embedding[valid_indices]

        if training and self.dropout > 0:
            dropout_mask = np.random.binomial(1, 1-self.dropout, embeddings.shape) / (1-self.dropout)
            embeddings *= dropout_mask

        for t in range(seq_len):
            x = embeddings[t]
            h = np.tanh(self.W_ih @ x + self.W_hh @ h + self.b_h)
            h = np.clip(h, -5, 5)
            hidden_states[t] = h

        return h, hidden_states, embeddings, sequence

    def backward(self, grad_output, hidden_states, embeddings, sequence, learning_rate=0.001):
        if len(sequence) == 0:
            return

        seq_len = len(sequence)
        grad_h = np.clip(grad_output, -1, 1)  # Less aggressive clipping (was -2, 2)

        effective_lr = learning_rate * 0.5  # Only 2x smaller (was 10x smaller)

        for t in reversed(range(min(seq_len, 20))):  # Process more steps (was 10)
            tanh_grad = np.maximum(1 - hidden_states[t]**2, 0.01)
            grad_h_raw = grad_h * tanh_grad
            grad_h_raw = np.clip(grad_h_raw, -0.2, 0.2)  # Less aggressive (was -0.5, 0.5)

            h_prev = hidden_states[t-1] if t > 0 else np.zeros(self.hidden_dim)

            self.W_ih -= effective_lr * np.outer(grad_h_raw, embeddings[t])
            self.W_hh -= effective_lr * np.outer(grad_h_raw, h_prev)
            self.b_h -= effective_lr * grad_h_raw

            valid_idx = min(max(sequence[t], 0), self.vocab_size-1)
            grad_x = self.W_ih.T @ grad_h_raw
            grad_x = np.clip(grad_x, -0.05, 0.05)  # Less aggressive
            self.embedding[valid_idx] -= effective_lr * grad_x

            if t > 0:
                grad_h = np.clip(self.W_hh.T @ grad_h_raw, -0.5, 0.5)

        self.embedding = np.clip(self.embedding, -5, 5)  # Was -2, 2
        self.W_ih = np.clip(self.W_ih, -2, 2)  # Was -1, 1
        self.W_hh = np.clip(self.W_hh, -2, 2)  # Was -1, 1
        self.b_h = np.clip(self.b_h, -2, 2)  # Was -1, 1


class ImprovedSentenceEncoder:
    def __init__(self, word_encoder):
        self.word_encoder = word_encoder
        hidden_dim = word_encoder.hidden_dim

        self.W_ih_sent = np.random.uniform(-np.sqrt(6.0/hidden_dim), np.sqrt(6.0/hidden_dim),
                                         (hidden_dim, hidden_dim)).astype(np.float32)
        self.W_hh_sent = word_encoder._orthogonal_init((hidden_dim, hidden_dim)).astype(np.float32) * 0.5
        self.b_h_sent = np.zeros(hidden_dim, dtype=np.float32)

    def forward(self, sentences, training=True):
        if not sentences:
            return [], []

        sentence_representations = []
        sentence_data = []

        for sentence in sentences:
            sent_rep, hidden_states, embeddings, sequence = self.word_encoder.forward(sentence, training)
            sentence_representations.append(sent_rep)
            sentence_data.append((hidden_states, embeddings, sequence))

        h_doc = np.zeros(self.word_encoder.hidden_dim, dtype=np.float32)
        contextual_reps = []
        doc_states = []

        for sent_rep in sentence_representations:
            h_doc = np.tanh(self.W_ih_sent @ sent_rep + self.W_hh_sent @ h_doc + self.b_h_sent)
            h_doc = np.clip(h_doc, -5, 5)
            contextual_reps.append(h_doc.copy())
            doc_states.append(h_doc.copy())

        return contextual_reps, (sentence_data, doc_states, sentence_representations)

    def backward(self, grad_outputs, forward_data, learning_rate=0.001):
        sentence_data, doc_states, sentence_representations = forward_data

        if len(grad_outputs) == 0:
            return

        grad_W_ih = np.zeros_like(self.W_ih_sent)
        grad_W_hh = np.zeros_like(self.W_hh_sent)
        grad_b_h = np.zeros_like(self.b_h_sent)

        grad_h_doc = np.zeros(self.word_encoder.hidden_dim)

        for t in reversed(range(len(grad_outputs))):
            grad_total = grad_outputs[t] + grad_h_doc
            tanh_grad = 1 - doc_states[t]**2
            grad_h_raw = grad_total * tanh_grad
            grad_h_raw = np.clip(grad_h_raw, -1, 1)

            h_prev = doc_states[t-1] if t > 0 else np.zeros(self.word_encoder.hidden_dim)
            sent_rep = sentence_representations[t]

            grad_W_ih += np.outer(grad_h_raw, sent_rep)
            grad_W_hh += np.outer(grad_h_raw, h_prev)
            grad_b_h += grad_h_raw

            grad_sent_rep = self.W_ih_sent.T @ grad_h_raw
            hidden_states, embeddings, sequence = sentence_data[t]
            self.word_encoder.backward(grad_sent_rep, hidden_states, embeddings, sequence, learning_rate)

            if t > 0:
                grad_h_doc = self.W_hh_sent.T @ grad_h_raw

        max_norm = 1.0
        grad_W_ih = np.clip(grad_W_ih, -max_norm, max_norm)
        grad_W_hh = np.clip(grad_W_hh, -max_norm, max_norm)
        grad_b_h = np.clip(grad_b_h, -max_norm, max_norm)

        self.W_ih_sent -= learning_rate * grad_W_ih
        self.W_hh_sent -= learning_rate * grad_W_hh
        self.b_h_sent -= learning_rate * grad_b_h
class ImprovedBinaryClassifier:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        
        # Better initialization for classification layer
        self.W_class = np.random.uniform(-np.sqrt(6.0/input_dim), np.sqrt(6.0/input_dim),
                                       (1, input_dim)).astype(np.float32)
        self.b_class = np.zeros(1, dtype=np.float32)
    
    def forward(self, representations):
        if not representations:
            return np.array([])
        
        # Vectorized computation
        reps_matrix = np.stack(representations)  # Shape: (num_sentences, hidden_dim)
        logits = reps_matrix @ self.W_class.T + self.b_class  # Broadcasting
        logits = np.clip(logits.flatten(), -10, 10)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        return probabilities
    
    def backward(self, grad_outputs, representations, learning_rate=0.001):
        if len(grad_outputs) == 0 or len(representations) == 0:
            return []
        
        # Vectorized backward pass
        reps_matrix = np.stack(representations)  # Shape: (num_sentences, hidden_dim)
        grad_outputs = np.array(grad_outputs)    # Shape: (num_sentences,)
        
        # Gradients for parameters
        # grad_W should be (1, hidden_dim), computed as sum of outer products
        grad_W = np.outer(grad_outputs, np.ones(reps_matrix.shape[1])) * reps_matrix
        grad_W = grad_W.sum(axis=0, keepdims=True)  # Sum over batch dimension
        grad_b = grad_outputs.sum()
        
        # Gradients for representations
        # grad_representations should be (num_sentences, hidden_dim)
        grad_representations = np.outer(grad_outputs, self.W_class.flatten())
        grad_representations = grad_representations.tolist()
        
        # Update parameters
        max_norm = 1.0
        grad_W = np.clip(grad_W, -max_norm, max_norm)
        grad_b = np.clip(grad_b, -max_norm, max_norm)
        
        self.W_class -= learning_rate * grad_W
        self.b_class -= learning_rate * grad_b
        
        return grad_representations
class ImprovedExtractiveRNNSummarizer:
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):  # Reduced dimensions for speed
        self.word_encoder = ImprovedRNNEncoder(vocab_size, embed_dim, hidden_dim)
        self.sentence_encoder = ImprovedSentenceEncoder(self.word_encoder)
        self.classifier = ImprovedBinaryClassifier(hidden_dim)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def forward(self, sentences, training=True):
        sentence_reps, forward_data = self.sentence_encoder.forward(sentences, training)

        if not sentence_reps:
            return np.array([]), None

        probabilities = self.classifier.forward(sentence_reps)
        return probabilities, (sentence_reps, forward_data)

    def backward(self, loss_gradients, forward_data, learning_rate=0.001):
        sentence_reps, sentence_forward_data = forward_data
        grad_sentence_reps = self.classifier.backward(loss_gradients, sentence_reps, learning_rate)
        self.sentence_encoder.backward(grad_sentence_reps, sentence_forward_data, learning_rate)
class TextPreprocessor:
    def __init__(self, vocab_size=5000):  # Reduced vocab size
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_counts = Counter()

    def build_vocabulary(self, texts):
        print("Building vocabulary...")
        for text in texts:
            words = self.tokenize(text)
            self.word_counts.update(words)

        most_common = self.word_counts.most_common(self.vocab_size - 4)

        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '<START>', 3: '<END>'}

        for idx, (word, _) in enumerate(most_common, start=4):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        print(f"Built vocabulary with {len(self.word_to_idx)} words")
        return self

    def tokenize(self, text):
        text = text.lower()
        return re.findall(r'\b\w+\b', text)

    def text_to_indices(self, text):
        words = self.tokenize(text)
        return [self.word_to_idx.get(word, 1) for word in words]

    def indices_to_text(self, indices):
        words = [self.idx_to_word.get(idx, '<UNK>') for idx in indices if idx != 0]
        return ' '.join(words)

def split_into_sentences(text, max_length=30):  # Reduced max length
    sentences = re.split(r'[.!?]+', text)
    processed = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 0:
            words = sent.split()
            if len(words) > max_length:
                for i in range(0, len(words), max_length):
                    chunk = ' '.join(words[i:i + max_length])
                    if chunk.strip():
                        processed.append(chunk.strip())
            elif len(words) >= 3:  # Only keep sentences with at least 3 words
                processed.append(sent)
    return processed[:10]  # Limit to max 10 sentences per document



