import tensorflow as tf
import pickle
import os
import numpy as np


NN_PATH = "C:/Users/Stefan Radev/Desktop/Projects/Hackathon/production/winning_project/UI/hackaton_UI/nn_models"


class AttentionClassifierModel(tf.keras.Model):

    def __init__(self, embeddings_matrix, n_classes, max_len=500, hidden_dim=128):
        super(AttentionClassifierModel, self).__init__()

        # Constuct from existing embeddings matrix
        if type(embeddings_matrix) is np.ndarray:
            self.embedding = tf.keras.layers.Embedding(
                embeddings_matrix.shape[0],
                embeddings_matrix.shape[1],
                weights=[embeddings_matrix],
                input_length=max_len,
                trainable=False)

        # Construct a new embedings matrix
        # Embeddings matrix is just a tuple with vocab size and embedding_dim
        else:
            self.embedding = tf.keras.layers.Embedding(embeddings_matrix[0], embeddings_matrix[1])
        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.CuDNNLSTM(hidden_dim,
                                      return_state=True,
                                      return_sequences=True))
        self.attention = BahdanauAttention(hidden_dim)
        self.dense = tf.keras.layers.Dense(64, activation='elu')
        self.cls = tf.keras.layers.Dense(n_classes)

    def call(self, inputs, probs=False):

        X = self.embedding(inputs)
        X = self.lstm(X)
        output = X[0]
        hiddens = tf.concat(X[1:], axis=-1)
        context, attention_weights = self.attention(hiddens, output)
        X = self.dense(context)
        logits = self.cls(X)
        if probs:
            return tf.nn.softmax(logits, axis=-1), attention_weights
        return logits, attention_weights

    def compute_loss(self, y_true, y_pred_logits):
        """Computes the loss between predicted and true labels."""

        return tf.losses.softmax_cross_entropy(y_true, y_pred_logits)



class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class AttentionAoLClassifier:

    def __init__(self, embeddings_matrix=(30000, 300), max_len=500, n_classes=10, hidden_dim=128):
        self._max_len = max_len
        self._load(hidden_dim, n_classes, embeddings_matrix)

    def _load(self, hidden_dim, n_classes, embeddings_matrix):
        """Load tokenizers and model."""

        # ----- Load tokenizers and create word indices ----- #
        self.tokenizer = pickle.load(open(os.path.join(NN_PATH, "classifier", "tokenizer.pkl"), "rb"))
        self.word_index = self.tokenizer.word_index
        self.index_word= {v:k for k,v in self.word_index.items()}

        # ----- Load embeddings ----- #
        # _, embeddings_matrix = get_embeddings("../embeddings/", self.word_index)

        # ----- Load model and weights ----- #
        self.model = AttentionClassifierModel(embeddings_matrix, n_classes, hidden_dim=hidden_dim)
        self.model.load_weights(os.path.join(NN_PATH, "classifier", "classifier_attention_" + "bidirectional_15epochs"))

    def _preprocess(self, text):
        """A very basic function to preprocess the texts. Needs to be imporved."""

        # Tokenize, pad and convert to tensor
        inputs = self.tokenizer.texts_to_sequences([text])
        decoded=self.decode(inputs[0])
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                               maxlen=self._max_len,
                                                               padding='post')
        inputs_tensor = tf.convert_to_tensor(inputs, dtype=tf.int32)
        return inputs_tensor,decoded

    def decode(self,sequence):
        return [self.index_word[s] for s in sequence if s!=0]

    def classify_text(self, raw_text):
        """Generates the title given the supplied raw text."""

        # Preprocess question
        processed_text,decoded= self._preprocess(raw_text)
        # Make predictions and return
        probs, attn_weights = self.model(processed_text, probs=True)
        probs, attn_weights = probs.numpy(), attn_weights.numpy()
        return probs, attn_weights,decoded