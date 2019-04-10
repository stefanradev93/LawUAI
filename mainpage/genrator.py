import tensorflow as tf
import pickle
import re
import os
from os import listdir


NN_PATH = "C:/Users/Stefan Radev/Desktop/Projects/Hackathon/production/winning_project/UI/hackaton_UI/nn_models"


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.CuDNNGRU(self.enc_units,
                                     return_sequences=True,
                                     return_state=True,
                                     recurrent_initializer='glorot_uniform') )

    def call(self, x):
        x = self.embedding(x)
        output = self.gru(x)
        out = output[0]
        hidden = tf.concat(output[1:], axis=-1)
        return out, hidden

    def initialize_hidden(self, batch_size):
        return tf.zeros((batch_size, self.enc_units))


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


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.gru = tf.keras.layers.CuDNNGRU(self.dec_units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output, probs=False):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        if probs:
            return tf.nn.softmax(x, axis=-1), state, attention_weights
        return x, state, attention_weights


class TitleGenerator:

    def __init__(self, max_len_questions=450, max_len_titles=10, vocab_size_q=30000,
                 vocab_size_t=30000, embedding_dim=128, enc_units=128, dec_units=256):
        self._max_len_q = max_len_questions
        self._max_len_t = max_len_titles
        self._load(vocab_size_q, vocab_size_t, embedding_dim, enc_units, dec_units)

    def _load(self, vocab_size_q, vocab_size_t, embedding_dim, enc_units, dec_units):
        """Load tokenizers and model."""

        # ----- Load tokenizers and create word indices ----- #
        self.questions_tokenizer = pickle.load(open(os.path.join(NN_PATH, "generator", "questions_tokenizer.pkl"), "rb"))
        self.titles_tokenizer = pickle.load(open(os.path.join(NN_PATH, "generator", "titles_tokenizer.pkl"), "rb"))
        self.word_index_q = self.questions_tokenizer.word_index
        self.word_index_t = self.titles_tokenizer.word_index
        self.index_word_q = {v: k for k, v in self.word_index_q.items()}
        self.index_word_t = {v: k for k, v in self.word_index_t.items()}

        # ----- Load model and weights ----- #
        self.encoder = Encoder(vocab_size_q, embedding_dim, enc_units=enc_units)
        self.decoder = Decoder(vocab_size_t, embedding_dim, dec_units=dec_units)
        self.encoder.load_weights(os.path.join(NN_PATH, "generator", "encoder_" + "bidirectional_20epochs"))
        self.decoder.load_weights(os.path.join(NN_PATH, "generator", "decoder_" + "bidirectional_20epochs"))

    def _preprocess(self, text):
        """A very basic function to preprocess the texts. Needs to be imporved."""

        # Reg exp cleaning
        text = text.lower().strip()
        text = re.sub(r"([?.!,¿])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        text = re.sub(r"[^a-zA-ZäüßöÄÖÜ?]+", " ", text)
        text = text.rstrip().strip()
        text = re.sub(r"/(^|\s+)(\S(\s+|$))+/", "", text)

        # Tokenize, pad and convert to tensor
        inputs = self.questions_tokenizer.texts_to_sequences([text])
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                               maxlen=self._max_len_q,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs, dtype=tf.int32)
        return inputs

    def generate_title(self, raw_text):
        """Generates the title given the supplied raw text."""

        # Preprocess question
        processed_text = self._preprocess(raw_text)
        generated_title = ''
        enc_out, enc_hidden = self.encoder(processed_text)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.word_index_t['<start>']], 0)

        # Use the language model
        for t in range(self._max_len_t):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)
            predicted_idx = tf.argmax(predictions[0]).numpy()
            dec_input = tf.expand_dims([predicted_idx], 0)
            generated_title += self.index_word_t[predicted_idx] + ' '
            if self.index_word_t[predicted_idx] == '<end>':
                return generated_title.replace(' <end> ', '').capitalize()
        return generated_title.replace(' <end> ','').capitalize()

