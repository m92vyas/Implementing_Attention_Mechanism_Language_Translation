import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense

class AttentionMechanism(tf.keras.layers.Layer):
  '''Class the calculates attention weights and corresponding weighted vector using simple dot product operation.'''
  def __init__(self, initializer = tf.keras.initializers.GlorotUniform() ):
    super().__init__()
    self.initializer = initializer

  def get_config(self):
    config = super().get_config()
    config.update({'initializer': self.initializer })
    return config

  def call(self,decoder_hidden_state,encoder_output):
    '''decoder input is transformed to match encoder output dimension and attention weights are calculated
    based on similarity using dot products and weighted sum of encoder hidden state vector is returned as context vector
    to be used by decoder'''
    initializer = self.initializer
    # initializing decoder transformation matrix
    values = initializer(shape=(decoder_hidden_state.shape[2],encoder_output.shape[2])) # [dec_embed_dim, encoder_lstm_units]
    # tranforming decoder input
    similarity1 = tf.matmul(decoder_hidden_state,values) # [b,1,dec_embed_dim] X [dec_embed_dim, encoder_lstm_units] = [b,1,encoder_lstm_units]
    # finding similarity score
    similarity = tf.matmul(similarity1,encoder_output, transpose_b=True) # [b,1,encoder_lstm_units] X [b,encoder_lstm_units,max_len] = [b,1,max_len]
    # normalizing scores using softmax
    attn_weights = tf.nn.softmax( similarity,axis=-1 ) # [b,1,max_len]
    # calculating weighted sum
    context_vector = tf.matmul(attn_weights,encoder_output) # [b,1,max_len] X -[b,max_len,encoder_lstm_units] = [b,1,encoder_lstm_units]
    context_vector = tf.squeeze(context_vector,axis=1) # [b,encoder_lstm_units]
    attn_weights = tf.transpose(attn_weights, perm=[0, 2, 1]) # [b,max_len,1]
    return context_vector, attn_weights