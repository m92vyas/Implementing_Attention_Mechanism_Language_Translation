import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense


class Encoder(tf.keras.layers.Layer):
    '''class that enocodes input sentence using lstm and returns encoded vector of shape [batch_size,input_length,lstm_units]'''
    def __init__(self, vocab_size, embedding_dim_enc, input_length, enc_units, dropout=0.0, recurrent_dropout=0.0):
      super().__init__()
      self.vocab_size = vocab_size
      self.embedding_dim_enc = embedding_dim_enc
      self.input_length = input_length
      self.enc_units= enc_units
      self.dropout = dropout 
      self.recurrent_dropout = recurrent_dropout 

    def get_config(self):
      config = super().get_config()
      config.update({'vocab_size': self.vocab_size, 'embedding_dim_enc': self.embedding_dim_enc,\
                     'input_length': self.input_length, 'enc_units': self.enc_units,
                     'lstm_dropout': self.lstm_dropout, 'recurrent_dropout':self.recurrent_dropout})
      return config

    def build(self, input_shape):
      self.embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim_enc, input_length=self.input_length,
                          mask_zero=True, name="embedding_layer_encoder") 
      self.lstm = LSTM(self.enc_units, return_state=True, return_sequences=True,dropout=self.dropout,\
                        recurrent_dropout=self.recurrent_dropout ,name="Encoder_LSTM")
      
    def call(self, input_sentances, training=True):
      '''input sentence is embeded and then passed to lstm'''
      input_embedd = self.embedding(input_sentances) # [b, max_len, embed-size]
      self.lstm_output, _, _ = self.lstm(input_embedd) # [b, max_len, lstm-units], 
      return self.lstm_output
    
    def initialize_states(self,batch_size):
      '''Given a batch size it will return intial hidden state and intial cell state'''
      return tf.zeros([batch_size, self.enc_units ]), tf.zeros([batch_size, self.enc_units ])

    def get_states(self):
      return self.lstm_state_h, self.lstm_state_c