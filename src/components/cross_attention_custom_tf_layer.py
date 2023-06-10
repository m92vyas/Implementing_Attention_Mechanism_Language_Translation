import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from src.components import attention_custom_tf_layer

class DecoderEncoderCrossAttention(tf.keras.Model):
  '''class that performs cross attention on decoder input, pass the attention updated input to
  lstm and then to final dense layer having units equal to output vocab size. the decoder input is passed
  one word at a time over batch'''
  def __init__(self,tar_vocab_size, embedding_dim_dec, input_length, dec_units, lstm_dropout, recurrent_dropout, trainable=True):
    super().__init__()
    self.tar_vocab_size = tar_vocab_size
    self.embedding_dim_dec = embedding_dim_dec
    self.input_length = input_length
    self.dec_units = dec_units
    self.lstm_dropout = lstm_dropout
    self.recurrent_dropout = recurrent_dropout 
    self.trainable = True

  def get_config(self):
      config = super().get_config()
      config.update({
          'tar_vocab_size': self.tar_vocab_size, 'embedding_dim_dec': self.embedding_dim_dec,
          'input_length': self.input_length, 'dec_units': self.dec_units, 'trainable': self.trainable})
      return config

  def build(self, input_shape):
    if self.trainable:
      self.embedding = Embedding(input_dim=self.tar_vocab_size, output_dim=self.embedding_dim_dec, input_length=self.input_length,
                        mask_zero=True, name="embedding_layer_decoder", trainable=True)
    #else:
      #self.embedding = Embedding(input_dim=self.tar_vocab_size, output_dim=self.embedding_dim_dec, input_length=self.input_length,
                        #mask_zero=True, name="embedding_layer_decoder", weights=[embedding_matrix], trainable=False)

    self.lstm = LSTM(self.dec_units, return_sequences=True, return_state=True,dropout=self.lstm_dropout,recurrent_dropout=self.recurrent_dropout, name="Encoder_LSTM")
    self.attention = attention_custom_tf_layer.AttentionMechanism()
    self.dense_layer = Dense(self.tar_vocab_size,activation=None)

  def call(self,input_to_decoder, encoder_output): 
    '''takes decoder single input over batch, encoder outputs, performs cross attention and returns returns logits'''
    # embedding decoder single input
    target_embedd = self.embedding(input_to_decoder) # [b,1,embedding_dim_dec]
    # getting attention updated embedding vector
    context_vector,_ = self.attention(target_embedd,encoder_output) #[b,encoder_lstm_units]
    # concataneting embeded input and attention updated input
    concat_input = tf.concat([target_embedd, tf.expand_dims(context_vector, 1)], -1)  # [b,1,(embedding_dim_dec+encoder_lstm_units)] 
    # passing to lstm
    self.lstm_output, self.lstm_state_h, self.lstm_state_c = self.lstm(concat_input) # [b,1,dec_lstm_units] , [b,1,dec_lstm_units] , [b,1,dec_lstm_units]
    # getting logits
    output = self.dense_layer(self.lstm_output)  # [b,1,tar_vocab_size]
    output = tf.squeeze(output,axis=1) # [b,tar_vocab_size]
    return output, self.lstm_state_h,self.lstm_state_c, context_vector     