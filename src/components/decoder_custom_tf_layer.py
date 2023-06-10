import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from src.components import cross_attention_custom_tf_layer

class DecoderBlock(tf.keras.Model):
  '''class which gives logits values for full decoder input length '''
  def __init__(self,out_vocab_size, embedding_dim_dec, input_length, dec_units,lstm_dropout,recurrent_dropout):
    super().__init__()
    self.out_vocab_size = out_vocab_size
    self.embedding_dim_dec = embedding_dim_dec
    self.input_length = input_length
    self.dec_units = dec_units
    self.lstm_dropout = lstm_dropout
    self.recurrent_dropout = recurrent_dropout

  def get_config(self):
    config = super().get_config()
    config.update({
        'out_vocab_size': self.out_vocab_size, 'embedding_dim_dec': self.embedding_dim_dec,
        'input_length': self.input_length, 'dec_units': self.dec_units})
    return config

  def build(self,input_shapes):
    self.crossattention = cross_attention_custom_tf_layer.DecoderEncoderCrossAttention(
                            self.out_vocab_size, self.embedding_dim_dec, self.input_length,\
                            self.dec_units ,self.lstm_dropout,self.recurrent_dropout,False)

  def call(self, input_to_decoder,encoder_output):
    # creating a empty array of length equal to input length to fill logits value
    all_outputs = tf.TensorArray(tf.float32,size=self.input_length)
    # iterating over individual input word
    for timestep in range(self.input_length):
      # getting logits value for current input word
      output, decoder_hidden_state,decoder_cell_state,_ = self.crossattention(input_to_decoder[:,timestep:timestep+1], encoder_output)
      all_outputs = all_outputs.write(timestep, output) #[max_len,b,tar_vocab_size]
    all_outputs = tf.transpose(all_outputs.stack(), [1,0,2]) #[b,max_len,tar_vocab_size]
    return all_outputs 