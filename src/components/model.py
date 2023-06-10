import tensorflow as tf
from src.components import encoder_custom_tf_layer, decoder_custom_tf_layer

class TranslationModel(tf.keras.Model):

  def __init__(self, encoder_inputs_length,decoder_inputs_length, vocab_size_ita,vocab_size_eng,\
                embedding_dim_enc,embedding_dim_dec,enc_units,dec_units,lstm_dropout,recurrent_dropout):
    super().__init__()
    self.encoder_inputs_length = encoder_inputs_length
    self.decoder_inputs_length = decoder_inputs_length
    self.vocab_size_ita = vocab_size_ita
    self.vocab_size_eng = vocab_size_eng
    self.embedding_dim_enc = embedding_dim_enc
    self.embedding_dim_dec = embedding_dim_dec
    self.enc_units = enc_units
    self.dec_units = dec_units
    self.lstm_dropout = lstm_dropout
    self.recurrent_dropout = recurrent_dropout

  def get_config(self):
    config = super().get_config()
    config.update({'encoder_inputs_length': self.encoder_inputs_length, 'decoder_inputs_length': self.decoder_inputs_length,
        'vocab_size_ita': self.vocab_size_ita, 'vocab_size_eng': self.vocab_size_eng, 'embedding_dim_enc': self.embedding_dim_enc,
        'embedding_dim_dec': self.embedding_dim_dec , 'enc_units': self.enc_units, 'dec_units': self.dec_units, 'att_units': self.att_units})
    return config

  def build(self,input_shapes):
      self.encoder = encoder_custom_tf_layer.Encoder(self.vocab_size_ita+1, self.embedding_dim_enc,
                        self.encoder_inputs_length, self.enc_units,self.lstm_dropout,
                        self.recurrent_dropout)
      self.decoder = decoder_custom_tf_layer.DecoderBlock(self.vocab_size_eng+1, self.embedding_dim_dec,
                        self.decoder_inputs_length, self.lstm_dropout,self.recurrent_dropout)
      
  def call(self, data):
      input,output = data[0], data[1]
      #passing input to encoder
      encoder_output = self.encoder(input) #[b,max_len,encoder_lstm_units]
      #passing output to decoder
      decoder_output = self.decoder(output, encoder_output) #[b,max_len,tar_vocab_size] #logits
      return decoder_output