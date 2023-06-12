import tensorflow as tf

encoder_inputs_length = 20
decoder_inputs_length = 20
embedding_dim_enc = 100
embedding_dim_dec = 100
enc_units = 128
dec_units = 128
lstm_dropout = 0.2
recurrent_dropout = 0.2
batch_size = 1024
optimizer = tf.keras.optimizers.Adam()