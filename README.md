# Bahdanau Attention Mechanism Implementation for Language Translation.

## Introduction:
- We have 362861 rows of Italian to English translated sentences as raw data.
- Appropriate preprocessing was done. Input english sentences(related to decoder block) were appended by 'start' and output decoder sentences were appended with 'end' token.
- 307077 sentences were used for training, 54191 sentences for validation and 1088 sentences as test dataset.
- Both italian and english sentences were tokenize and maximum sequence length of 20 tokens was selected. Finally we had 13335 english tokens and 27402 italian tokens.
- Appropriate Dataset Loader code was written to return encoder sequence, decoder input/output sequence.

## Model:
  ![image](https://github.com/m92vyas/Implementing_Attention_Mechanism_for_language_translation/assets/126826477/54a35bbc-c3e3-41a9-b192-72a2aac61981)
  ref: https://guillaumegenthial.github.io/sequence-to-sequence.html

## Encoder Layer:
  - Italian tokens were embedded to vectors as per given dimensions using embedding layer. Output dimensions: [batch, max_len, embed-size]
  - Individual LSTM output are used to get cross attention score in further layer. LSTM output dimensions: [batch, max_len, lstm-units]
 
## Attention Mechanism Layer:
  - Decoder input is transformed to match encoder output dimensions and attention weights are calculated based on similarity using dot 
    products and weighted sum of encoder hidden state vector is returned as context vector to be used by decoder.
  - Context vector dimensions: [batch,encoder_lstm_units]
  - Attention weights dimensions: [batch,max_len,1]

## Decoder Encoder Cross Attention Layer:
  - It performs cross attention between embedded decoder input(can be glove vectors) and embedded encoder input using previous attention 
    mechanism layer, concatanate the attention updated/weighted decoder input with embedded decoder input and pass it to to lstm layer. 
    Then to dense layer having units equal to output vocab size. The decoder input is passeed one word at a time over batch(matrix form) 
    i.e. cross attention is performed one embedded token at a time over whole batch.
  - Final output dimensions: [batch,tar_vocab_size]
 
## Decoder Layer:
  - It performs cross attention using Decoder Encoder Cross Attention Layer and gives logits values for full decoder input length.
  - Final Logits output shape: [batch,max_len,tar_vocab_size]

## Final Translation Model:
  - Using dataset generator appropriate data is passed to encoder and decoder block and final logits values are returned over whole batch.

## Custom Loss Function and Metric:
  - Custom loss function and metric will not consider the loss for padded zero.
 
## Training:
  - Following hyperparameters are choosen for training the model:
    - encoder_inputs_length = 20
    - decoder_inputs_length = 20
    - vocab_size_ita = vocab_size_ita
    - vocab_size_eng = vocab_size_eng
    - embedding_dim_enc = 100
    - embedding_dim_dec = 100
    - enc_units = 128
    - dec_units = 128
    - lstm_dropout = 0.2
    - recurrent_dropout = 0.2
    - optimizer = tf.keras.optimizers.Adam()
  - After 70 epochs we get validation accuracy of 0.86% (model not trained further due to resources constraints)
  - Some translated sentence from test datasets:
    
    - Italian:  vedo cosavete fatto lì
    
      English True:  i see what you did there <end>
    
      Model Translation:  i i see what you have done there <end>
    
    
    - Italian:  tom non è un fisico
      
      English True:  tom is not a physician <end>
      
      Model Translation:  tom tom is not a physician <end>
    
    
    - Italian:  cè un costo di consegna
    
      English True:  is there a delivery charge <end>
      
      Model Translation:  there there is a charge of the delivery <end>
    
    
    - Italian:  è un tizio strano
    
      English True:  he is a strange guy <end>
      
      Model Translation:  it it is a strange guy <end>
 
    
    
    - Italian:  tutti qua sanno che non mangiamo la carne di maiale
    
      English True:  everyone here knows that we do not eat pork <end>
      
      Model Translation:  everyone everyone here knows we do not eat pork <end>

  - Average test data bleu score:  0.4451662890214658
  - Average test data cumulative 4-gram bleu score:  1.479362713798278e-231
