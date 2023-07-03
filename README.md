# Bahdanau Attention Mechanism Implementation for Language Translation.

## Introduction:
- We have 362861 rows of Italian to English translated sentences as raw data.
- Appropriate preprocessing was done. Input english sentences(related to decoder block) were appended by 'start' and output decoder sentences were appended with 'end' token.
  
|italian                    |english_inp                           |english_out                         |
|---------------------------|--------------------------------------|------------------------------------|
|avevo tutto sotto controllo|<start> i had everything under control|i had everything under control <end>|

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
  

