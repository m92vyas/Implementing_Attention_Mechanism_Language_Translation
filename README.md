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
    Context vector dimensions: [batch,encoder_lstm_units]
    attention weights dimensions: [batch,max_len,1]
  

