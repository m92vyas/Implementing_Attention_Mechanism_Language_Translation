from src.components import model_attn, custom_loss_function
from src.utils import load_object
import training_parameters as param
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class PredictPipeline:
    def __init__(self, tokenizer_ita_path, tokenizer_eng_path, model_path):
        self.tokenizer_ita_path=tokenizer_ita_path
        self.tokenizer_eng_path=tokenizer_eng_path
        self.model_path=model_path
        
    def predict(self, input_sentence):
      '''takes italian input sentence and outputs translated english sentence'''

      tknizer_ita = load_object(self.tokenizer_ita_path)
      tknizer_eng = load_object(self.tokenizer_eng_path)
      encoder_seq = tknizer_ita.texts_to_sequences([input_sentence]) # tokenizing
      encoder_seq = pad_sequences(encoder_seq, maxlen=20, dtype='int32', padding='post') #padding to len 20

      model  = model_attn.TranslationModel(param.encoder_inputs_length,param.decoder_inputs_length, 27459,13385,param.embedding_dim_enc,\
                              param.embedding_dim_dec,param.enc_units,param.dec_units, param.lstm_dropout, param.recurrent_dropout)

      #model.compile(optimizer=tf.keras.optimizers.Adam(), loss=custom_loss_function.loss_function, metrics=[custom_loss_function.masked_acc])
      #model.fit(train_dataloader, steps_per_epoch=train_steps, epochs=1)
      
      model.load_weights(filepath=self.model_path)

      encoder_output = model.encoder(encoder_seq) # encoding

      decoder_input = tknizer_eng.texts_to_sequences(['<start>']) # feeding <start> token as 1st decoder input
      decoder_state_h = tf.zeros([1024, param.enc_units])
      decoder_state_c = tf.zeros([1024, param.enc_units])
      prediction=[]

      for i in range(20): # iterating over full max_length
        if i>0 and prediction[-1]==['<end>']: # end translation when <end> token predicted
          break
        else:
          input=tf.expand_dims(decoder_input[0][-1],0)  
          if i==0:
            input=tf.expand_dims(input,0)
          decoder_output, state_h,state_c, context_vector= model.layers[1].crossattention(input, encoder_output)
          index = tf.argmax(decoder_output,axis=-1).numpy()
          prediction.append(tknizer_eng.sequences_to_texts([index]))
          decoder_input[0].append(index)
          decoder_state_h = state_h
          decoder_state_c = state_c
        
      predicted_sent= prediction[0][0]
      for word in prediction:
        predicted_sent = predicted_sent + ' ' + word[0]
      return predicted_sent