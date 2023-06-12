from src.components import dataset_loader, model_attn, custom_loss_function, callbacks
import training_parameters as param
import pandas as pd
import tensorflow as tf

class ModelTraining():
  
  def __init__(self):
    pass

  def dataset_generator(self, train_set_path, vali_set_path, batch_size=1024):

    self.train = pd.read_csv(train_set_path)
    self.val = pd.read_csv(vali_set_path)
    self.batch_size = batch_size
    
    tknizer_ita, tknizer_eng = dataset_loader.tokenizer_ita_eng(train_set_path)
    self.vocab_size_eng=len(tknizer_eng.word_index.keys())
    self.vocab_size_ita=len(tknizer_ita.word_index.keys())

    train_dataset = dataset_loader.Dataset(self.train, tknizer_ita, tknizer_eng, 20)
    val_dataset  = dataset_loader.Dataset(self.val, tknizer_ita, tknizer_eng, 20)

    self.train_dataloader = dataset_loader.Dataloder(train_dataset, batch_size=batch_size)
    self.val_dataloader = dataset_loader.Dataloder(val_dataset, batch_size=batch_size)

  def training(self, translation_model=model_attn.TranslationModel, epochs=10):

    train_steps=self.train.shape[0]//self.batch_size
    valid_steps=self.val.shape[0]//self.batch_size
    
    mdl  = model_attn.TranslationModel(param.encoder_inputs_length, param.decoder_inputs_length, self.vocab_size_ita,\
                             self.vocab_size_eng, param.embedding_dim_enc, param.embedding_dim_dec,\
                             param.enc_units, param.dec_units, param.lstm_dropout, param.recurrent_dropout)

    mdl.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=custom_loss_function.loss_function,
                metrics=[custom_loss_function.masked_acc])
    
    callbk=callbacks.PrepareCallback()

    mdl.fit(self.train_dataloader, steps_per_epoch=train_steps,
            epochs=epochs, validation_data=self.val_dataloader,
            validation_steps=valid_steps, callbacks=callbk.get_callbacks)