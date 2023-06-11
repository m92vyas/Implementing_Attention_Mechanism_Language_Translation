import tensorflow as tf
import datetime
import os
from dataclasses import dataclass


@dataclass
class CallbackConfig:
    tensorboard_log_dir: str=os.path.join('artifacts',"logs",'model','fits', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_save_path = 'artifacts/model.{epoch:02d}-{masked_acc:.4f}.h5'


class PrepareCallback:
    def __init__(self):
        self.callbackconfig = CallbackConfig()

    
    @property
    def _tensorboard_callbacks(self):
        log_dir = self.callbackconfig.tensorboard_log_dir
        return tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1,write_graph=True)
    

    @property
    def _model_save_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
                     filepath=self.callbackconfig.model_save_path, 
                     save_freq='epoch', verbose=0, monitor='masked_acc', 
                     save_weights_only=True, save_best_only=True
                 )


    def get_callbacks(self):
        return [
            self._tensorboard_callbacks,
            self._model_save_callbacks
        ]