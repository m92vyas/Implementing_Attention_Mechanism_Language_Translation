from src.components import training
from src.exception import CustomException
from src.logger import logging
import sys

class TrainPipeline:
    def __init__(self, train_data_path, vali_data_path):
        self.train_data_path=train_data_path
        self.vali_data_path=vali_data_path

    def train(self):
        try:
            obj=training.ModelTraining()
            obj.dataset_generator(self.train_data_path,self.vali_data_path)
            obj.training()
        
        except Exception as e:
            raise CustomException(e,sys)