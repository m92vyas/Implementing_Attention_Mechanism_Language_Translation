import re
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import urllib.parse
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


def decontractions(phrase): #https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python/47091490#47091490
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess_eng(text):
    text = decontractions(text)
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    text = text.lower()
    return text

def preprocess_ita(text):
    text = decontractions(text)
    text = re.sub("""[$)\?"'.°!;\'€%:,(/]""" , "" , text)
    text = re.sub('\u200b', ' ', text)
    text = re.sub('\xa0', ' ', text)
    text = re.sub('-', ' ', text)
    text = text.lower()
    return text


@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    val_data_path: str=os.path.join('artifacts',"val.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self,input_data_path):
        self.ingestion_config=DataIngestionConfig()
        self.input_data_path=input_data_path

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            #!wget http://www.manythings.org/anki/ita-eng.zip
            #!unzip ita-eng.zip
            with open(self.input_data_path, 'r', encoding="utf8") as f:
              eng=[]
              ita=[]
              for i in f.readlines():
                a = i.split("\t")
                eng.append(a[0])
                ita.append(a[1])
            data = pd.DataFrame()
            data['english'] = eng
            data['italian'] = ita
            data.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("raw data saved")

            #preprocessing:
            logging.info("preprocessing data")
            data['english'] = data['english'].apply(preprocess_eng)
            data['italian'] = data['italian'].apply(preprocess_ita)

            #making data model ready:
            data['italian_len'] = data['italian'].str.split().apply(len)
            data = data[data['italian_len'] < 20]

            data['english_len'] = data['english'].str.split().apply(len)
            data = data[data['english_len'] < 20]

            data['english_inp'] = '<start> ' + data['english'].astype(str) # decoder input starts with <start> token
            data['english_out'] = data['english'].astype(str) + ' <end>'   # decoder output ends with <end> token

            data = data.drop(['english','italian_len','english_len'], axis=1)

            logging.info("preprocessing done")

            #train,validation,test split:
            logging.info("train, validation, test split initiated")
            trainval, test = train_test_split(data, test_size=0.003) # for test set getting appox. 1000 sentences
            train, val = train_test_split(trainval, test_size=0.15)

            train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            val.to_csv(self.ingestion_config.val_data_path,index=False,header=True)
            test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("train, validation, test data saved")

        except Exception as e:
            raise CustomException(e,sys)
        

#if __name__=="__main__":
    #obj=DataIngestion('notebook\data\ita.txt')
    #obj.initiate_data_ingestion()