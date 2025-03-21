import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()
from nltk.corpus import stopwords
stopword = set(stopwords.words('english')) 
import nltk
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split


# this is to make sure our logs directory exists
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# Logging configuration
logger=logging.getLogger("data_ingestion")
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s- %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# will be creating data loding function

def data_load(data_path:str)->pd.DataFrame:
    ''' Loading data from csv file '''
    try:
        df=pd.read_csv(data_path)
        logger.debug("Data loaded from %s",data_path)
        return df
    
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the  CSV file: %s",e)
        raise
    
    except Exception as e:
        logger.error("Unexpected error occur while loading the data: %s",e)
        raise

# we will be creating data preprocessing function

def pre_process(df:pd.DataFrame)->pd.DataFrame:
    '''Pre_process the data'''
    try:
        df["label"]=df["class"].map({0:"hat_speech_detected",1:"offensive_language_detected",2:"neutral"})
        df=df[["tweet","label"]]

        ''' dropping the null value and appling cleaning process '''
        
        def clean_data(text):
            text = str(text).lower()
            text = re.sub(r'\[.*\]', '', text)
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'<.*?>', '', text)
            text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub(r'\n', '', text)
            text = re.sub(r'\w*\d\w*', '', text)
            text = [word for word in text.split(' ') if word not in stopword]
            text = " ".join(text)
            text = [stemmer.stem(word) for word in text.split(' ')]
            text = " ".join(text)
            return text
        
        
        df["tweet"] = df["tweet"].dropna().apply(clean_data)

        logger.debug("Data pre_process completed")
        return df
    
    except Exception as e:
        logger.log("Unexpected error occur while processing %s",e)
        raise

# we are saving the processed data using save function

def save_data(train_df:pd.DataFrame,test_df:pd.DataFrame,data_path:str)->None:
    ''' save the train and test dataset'''
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_df.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_df.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logger.debug("train and test data saved to %s",raw_data_path)
    except Exception as e:
        logger.error("unexpected error occur while aving the data: %s",e)
        raise

def main():
    try:
        test_size=0.2
        data_path="https://raw.githubusercontent.com/Roahn333singh/Datasets/refs/heads/main/labeled_data.csv"
        df=data_load(data_path=data_path)
        final_df=pre_process(df=df)
        train_df,test_df=train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_df=train_df,test_df=test_df,data_path='./data')

    except Exception as e:
        logger.error("Failed to complete the data ingestion process: %s",e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()
    




