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
from sklearn.preprocessing import LabelEncoder


log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# Logging configuration
logger=logging.getLogger("data_preprocessing")
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s- %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def process_target_col(df:pd.DataFrame,target_col="label"):

    '''Encoding the target column'''
    try:
        logger.debug("Started preprocessing dataframe ....")
        label_encoder=LabelEncoder()
        df[target_col]=label_encoder.fit_transform(df[target_col])
        logger.debug("target column %s encoded...",target_col)
        return df

    except KeyError as e:
        logger.error("column not found: %s",e)
        raise

    except Exception as e:
        logger.error("error during text normalization: %s",e)
        raise

def main(target_col="label"):
    '''main function to read the raw data , process it and save the data'''

    try:
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')
        logger.debug("data loaded successfully !!!")

        '''transform the data'''
        train_processed_data=process_target_col(df=train_data,target_col="label")
        test_processed_data=process_target_col(df=test_data,target_col="label")

        ''' storing the data inside data/processed '''
        data_path=os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
        test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)


        logger.debug("Processed data saved to %s ",data_path)

    except FileNotFoundError as e:
        logger.error("File not found: %s",e)
    except pd.errors.EmptyDataError as e:
        logger.error("No data %s",e)
    except Exception as e:
        logger.error("Failed to complete the Data transformation process: %s",e)
        print(f"Error {e}")


''' calling main function to  concatinate whole pipeline flow'''

if __name__=="__main__":
    main()






    
















    




    
    
    
