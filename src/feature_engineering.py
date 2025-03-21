import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import os
import yaml

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# Logging configuration
logger=logging.getLogger("feature_engineering.log")
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'feature_engineering.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s- %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


'''Adding yaml config code'''

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

'''--------------------------------------------------------'''

def load_data(data_path:str)->pd.DataFrame:
    '''Loding data from csv file'''
    try:
        df=pd.read_csv(data_path)
        logger.debug("Data loaded successfully !!")
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occur while loading the data: %s",e)
        raise

def apply_tf_idf(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int)->tuple:

    try:
        train_data["tweet"] = train_data["tweet"].fillna("")
        test_data["tweet"] = test_data["tweet"].fillna("")

        X_train=train_data["tweet"].values
        y_train=train_data["label"].values

        X_test=test_data["tweet"].values
        y_test=test_data["label"].values


        tfidf = TfidfVectorizer(max_features=max_features)

        X_tfidf_train= tfidf.fit_transform(X_train)

        X_tfidf_test = tfidf.fit_transform(X_test)

        '''opening sparse matrix to array form '''
        train_df=pd.DataFrame(X_tfidf_train.toarray())
        train_df["label"]=y_train

        test_df=pd.DataFrame(X_tfidf_test.toarray())
        test_df["label"]=y_test

        logger.debug("tf_idf is applied and data transformed ")

        return train_df,test_df
    
    except Exception as e:
        logger.error("Error during Tf-Idf Transformation %s",e)
        raise

def save_data(df:pd.DataFrame,file_path:str)->None:
    '''Save the dataframe to the csv file'''

    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug('Data saved to %s',file_path)
    except Exception as e:
        logger.error("Unexpected error occur while saving the data : %s",e)
        raise

def main():
    try:
        params=load_params(params_path="params.yaml")
        max_features=params['feature_engineering']['max_feature']
        # max_features=5000
        train_data=load_data("./data/interim/train_processed.csv")
        test_data=load_data("./data/interim/test_processed.csv")

        train_df,test_df=apply_tf_idf(train_data=train_data,test_data=test_data,max_features=max_features)

        save_data(train_df,os.path.join("./data","processed","train_tf_idf.csv"))
        save_data(test_df,os.path.join("./data","processed","test_tf_idf.csv"))

    except Exception as e:
        logger.error("Failed to complete the feature engineering process : %s",e)
        print(f"Error {e}")

if __name__== "__main__":
    main()
        










