import logging
import pandas as pd 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# Logging configuration
logger=logging.getLogger("training_model.log")
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'training_model.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s- %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_path:str)->pd.DataFrame:
    try:
        '''Loading data from csv file
        '''
        df=pd.read_csv(data_path)
        logger.debug("Data loaded from path %s with shape %s",data_path,df.shape)
        return df
    
    except pd.errors.ParserError as e:
        logger.error("Failed to parse csv file : %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error occur while loading the data %s",e)
        raise



def train_model(X_train:np.ndarray,y_train:np.ndarray,params:dict)->RandomForestClassifier:
    """
    Training Random forest model
    :X_train->Training feature
    :y_training->Training Label
    :params->dictionary of hyperparams
    :returns Trained Random forest classifier
    """

    try:
        if X_train.shape[0]!=y_train.shape[0]:
            raise ValueError("The number of sample in x_train and y_train must be the same.")
        logger.debug("initilizing Randomforest modelwith parameter %s",params)
        clf=RandomForestClassifier(n_estimators=params["n_estimators"],random_state=params["random_state"])

        logger.debug("Model training started with  %d samples",X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.debug("Model training completed !!")

        return clf
    except ValueError as e:
        logger.error("Value error during Model training: %s",e)
        raise
    except Exception as e:
        logger.error("Error During Model Training : %s",e)
        raise

def save_model(model,file_path:str)->None:
    '''
    SAVING MODEL to a file
    :model-> trained model object
    :file_path-> file path where model is being saved 
    '''
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,"wb") as file:
            pickle.dump(model,file)

        logger.debug("Model saved to %s ",file_path)
    except FileNotFoundError as e:
        logger.error("File not found: %s ",e)
        raise

    except Exception as e:
        logger.error("Error occured while saving the model: %s",e)
        raise

def main():
    try:
        params={"n_estimators":100,"random_state":42}
        df=load_data("./data/processed/test_tf_idf.csv")
        X_train=df.iloc[:,:-1].values
        y_train=df.iloc[:,-1].values
        clf=train_model(X_train=X_train,y_train=y_train,params=params)
        model_save_path='models/model.pkl'
        save_model(clf,model_save_path)

    except Exception as e:
        logger.error("Failes to complete model building process : %s",e)
        print(f"Error {e}")

if __name__=='__main__':
    main()






