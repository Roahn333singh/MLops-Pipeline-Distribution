import logging
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# Logging configuration
logger=logging.getLogger("evaluation_model.log")
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'evaluation_model.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s- %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(model_path:str):
    '''Load the trained model from a file'''

    try:
        with open(model_path,"rb") as file:
            model=pickle.load(file)
        logger.debug("Model loaded from %s path :",model_path)
        return model
    except FileNotFoundError as e:
        logger.error("File not Found %s",e)
        raise
    except Exception as e:
        logger.error("Unwanted error occur while loading the Model : %s",e)
        raise

def load_data(data_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(data_path)
        logger.debug("Data added from the path %s ",data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file : %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected Error occur while loadinf csv file : %s",e)
        raise

def eval_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    '''Evaluate the model and return the evaluated metrics'''

    try:
        # Make predictions
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')  
        recall = recall_score(y_test, y_pred, average='weighted')  
        f1 = f1_score(y_test, y_pred, average='weighted') 
        support = len(y_test)  
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support  # The total support value (number of test samples)
        }

        logger.debug("Model Evaluation completed !!")
        return metrics
    except Exception as e:
        logger.error("Error during model Evaluation %s", e)
        raise

def save_metrics(metrics:dict,file_path:str)->None:
    '''Save model eval metrics to a JSON file'''

    try:
        '''Making sure file exist'''
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug("Metrics saved to file %s ",file_path)
    except Exception as e:
        logger.error("Unexpected error occur while saving meetrics %s ",e)
        raise

def main():

    try:
        clf=load_model('./models/model.pkl')
        df=load_data('./data/processed/test_tf_idf.csv')
        X_test=df.iloc[:,:-1].values
        y_test=df.iloc[:,-1].values

        model_eval=eval_model(clf=clf,X_test=X_test,y_test=y_test)

        save_metrics(model_eval,'reports/metrics.json')

    except Exception as e:
        logger.error("Failed to complete model evaluation process : %s ",e)
        print(f"Error {e}")

if __name__=='__main__':
    main()


        


