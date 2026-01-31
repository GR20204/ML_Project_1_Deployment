import os
import sys
from dataclasses import dataclass
import dill
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
import dill as dl
from sklearn.model_selection import train_test_split
import pickle
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:  
            dl.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(xtrain,ytrain,xtest,ytest,models):
    try: 
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(xtrain,ytrain)
            ytrainpred = model.predict(xtrain)
            ytestpred = model.predict(xtest)
            train_model_score = r2_score(ytrain,ytrainpred)
            test_model_score = r2_score(ytest,ytestpred)
            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)