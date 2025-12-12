from dataclasses import dataclass
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from src.utils import evaluate_model,save_object
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    confusion_matrix,
)

from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    Trained_Model_File_Path = os.path.join("artifacts","model.pkl")
     

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()
    
       
    def initiate_model_training(self,train_path,test_path):
        try:
            logging.info("splitting training and test data")

            x_train,y_train,x_test,y_test = (
                train_path[:,:-1],
                train_path[:,-1],
                test_path[:,:-1],
                test_path[:,-1]
            )
            models ={
                'LinearRegression':LinearRegression(),
                'KNN' : KNeighborsRegressor(),
                'RandomForest':RandomForestRegressor(),
                'DecisionTree':DecisionTreeRegressor(),
                'XGboost':XGBRegressor(),
                'SVR': SVR(),
                'Adaboost':AdaBoostRegressor()
                 }
            

            model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                               models=models)
            
            best_model_score = max(model_report.values())

            best_model_name = max(model_report, key=model_report.get)
            best_model= models[best_model_name]

            print("Best Model:", best_model_name)
            print("Best Score:", best_model_score)

            if best_model_score < 0.6:
                raise CustomException("no best model found")
            logging.info("best model found") 

            save_object(
                file_path=self.model_trainer_config.Trained_Model_File_Path,
                obj=best_model
    
            )
            predicted=best_model.predict(x_test)
            r2= r2_score(y_test,predicted)

            return r2
            



        except Exception as e:
            raise CustomException(e,sys)   
            
    




    