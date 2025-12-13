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
                'XGBoost':XGBRegressor(),
                'SVR': SVR(),
                'Adaboost':AdaBoostRegressor()
                 }
            params = {
                 "LinearRegression": {
                 "fit_intercept": [True, False],
                 "positive": [False, True]
                          },

                 "DecisionTree": {
                 "max_depth": [None, 5, 10, 20],
                 "min_samples_split": [2, 5, 10],
                 "min_samples_leaf": [1, 2, 4]
                               },

                 "RandomForest": {
                 "n_estimators": [100, 200],
                 "max_depth": [None, 10, 20],
                 "min_samples_split": [2, 5],
                 "min_samples_leaf": [1, 2]
                                },

                 "Adaboost": {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
                        # "loss": ["linear", "square", "exponential"]
                                 },


                 "SVR": {
                         "C": [0.1, 1, 10],
                           "kernel": ["rbf", "linear"],
                          "gamma": ["scale", "auto"]
                     },

                     "KNN": {
                     "n_neighbors": [3, 5, 7, 9],
                     "weights": ["uniform", "distance"],
                     "p": [1, 2]   # 1 = Manhattan, 2 = Euclidean
                         },

                     "XGBoost": {
                     "n_estimators": [100, 200],
                     "learning_rate": [0.05, 0.1],
                     "max_depth": [3, 5, 7],
                     "subsample": [0.8, 1.0],
                     "colsample_bytree": [0.8, 1.0]
                        }

            }

            model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,
                                               models=models,params=params)
            
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
            
    




    