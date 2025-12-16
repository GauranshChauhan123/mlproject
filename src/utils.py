import pandas as pd
import sys
import dill 
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import os

def save_object(file_path,obj):
    try:
        
        dir_path= os.path.dirname(file_path) 
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,y_train,x_test,y_test,models,params):
    try:

        report={}
        for name, model in models.items():
            grid = GridSearchCV(estimator=model,param_grid=params[name],n_jobs=-1,cv=3)
            grid.fit(x_train,y_train)
            model.set_params(**grid.best_params_)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            r2 = r2_score(y_test, y_pred)
            report[name]=r2

        return report
                       
    except Exception as e:
               raise CustomException(e,sys)
    

def load_object(file_path):
     try:
          with open(file_path,'rb') as f:
              return dill.load(f)
          
     except Exception as e:
          raise CustomException(e,sys)     



