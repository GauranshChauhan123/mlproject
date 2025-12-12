import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            logging.info("Data Transformation: Identifying numerical and categorical columns")

            numerical_columns = ["reading score", "writing score"]
            categorical_columns = [
                "gender", "race/ethnicity", "parental level of education", 
                "lunch", "test preparation course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("impute", SimpleImputer(strategy="median"))
                ] 
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Pipelines created successfully")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            logging.info("Reading train and test data")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Splitting features and target")

            target_column = "math score"

            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            logging.info("Applying preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()

            X_train_array = preprocessor_obj.fit_transform(X_train)
            X_test_array = preprocessor_obj.transform(X_test)

            # Combine X and y back for model training
            train_arr = np.c_[X_train_array, np.array(y_train)]
            test_arr = np.c_[X_test_array, np.array(y_test)]

            logging.info("Saving preprocessor object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path                          
,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path                          
            )

        except Exception as e:
            raise CustomException(e, sys)
