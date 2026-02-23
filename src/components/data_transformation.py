import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifact", "preprocessor.pkl")
    label_encoder_path = os.path.join("artifact", "label_encoder.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.label_encoder=LabelEncoder()

    def get_data_transformer_object(self):
        '''
        Creates preprocessing pipelines for numerical and categorical features (X only)
        '''
        try:
            num_cols = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO',
                        'Proximity_to_Industrial_Areas', 'Population_Density']
            cat_cols = []  # no categorical features in X, only in target

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, num_cols),
                ],
                #remainder="drop"  # drop target column
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Air Quality"
            numerical_columns = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO','Proximity_to_Industrial_Areas', 'Population_Density']

            # Split into input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            # Encode target labels
            target_feature_train_encoded = self.label_encoder.fit_transform(target_feature_train_df)
            target_feature_test_encoded = self.label_encoder.transform(target_feature_test_df)

            logging.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")

            # Fit & transform features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Concatenate with target
            train_arr = np.c_[input_feature_train_arr, target_feature_train_encoded]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_encoded]

            logging.info(f"Saved preprocessing object.")
            # Save preprocessor
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Save label encoder
            save_object(
                file_path=self.data_transformation_config.label_encoder_path,
                obj=self.label_encoder
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
