import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):

        try:
            # Paths
            model_path = os.path.join("artifact", "model.pkl")
            preprocessor_path = os.path.join("artifact", "preprocessor.pkl")
            encoder_path = os.path.join("artifact", "label_encoder.pkl")

            # Load saved objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            label_encoder = load_object(file_path=encoder_path)

            # Transform input
            data_scaled = preprocessor.transform(features)

            # Predict (encoded number)
            pred_encoded = model.predict(data_scaled)

            # Decode number -> string
            pred_label = label_encoder.inverse_transform(
                pred_encoded.astype(int)
            )

            return pred_label

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Temperature: float,
        Humidity: float,
        PM2_5: float,
        PM10: float,
        NO2: float,
        SO2: float,
        CO: float,
        Proximity_to_Industrial_Areas: float,
        Population_Density: float,
    ):

        self.Temperature = Temperature
        self.Humidity = Humidity
        self.PM2_5 = PM2_5
        self.PM10 = PM10
        self.NO2 = NO2
        self.SO2 = SO2
        self.CO = CO
        self.Proximity_to_Industrial_Areas = Proximity_to_Industrial_Areas
        self.Population_Density = Population_Density

    def get_data_as_data_frame(self):

        try:
            custom_data_input_dict = {
                "Temperature": [self.Temperature],
                "Humidity": [self.Humidity],

                "PM2.5": [self.PM2_5],

                "PM10": [self.PM10],
                "NO2": [self.NO2],
                "SO2": [self.SO2],
                "CO": [self.CO],
                "Proximity_to_Industrial_Areas": [
                    self.Proximity_to_Industrial_Areas
                ],
                "Population_Density": [self.Population_Density],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)