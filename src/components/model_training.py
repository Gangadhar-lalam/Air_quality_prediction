import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=500),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "KNN": KNeighborsClassifier(),
                "XGBoost": XGBClassifier(eval_metric="mlogloss"),
                "CatBoost": CatBoostClassifier(verbose=False),
                "AdaBoost": AdaBoostClassifier(),
            }

            # params = {
            #     "Logistic Regression": {
            #         "C": [0.1, 1, 10],
            #         "solver": ["liblinear", "lbfgs"],
            #     },
            #     "Decision Tree": {
            #         "criterion": ["gini", "entropy"],
            #         "max_depth": [3, 5, 10, None],
            #     },
            #     "Random Forest": {
            #         "n_estimators": [50, 100, 200],
            #         "max_depth": [3, 5, 10, None],
            #     },
            #     "Gradient Boosting": {
            #         "n_estimators": [50, 100, 200],
            #         "learning_rate": [0.01, 0.05, 0.1],
            #         "subsample": [0.8, 1.0],
            #     },
            #     "KNN": {
            #         "n_neighbors": [3, 5, 7, 9],
            #         "weights": ["uniform", "distance"],
            #     },
            #     "XGBoost": {
            #         "n_estimators": [50, 100, 200],
            #         "learning_rate": [0.01, 0.05, 0.1],
            #         "max_depth": [3, 5, 7],
            #     },
            #     "CatBoost": {
            #         "depth": [6, 8, 10],
            #         "learning_rate": [0.01, 0.05, 0.1],
            #         "iterations": [50, 100],
            #     },
            #     "AdaBoost": {
            #         "n_estimators": [50, 100, 200],
            #         "learning_rate": [0.01, 0.05, 0.1],
            #     },
            # }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                # param=params,
            )

            ## Best model score
            best_model_score = max(sorted(model_report.values()))

            ## Best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best model found: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            acc = accuracy_score(y_test, predicted)
            return acc

        except Exception as e:
            raise CustomException(e, sys)
