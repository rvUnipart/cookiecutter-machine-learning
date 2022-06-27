import argparse
import warnings

import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import config
import model_dispatcher

def run(fold, model):
    df = pd.read_csv(config.TRAINING_FILE)
    df.drop("Unnamed: 0", axis=1, inplace=True)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    X_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    X_valid = df_valid.drop("label", axis=1).values
    y_valid = df_valid.label.values

    clf = model_dispatcher.models[model]
    clf.fit(X_train, y_train)
    preds = clf.predict(X_valid)

    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f"Fold={fold}, Accuracy={accuracy}")

    joblib.dump(clf, f"{config.MODEL_OUTPUT}/dt_{fold}.bin")

    return accuracy

if __name__ == "__main__":
    with mlflow.start_run():
        warnings.filterwarnings("ignore")
        
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--fold",
            type=int
        )
        parser.add_argument(
            "--model",
            type=str
        )

        args = parser.parse_args()

        accuracy = run(
            fold=args.fold,
            model=args.model
        )

        mlflow.log_param("fold", args.fold)
        mlflow.log_param("model", args.model)
        mlflow.log_metric("accuracy", accuracy)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(args.model, "model", registered_model_name="ModelNameGoesHere")
        else:
            mlflow.sklearn.log_model(args.model, "model")
