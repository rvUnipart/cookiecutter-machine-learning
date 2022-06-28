import argparse
import warnings

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

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, nargs='?', default=1)
    parser.add_argument("--model", type=str, nargs='?', default="rf")
    parser.add_argument("--name", type=str, nargs='?', default="default")
    args = parser.parse_args()
        
    with mlflow.start_run(run_name=args.name) as mlflow_run:
        warnings.filterwarnings("ignore")  

        accuracy = run(
            fold=args.fold,
            model=args.model
        )

        mlflow.log_param("fold", args.fold)
        mlflow.log_param("model", args.model)
        mlflow.log_metric("accuracy", accuracy)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(args.model, "model", registered_model_name="ModelNameGoesHere")
        else:
            mlflow.sklearn.log_model(args.model, "model")
