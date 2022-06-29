import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import config
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    print("Loading data")
    mnist_train_df = pd.read_csv(config.RAW_DATA_FILE)

    X = mnist_train_df.drop("label", axis=1)
    y = mnist_train_df.loc[:,'label']

    print("Creating fold object")
    skf = StratifiedKFold(n_splits=10, random_state=24, shuffle=True)

    print("Assigning folds")
    fold = 1
    mnist_train_df["kfold"] = ""
    for train_index, test_index in skf.split(X, y):
        mnist_train_df.loc[test_index,:] = mnist_train_df.loc[test_index,:].assign(kfold=fold)
        fold += 1

    print("Writing to csv")
    filepath = config.TRAINING_FILE
    mnist_train_df.to_csv(filepath, index=False) 
