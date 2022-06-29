import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
import config
import logging
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info('Loading data')
    mnist_train_df = pd.read_csv(config.RAW_DATA_FILE)

    X = mnist_train_df.drop("label", axis=1)
    y = mnist_train_df.loc[:,'label']

    logger.info('Creating fold object')
    skf = StratifiedKFold(n_splits=10, random_state=24, shuffle=True)

    logger.info('Assigning folds')
    fold = 1
    mnist_train_df["kfold"] = ""
    for train_index, test_index in skf.split(X, y):
        mnist_train_df.loc[test_index,:] = mnist_train_df.loc[test_index,:].assign(kfold=fold)
        fold += 1

    logger.info('Writing to csv')
    filepath = config.TRAINING_FILE
    mnist_train_df.to_csv(filepath, index=False) 
