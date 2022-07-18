import config
import logging
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def create_folds(raw_df):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info('Creating folds')
    X = raw_df.drop("label", axis=1)
    y = raw_df.loc[:,'label']

    skf = StratifiedKFold(n_splits=10, random_state=24, shuffle=True)

    fold = 1
    raw_df["kfold"] = ""
    for train_index, test_index in skf.split(X, y):
        raw_df.loc[test_index,:] = raw_df.loc[test_index,:].assign(kfold=fold)
        fold += 1

    filepath = config.TRAINING_FILE
    raw_df.to_csv(filepath, index=False)
    logger.info('Folds done')
