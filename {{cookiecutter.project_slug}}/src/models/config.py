import os, sys

RAW_DATA_FILE = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../data/raw/mnist_train.csv'))

TRAINING_FILE = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../data/processed/mnist_train_folds.csv'))

SAME_DIR = os.path.dirname(os.path.realpath(__file__))
