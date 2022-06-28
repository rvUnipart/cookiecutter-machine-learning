import os, sys

##TRAINING_FILE = "../../data/processed/mnist_train_folds.csv" # original, relative path method
TRAINING_FILE = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../data/processed/mnist_train_folds.csv'))

SAME_DIR = os.path.dirname(os.path.realpath(__file__))
