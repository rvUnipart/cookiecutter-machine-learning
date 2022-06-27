import os, sys

##TRAINING_FILE = "../../data/processed/mnist_train_folds.csv"
TRAINING_FILE = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../data/processed/mnist_train_folds.csv'))

##MODEL_OUTPUT = "../../models" # original relative approach
MODEL_OUTPUT = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../../models'))

SAME_DIR = os.path.dirname(os.path.realpath(__file__))
