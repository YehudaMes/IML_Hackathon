import os

import numpy as np
import pandas as pd

DATA_PATH = "../../../Agoda - Data/agoda_cancellation_train.csv"
TRAIN_PATH = "./data/train/"
VALIDATION_PATH = "./data/validation/"


def create_data_directories():
    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(VALIDATION_PATH, exist_ok=True)


def validation_indexes(data_len, validation_len):
    np.random.seed(26)
    return np.random.choice(np.arange(data_len), validation_len)


def split_save_data():
    df = pd.read_csv(DATA_PATH)
    validation_inds = validation_indexes(len(df), len(df) // 4)
    validation_set = df.iloc[validation_inds]
    validation_set.to_csv(VALIDATION_PATH + "validation.csv")
    df.drop(validation_inds).to_csv(VALIDATION_PATH + "train.csv")

if __name__ == '__main__':
    create_data_directories()
    split_save_data()
