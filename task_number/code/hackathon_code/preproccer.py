import pandas as pd

TRAIN_PATH = "/data/train.cvs"
# VALIDATION_PATH = "/data/train.cvs"

data = pd.read_csv(TRAIN_PATH)

if __name__ == "__main__":
    print("Hello ML!")

