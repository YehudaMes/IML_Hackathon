import pandas as pd

TRAIN_PATH = "./data/train.csv"
# VALIDATION_PATH = "./data/train.csv"

data = pd.read_csv(TRAIN_PATH)
print(data.head())
print(data.columns)

if __name__ == "__main__":
    print("Hello ML!")
