import pandas as pd

TRAIN_PATH = "./data/train.csv"
VALIDATION_PATH = "./data/validation.csv"

train_data = pd.read_csv(TRAIN_PATH)
validation_data = pd.read_csv(VALIDATION_PATH)
pd.set_option('display.max_columns', None)

print(len(train_data))
print(len(validation_data))

# print(data.head())

if __name__ == "__main__":
    print("Hello ML!")
