import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "agoda_data/agoda_cancellation_train.csv"
TRAIN_PATH = "data/train.csv"
VALIDATION_PATH = "data/validation.csv"
RANDOM_SEED = 0
TEST_SIZE = 0.2


def split_data():
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(DATA_PATH)

    # Split the data into training and test sets
    train_data, test_data = train_test_split(data, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    # Save the training set to a CSV file
    train_data.to_csv(TRAIN_PATH, index=False)

    # Save the test set to a CSV file
    test_data.to_csv(VALIDATION_PATH, index=False)


if __name__ == "__main__":
    split_data()
