import pandas as pd


def naive_preprocess():
    TRAIN_PATH = "./data/train.csv"
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(TRAIN_PATH)
    data.fillna(0, inplace=True)
    # Remove textual columns
    y = data["cancellation_datetime"].notnull().astype(int)

    data = data.select_dtypes(exclude=['object'])
    # Identify columns with boolean values
    boolean_columns = data.select_dtypes(include=bool).columns

    # Convert boolean columns to integers
    data[boolean_columns] = data[boolean_columns].astype(int)

    # Specify the column names to remove
    columns_to_remove = ['h_booking_id', 'hotel_id', 'h_customer_id']

    # Remove the specified columns
    return data.drop(columns=columns_to_remove), y
