import pandas as pd
import numpy as np

from .preprocess_util import common_column_edit

COLUMNS_DATA_PATH = '../hackathon_code/columns_data/task1_columns.txt'

MEANS = {'hotel_star_rating': 3,
         'no_of_adults': 2,
         'no_of_extra_bed': 0,
         'no_of_room': 1,
         'no_of_children': 0,
         'original_selling_amount': 219,
         "booking_to_checkin": 35}

COLS_TO_DROP = ["h_booking_id",
                "hotel_id",
                "cancellation_datetime",
                "checkin_date",
                "checkout_date",
                "hotel_brand_code",
                "hotel_chain_code",
                "hotel_live_date",
                "booking_datetime",
                "request_nonesmoke",
                "request_latecheckin",
                "request_highfloor",
                "request_largebed",
                "request_twinbeds",
                "request_airport",
                "request_earlycheckin",
                "h_customer_id",
                "origin_country_code",
                "origin_country_code",
                "hotel_city_code",
                "is_user_logged_in",
                "original_payment_currency",
                "original_payment_method",
                "cancellation_policy_code",
                "language"]

COLUMNS_TO_DUMMIES = [
    "accommadation_type_name",
    "charge_option",
    "guest_nationality_country_name",
    "hotel_country_code",
    "hotel_area_code",
    "original_payment_type",
    "customer_nationality"]


def preprocess_train_task1(path, with_edit):
    df = pd.read_csv(path)
    df["cancellation_indicator"] = df["cancellation_datetime"].notnull().astype(int)  # Task 1 labeling

    if with_edit:
        df = common_column_edit(df, COLS_TO_DROP, COLUMNS_TO_DUMMIES)

    df = df[df['hotel_star_rating'].isin(np.arange(0, 5.5, 0.5))]
    X, y = df.drop('cancellation_indicator', axis=1), df['cancellation_indicator']

    # Save the column names as a text file
    with open(COLUMNS_DATA_PATH, 'w') as file:
        file.write('\n'.join(X.columns))

    return X, y


def preprocess_data_to_validation_task1(path):
    df = pd.read_csv(path)

    # Read columns feature of Trained model
    with open(COLUMNS_DATA_PATH, 'r') as file:
        desired_columns = file.read().splitlines()

    df["cancellation_indicator"] = df["cancellation_datetime"].notnull().astype(int)  # Task 1 labeling

    df = common_column_edit(df, COLS_TO_DROP, COLUMNS_TO_DUMMIES)

    df.loc[(~df['hotel_star_rating'].isin(np.arange(0, 5, 0.5))), 'hotel_star_rating'] = MEANS['hotel_star_rating']
    fill_means(df)

    df = df.reindex(columns=desired_columns, fill_value=0)

    X, y = df.drop('cancellation_indicator', axis=1), df['cancellation_indicator']
    return X, y


def fill_means(df):
    df.loc[(df['original_selling_amount'] < 10) | (df['original_selling_amount'] > 5000),
           'original_selling_amount'] = MEANS['original_selling_amount']
    df.loc[(df['booking_to_checkin'] < 0) | (df['booking_to_checkin'] > 350),
           'booking_to_checkin'] = MEANS['booking_to_checkin']


def preprocess_predict_task1(path):
    df = pd.read_csv(path)

    # Read columns feature of Trained model
    with open(COLUMNS_DATA_PATH, 'r') as file:
        desired_columns = file.read().splitlines()

    df = common_column_edit(df, COLS_TO_DROP, COLUMNS_TO_DUMMIES)

    df.loc[(~df['hotel_star_rating'].isin(np.arange(0, 5, 0.5))), 'hotel_star_rating'] = MEANS['hotel_star_rating']
    fill_means(df)

    df = df.reindex(columns=desired_columns, fill_value=0)
    return df


def load_train_data_task1():
    DATA_PATH = "../hackathon_code/data/train.csv"
    return preprocess_train_task1(DATA_PATH, True)


def load_validation_data_task1():
    path = "../hackathon_code/data/validation.csv"
    return preprocess_data_to_validation_task1(path)


def load_train_agoda_data_task1(with_edit):
    DATA_PATH = "../hackathon_code/agoda_data/agoda_cancellation_train.csv"
    return preprocess_train_task1(DATA_PATH, with_edit)


def load_test_agoda_data_task1():
    path = "./hackathon_code/agoda_data/Agoda_Test_1.csv"
    return preprocess_predict_task1(path)
