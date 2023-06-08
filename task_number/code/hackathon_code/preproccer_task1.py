import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

COLUMNS_DATA_PATH = './columns_data/task1_columns.txt'

MEANS = {'hotel_star_rating': 3,
         'no_of_adults': 2,
         'no_of_extra_bed': 0,
         'no_of_room': 1,
         'no_of_children': 0,
         'days_before_cancelled': 10,
         'original_selling_amount': 219,
         "booking_to_checkin": 35}

COLS_TO_DROP = ["h_booking_id", "hotel_id", "cancellation_datetime", "checkin_date", "checkout_date",
                "hotel_brand_code", "hotel_chain_code", "hotel_live_date", "booking_datetime",
                "request_nonesmoke", "request_latecheckin", "request_highfloor", "request_largebed",
                "request_twinbeds", "request_airport", "request_earlycheckin",
                # "customer_nationality",
                "h_customer_id",
                # "no_of_children", "no_of_adults",
                # "no_of_extra_bed", "no_of_room",
                "origin_country_code", "hotel_city_code", "is_user_logged_in", "original_payment_currency",
                "original_payment_method", "guest_is_not_the_customer", "language"]

COLUMNS_TO_DUMMIES = [
    "accommadation_type_name", "charge_option", "guest_nationality_country_name",
    "hotel_country_code", "hotel_area_code", "is_first_booking", "cancellation_policy_code",
    "original_payment_type", "customer_nationality"]


def booking_to_checkin_feature(df):
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    df["booking_to_checkin"] = pd.to_datetime(df["checkin_date"]) - pd.to_datetime(df["booking_datetime"])
    df["booking_to_checkin"] = df["booking_to_checkin"].fillna(pd.Timedelta(0)).dt.days.astype(int)


def preprocess_train_task1(path):
    df = pd.read_csv(path)

    df["cancellation_indicator"] = df["cancellation_datetime"].notnull().astype(int)  # Task 1 labeling
    booking_to_checkin_feature(df)
    df = df.drop(COLS_TO_DROP, axis=1)
    df = df[df['hotel_star_rating'].isin(np.arange(0, 5.5, 0.5))]
    df = df[df['booking_to_checkin'].isin(np.arange(0, 350, 1))]
    df = pd.get_dummies(df, columns=COLUMNS_TO_DUMMIES)

    # Save the column names as a text file
    with open(COLUMNS_DATA_PATH, 'w') as file:
        file.write('\n'.join(df.columns))

    X, y = df.drop('cancellation_indicator', axis=1), df['cancellation_indicator']
    return X, y


def preprocess_data_to_validation_task1(path):
    df = pd.read_csv(path)

    # Read columns feature of Trained model
    with open(COLUMNS_DATA_PATH, 'r') as file:
        desired_columns = file.read().splitlines()

    df["cancellation_indicator"] = df["cancellation_datetime"].notnull().astype(int)  # Task 1 labeling
    booking_to_checkin_feature(df)

    df.loc[(~df['hotel_star_rating'].isin(np.arange(0, 5, 0.5))), 'hotel_star_rating'] = MEANS['hotel_star_rating']
    df.loc[(df['original_selling_amount'] < 10) | (df['original_selling_amount'] > 5000),
           'original_selling_amount'] = MEANS['original_selling_amount']
    df.loc[(df['booking_to_checkin'] < 0) | (df['booking_to_checkin'] > 350),
           'booking_to_checkin'] = MEANS['booking_to_checkin']

    df = df.drop(COLS_TO_DROP, axis=1)
    df = pd.get_dummies(df, columns=COLUMNS_TO_DUMMIES)
    df = df.reindex(columns=desired_columns, fill_value=0)

    X, y = df.drop('cancellation_indicator', axis=1), df['cancellation_indicator']
    return X, y


def preprocess_predict_task1(path):
    df = pd.read_csv(path)

    # Read columns feature of Trained model
    with open(COLUMNS_DATA_PATH, 'r') as file:
        desired_columns = file.read().splitlines()

    booking_to_checkin_feature(df)
    df.loc[(~df['hotel_star_rating'].isin(np.arange(0, 5, 0.5))), 'hotel_star_rating'] = MEANS['hotel_star_rating']
    df.loc[(df['original_selling_amount'] < 10) | (df['original_selling_amount'] > 5000),
           'original_selling_amount'] = MEANS['original_selling_amount']
    df.loc[(df['booking_to_checkin'] < 0) | (df['booking_to_checkin'] > 350),
           'booking_to_checkin'] = MEANS['booking_to_checkin']

    cols_to_drop = COLS_TO_DROP.copy()
    cols_to_drop.remove("cancellation_datetime")

    df = df.drop(cols_to_drop, axis=1)
    df = pd.get_dummies(df, columns=COLUMNS_TO_DUMMIES)

    desired_columns.remove('cancellation_indicator')
    df = df.reindex(columns=desired_columns, fill_value=0)
    return df


def load_train_data_task1():
    DATA_PATH = "data/train.csv"
    return preprocess_train_task1(DATA_PATH)


def load_validation_data_task1():
    path = "data/validation.csv"
    return preprocess_data_to_validation_task1(path)


def load_train_agoda_data_task1():
    DATA_PATH = "agoda_data/agoda_cancellation_train.csv"
    return preprocess_train_task1(DATA_PATH)


def load_test_agoda_data_task1():
    path = "agoda_data/Agoda_Test_1.csv"
    return preprocess_predict_task1(path)


load_train_agoda_data_task1()
