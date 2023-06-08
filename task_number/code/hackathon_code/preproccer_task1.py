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
                        'days_before_cancelled': 10}

COLS_TO_DROP = ["h_booking_id", "hotel_id", "cancellation_datetime", "checkin_date", "checkout_date",
                "hotel_brand_code", "hotel_chain_code", "hotel_live_date", "booking_datetime",
                "request_nonesmoke", "request_latecheckin", "request_highfloor", "request_largebed",
                "request_twinbeds", "request_airport", "request_earlycheckin", "customer_nationality",
                "h_customer_id", "no_of_children", "no_of_adults", "no_of_extra_bed", "no_of_room",
                "origin_country_code", "hotel_city_code", "is_user_logged_in", "original_payment_currency",
                "original_payment_method", "guest_is_not_the_customer", "language"]

COLUMNS_TO_DUMMIES = [
    "accommadation_type_name", "charge_option", "guest_nationality_country_name",
    "hotel_country_code", "hotel_area_code", "is_first_booking", "cancellation_policy_code",
    "original_payment_type"]


def produce_days_before_cancelling_feature(df):
    df["cancellation_datetime"] = pd.to_datetime(df["cancellation_datetime"])
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    df["days_before_cancelled"] = df["checkin_date"] - df["cancellation_datetime"]
    df["days_before_cancelled"] = df["days_before_cancelled"].fillna(pd.Timedelta(0)).dt.days.astype(int)


def load_train_data():
    DATA_PATH = "data/train.csv"
    return preprocess_train_task1(DATA_PATH)


def load_agoda_data():
    DATA_PATH = "Agoda - Data/agoda_cancellation_train.csv"
    return preprocess_train_task1(DATA_PATH)


# Classification
def preprocess_train_task1(path):
    df = pd.read_csv(path)

    df["cancellation_indicator"] = df["cancellation_datetime"].notnull().astype(int)  # Task 1 labeling

    produce_days_before_cancelling_feature(df)  # todo Takes care on negative days
    df = df.drop(COLS_TO_DROP, axis=1)
    df = df[df['hotel_star_rating'].isin(np.arange(0, 5.5, 0.5))]
    df = pd.get_dummies(df, columns=COLUMNS_TO_DUMMIES)

    # Save the column names as a text file
    with open(COLUMNS_DATA_PATH, 'w') as file:
        file.write('\n'.join(df.columns))

    return df


def preprocess_test_task1(path):
    df = pd.read_csv(path)

    # Read columns feature of Trained model
    with open(COLUMNS_DATA_PATH, 'r') as file:
        desired_columns = file.read().splitlines()

    df["cancellation_indicator"] = df["cancellation_datetime"].notnull().astype(int)  # Task 1 labeling
    produce_days_before_cancelling_feature(df)

    To_tipull = ['hotel_star_rating', 'original_selling_amount', 'cancellation_indicator', 'days_before_cancelled']
    df.loc[df['hotel_star_rating'] < 0, 'hotel_star_rating'] = replacement_value
    df.loc[(~df['hotel_star_rating'].isin(np.arange(0, 5, 0.5))), 'hotel_star_rating'] = MEANS['hotel_star_rating']

    df.loc[(~df['original_selling_amount'].isin(np.arange(0, 5, 0.5))), 'hotel_star_rating'] = MEANS['hotel_star_rating']

    df = df.drop(COLS_TO_DROP, axis=1)
    # df = df[df['hotel_star_rating'].isin(np.arange(0, 5.5, 0.5))]
    df = pd.get_dummies(df, columns=COLUMNS_TO_DUMMIES)
    df = df.reindex(columns=desired_columns, fill_value=0)
    return df


def preprocess_predict_task1(path):
    pass


def preprocess_validation_task1():
    path = "data/validation.csv"
    preprocess_test_task1(path)


load_agoda_data()
# preprocess_validation_task1()
