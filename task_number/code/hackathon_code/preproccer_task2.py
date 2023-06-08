import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split

COLUMNS_DATA_PATH = './columns_data/task2_columns.txt'
DATA_PATH = "agoda_data/agoda_cancellation_train.csv"  # todo: at end this one should be used!
# DATA_PATH = "data/train.csv"

# pd.set_option('display.max_rows', None)

FULL_TRAIN_MEAN_DICT = {'hotel_star_rating': 3,
                        'no_of_adults': 2,
                        'no_of_extra_bed': 0,
                        'no_of_room': 1,
                        'no_of_children': 0,
                        'days_before_cancelled': 10}

COLS_TO_DROP = ["h_booking_id", "hotel_id", "cancellation_datetime", "checkin_date", "checkout_date",
                "hotel_brand_code", "hotel_area_code", "hotel_live_date", "booking_datetime",
                "request_nonesmoke", "request_latecheckin", "request_highfloor", "request_largebed",
                "request_twinbeds", "request_airport", "request_earlycheckin",
                "h_customer_id", "is_user_logged_in", "original_payment_currency",
                "original_payment_method", "guest_is_not_the_customer", "language"]

COLUMNS_TO_DUMMIES = [
    "accommadation_type_name", "charge_option", "guest_nationality_country_name",
    "hotel_country_code", "hotel_chain_code", "is_first_booking", "cancellation_policy_code",
    "original_payment_type", "hotel_city_code", "origin_country_code", "customer_nationality"]

COLUMS_TO_CHECK = ["charge_option", "guest_nationality_country_name"]


def booking_to_checkin_feature(df):
    df["booking_to_checkin"] = pd.to_datetime(df["checkin_date"]) - pd.to_datetime(df["booking_datetime"])
    df["booking_to_checkin"] = df["booking_to_checkin"].fillna(pd.Timedelta(0)).dt.days.astype(int)

    df["checkin_to_checkout"] = pd.to_datetime(df["checkout_date"]) - pd.to_datetime(df["checkin_date"])
    df["checkin_to_checkout"] = df["checkin_to_checkout"].fillna(pd.Timedelta(1)).dt.days.astype(int)


# Regression
def preprocess_train_task2(data_path):
    df = pd.read_csv(data_path)

    df["cancellation_indicator"] = df["cancellation_datetime"].notnull().astype(int)  # Task 1 labeling
    booking_to_checkin_feature(df)

    df = df[df['hotel_star_rating'].isin(np.arange(0, 5.5, 0.5))]
    df['hotel_star_rating'] = df['hotel_star_rating'].clip(lower=0)

    df = df.drop(COLS_TO_DROP, axis=1)

    df = df[df['hotel_star_rating'].isin(np.arange(0, 5.5, 0.5))]
    df = df[df['no_of_adults'].isin(np.arange(0, 21, 1))]
    df = df[df['no_of_extra_bed'].isin(np.arange(0, 6, 1))]
    df = df[df['no_of_room'].isin(np.arange(1, 10, 1))]
    df = df[df['no_of_children'].isin(np.arange(0, 11, 1))]

    # df = pd.get_dummies(df, columns=['accommadation_type_name'])
    # yp = 'accommadation_type_name'
    # df = pd.get_dummies(df, columns=COLUMNS_TO_DUMMIES)

    # Save the column names as a text file
    with open(COLUMNS_DATA_PATH, 'w') as file:
        file.write('\n'.join(df.columns))

    return df


def preprocess_test_task2(path):
    df = pd.read_csv(path)

    # Read columns feature of Trained model
    with open(COLUMNS_DATA_PATH, 'r') as file:
        desired_columns = file.read().splitlines()

    desired_columns.remove("cancellation_indicator")
    desired_columns.remove("original_selling_amount")
    df = df.drop(COLS_TO_DROP, axis=1)

    df = df.reindex(columns=desired_columns, fill_value=0)
    return df


if __name__ == "__main__":
    preprocess_train_task2(DATA_PATH)
