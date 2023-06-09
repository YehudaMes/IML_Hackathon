import pandas as pd
import numpy as np

from .preprocess_util import common_column_edit

COLUMNS_DATA_PATH = './columns_data/task2_columns.txt'
DATA_PATH = "agoda_data/agoda_cancellation_train.csv"  # todo: at end this one should be used!
# DATA_PATH = "data/train.csv"

# pd.set_option('display.max_rows', None)

FULL_TRAIN_MEAN_DICT = {'hotel_star_rating': 3,
                        'no_of_adults': 2,
                        'no_of_extra_bed': 0,
                        'no_of_room': 1,
                        'no_of_children': 0,
                        'booking_to_checkin': 35}

COLS_TO_DROP = ["h_booking_id",
                "hotel_id",
                "cancellation_datetime",
                "checkin_date",
                "checkout_date",
                "hotel_brand_code",
                "hotel_area_code",
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
    "hotel_chain_code",
    "original_payment_type",
    "hotel_city_code",
    "origin_country_code",
    "customer_nationality"]

COLUMS_TO_CHECK = ["charge_option", "guest_nationality_country_name"]




# Regression
def preprocess_train_task2_raw(data_path):
    df = pd.read_csv(data_path)

    df["cancellation_indicator"] = df["cancellation_datetime"].notnull().astype(int)  # Task 1 labeling

    df=common_column_edit(df, COLS_TO_DROP, COLUMNS_TO_DUMMIES)

    df = df[df['hotel_star_rating'].isin(np.arange(0, 5.5, 0.5))]
    df['hotel_star_rating'] = df['hotel_star_rating'].clip(lower=0)

    df = df[df['hotel_star_rating'].isin(np.arange(0, 5.5, 0.5))]
    df = df[df['no_of_adults'].isin(np.arange(0, 21, 1))]
    df = df[df['no_of_extra_bed'].isin(np.arange(0, 6, 1))]
    df = df[df['no_of_room'].isin(np.arange(1, 10, 1))]
    df = df[df['no_of_children'].isin(np.arange(0, 11, 1))]



    return df

def task_2_train_preprocess(path, save_columns=True):
    df = preprocess_train_task2_raw(path)
    train_y = df.original_selling_amount
    classifier_y = df.cancellation_indicator
    train_X = df.drop(columns=['original_selling_amount', 'cancellation_indicator'])
    # Save the column names as a text file
    if save_columns:
        with open(COLUMNS_DATA_PATH, 'w') as file:
            file.write('\n'.join(train_X.columns))
    else:
        with open(COLUMNS_DATA_PATH, 'r') as file:
            desired_columns = file.read().splitlines()
            train_X = train_X.reindex(columns=desired_columns, fill_value=0)
    return train_X, train_y, classifier_y

def fill_means(df):
    df.loc[(df['booking_to_checkin'] < 0) | (df['booking_to_checkin'] > 350),
           'booking_to_checkin'] = FULL_TRAIN_MEAN_DICT['booking_to_checkin']


def preprocess_test_task2(path):
    df = pd.read_csv(path)
    ids=df.h_booking_id
    # Read columns feature of Trained model
    with open(COLUMNS_DATA_PATH, 'r') as file:
        desired_columns = file.read().splitlines()
    df=common_column_edit(df, COLS_TO_DROP, COLUMNS_TO_DUMMIES)

    df.loc[(~df['hotel_star_rating'].isin(np.arange(0, 5, 0.5))), 'hotel_star_rating'] = FULL_TRAIN_MEAN_DICT['hotel_star_rating']
    fill_means(df)
    df = df.drop(columns=COLS_TO_DROP,errors='ignore')

    df = df.reindex(columns=desired_columns, fill_value=0)
    return ids,df


if __name__ == "__main__":
    preprocess_train_task2_raw(DATA_PATH)
