import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# DATA_PATH = "../../../agoda_data/agoda_cancellation_train.csv" // todo: at end this one should be used!
DATA_PATH = "data/train.csv"

COLS_TO_DROP = ["h_booking_id", "hotel_id", "cancellation_datetime", "checkin_date", "checkout_date",
                "hotel_brand_code", "hotel_chain_code", "hotel_live_date", "booking_datetime",
                "request_nonesmoke", "request_latecheckin", "request_highfloor", "request_largebed",
                "request_twinbeds", "request_airport", "request_earlycheckin", "customer_nationality",
                "h_customer_id", "no_of_children", "no_of_adults", "no_of_extra_bed", "no_of_room",
                "origin_country_code", "hotel_city_code", "is_user_logged_in", "original_payment_currency",
                "original_payment_method", "guest_is_not_the_customer", "language"]

pd.set_option('display.max_rows', None)

COLUMNS_TO_DUMMIES = [
    "accommadation_type_name", "charge_option", "guest_nationality_country_name",
    "hotel_country_code", "hotel_area_code", "is_first_booking", "cancellation_policy_code",
    "original_payment_type"
]


# Classification
def load_data():
    df = pd.read_csv(DATA_PATH)

    df["cancellation_indicator"] = df["cancellation_datetime"].notnull().astype(int)  # Task 1 labeling

    produce_days_before_cancelling_feature(df)  # todo Takes care on negative days

    df = df.drop(COLS_TO_DROP, axis=1)
    df = df[df['hotel_star_rating'].isin(np.arange(0, 5.5, 0.5))]

    counts = df.groupby('is_first_booking')['cancellation_indicator'].apply(
        lambda x: pd.Series({'nan_count': x.isna().sum(), 'not_nan_count': x.notna().sum()}))

    # todo dummies: "accommadation_type_name", "charge_option", "guest_nationality_country_name",
    #   "hotel_country_code", "hotel_area_code", "is_first_booking", "cancellation_policy_code",
    #    "original_payment_type"

    # x = df['guest_is_not_the_customer', 'cancellation_indicator'].groupby('guest_is_not_the_customer')
    # grouped_counts = df.groupby('guest_is_not_the_customer')['cancellation_indicator'].value_counts()
    # print(grouped_counts)

    df = pd.get_dummies(df, columns=COLUMNS_TO_DUMMIES)

    return df
    print(df.columns.size)
    # print(df.dtypes)

    # for col in ['sqft_lot', 'sqft_living', 'price']:
    #     df = df[df[col] > 0]
    #
    # for col in ['sqft_basement', 'sqft_above', 'yr_renovated']:
    #     df = df[df[col] >= 0]
    #
    # df = df[df['sqft_lot'] < 2500000]
    # df = df[df['sqft_living'] < 11000]
    # df = df[df['sqft_above'] < 8000]
    # df = df[df['sqft_basement'] < 2750]
    # df = df[df['price'] < 7100000]
    # df = df[(df['yr_renovated'] == 0) | ((df['yr_renovated'] >= 1900) & (df['yr_renovated'] <= 2015))]
    #
    # df = df[df['condition'].isin(range(1, 6)) &
    #         df['yr_built'].isin(range(1900, 2016)) &
    #         df['bedrooms'].isin(range(1, 15)) &
    #         df['bathrooms'].isin(np.arange(1, 5, 0.25)) &  # todo maybe less bathrooms!
    #         df['floors'].isin(np.arange(1, 4, 0.5)) &
    #         df['view'].isin(range(0, 5)) &
    #         df['grade'].isin(range(1, 14)) &
    #         df['waterfront'].isin([0, 1])]
    #
    # df = df.dropna().drop_duplicates()
    # df['zipcode'] = df['zipcode'].astype(int)
    # df = df[df['zipcode'].isin(range(98001, 98289))]
    # df = pd.get_dummies(df, prefidf='zipcode', columns=['zipcode'])
    # df['yr_built'] = df['yr_built'].astype(int)
    #
    # df['yr_renovated'] = np.where(df['yr_renovated'] < 1985, 0, df['yr_renovated'])
    #
    # df['yr_built'] = (df['yr_built'] // 10)
    # df = pd.get_dummies(df, prefidf='yr_built', columns=['yr_built'])


def produce_days_before_cancelling_feature(df):
    df["cancellation_datetime"] = pd.to_datetime(df["cancellation_datetime"])
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    df["days_before_cancelled"] = df["checkin_date"] - df["cancellation_datetime"]
    df["days_before_cancelled"] = df["days_before_cancelled"].fillna(pd.Timedelta(0)).dt.days.astype(int)


load_data()
