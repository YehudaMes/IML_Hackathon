import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split

DATA_PATH = "../../../Agoda - Data/agoda_cancellation_train.csv"  # todo: at end this one should be used!
# DATA_PATH = "data/train.csv"

# pd.set_option('display.max_rows', None)

FULL_TRAIN_MEAN_DICT = {'hotel_star_rating': 3,
                        'no_of_adults': 2,
                        'no_of_extra_bed': 0,
                        'no_of_room': 1,
                        'no_of_children': 0,
                        'days_before_cancelled': 10}

SMALL_TRAIN_MEAN_DICT = {'hotel_star_rating': 3,
                         'no_of_adults': 2,
                         'no_of_extra_bed': 0,
                         'no_of_room': 1,
                         'no_of_children': 0,
                         'days_before_cancelled': 10}

COLS_TO_DROP = ["h_booking_id", "hotel_id", "cancellation_datetime", "checkin_date", "checkout_date",
                "hotel_brand_code", "hotel_chain_code", "hotel_live_date", "booking_datetime",
                "request_nonesmoke", "request_latecheckin", "request_highfloor", "request_largebed",
                "request_twinbeds", "request_airport", "request_earlycheckin",
                "h_customer_id", "is_user_logged_in", "original_payment_currency",
                "original_payment_method", "guest_is_not_the_customer", "language"]

COLUMNS_TO_DUMMIES = [
    "accommadation_type_name", "charge_option", "guest_nationality_country_name",
    "hotel_country_code", "hotel_area_code", "is_first_booking", "cancellation_policy_code",
    "original_payment_type", "hotel_city_code", "origin_country_code", "customer_nationality"]

COLUMS_TO_CHECK = ["charge_option", "guest_nationality_country_name"]


def produce_days_before_cancelling_feature(df):
    df["cancellation_datetime"] = pd.to_datetime(df["cancellation_datetime"])
    df["checkin_date"] = pd.to_datetime(df["checkin_date"])
    df["days_before_cancelled"] = df["checkin_date"] - df["cancellation_datetime"]
    df["days_before_cancelled"] = df["days_before_cancelled"].fillna(pd.Timedelta(0)).dt.days.astype(int)


# Regression
def preprocess_train_task2(data_path):
    df = pd.read_csv(data_path)

    df["cancellation_indicator"] = df["cancellation_datetime"].notnull().astype(int)  # Task 1 labeling

    produce_days_before_cancelling_feature(df)
    df = df[df['hotel_star_rating'].isin(np.arange(0, 5.5, 0.5))]
    df['hotel_star_rating'] = df['hotel_star_rating'].clip(lower=0)

    df = df.drop(COLS_TO_DROP, axis=1)

    df = df[df['hotel_star_rating'].isin(np.arange(0, 5.5, 0.5))]
    df = df[df['no_of_adults'].isin(np.arange(0, 21, 1))]
    df = df[df['no_of_extra_bed'].isin(np.arange(0, 6, 1))]
    df = df[df['no_of_room'].isin(np.arange(1, 10, 1))]
    df = df[df['no_of_children'].isin(np.arange(0, 11, 1))]
    # df = df[df['days_before_cancelled'].isin(np.arange(0, 360, 1))]
    df.loc[df['days_before_cancelled'] < 0, 'days_before_cancelled'] = 0

    # print(df['accommadation_type_name'].unique())
    # df = pd.get_dummies(df, columns=['accommadation_type_name'])
    # yp = 'accommadation_type_name'

    # df = pd.get_dummies(df, columns=COLUMNS_TO_DUMMIES)
    #
    # columns = []
    # for col in COLUMNS_TO_DUMMIES:
    #     filtered_cols = df.filter(regex=f'^{col}')
    #     columns += filtered_cols.columns.tolist()
    #
    # print(columns)
    # df = pd.get_dummies(df, columns=COLUMNS_TO_DUMMIES)
    preprocess_test_task2(df)


def preprocess_test_task2(df):
    # cols = "guest_nationality_country_name", "customer_nationality"
    # df = pd.get_dummies(df, columns=cols)
    # cancellation_indicator

    # and count the occurrences of 1s and 0s
    counts = df.groupby(['customer_nationality', 'cancellation_indicator']).size().unstack(fill_value=0)

    # Create a bar plot
    fig = px.histogram(df, x='customer_nationality', color='cancellation_indicator', barmode='group')
    # fig = px.histogram(counts, x='customer_nationality', color='cancellation_indicator', barmode='group')

    # Update layout
    fig.update_layout(
        title='Cancellation Indicator by Customer Nationality',
        xaxis_title='Customer Nationality',
        yaxis_title='Count',
        legend_title='Cancellation Indicator',
        xaxis={'categoryorder': 'total descending'},
        barmode='group'
    )

    # Show the plot
    fig.show()

    # Group the DataFrame by 'customer_nationality' and calculate the value counts of 'cancellation_indicator'
    grouped_counts = df.groupby('customer_nationality')['cancellation_indicator'].value_counts(normalize=True).unstack()

    # Create a new column with the ratio of 1s
    grouped_counts['Ratio of 1s'] = grouped_counts[1] * 100

    # Create a new column with the ratio of 0s
    grouped_counts['Ratio of 0s'] = grouped_counts[0] * 100

    # Reset the index to have 'customer_nationality' as a regular column
    grouped_counts = grouped_counts.reset_index()

    # Create a bar plot
    fig = px.bar(grouped_counts, x='customer_nationality', y=['Ratio of 1s', 'Ratio of 0s'],
                 title='Cancellation Indicator by Customer Nationality',
                 labels={'value': 'Ratio', 'variable': 'Cancellation Indicator'})

    # Update layout
    fig.update_layout(
        yaxis=dict(tickformat='.1f', title='Ratio (%)'),
        xaxis_title='Customer Nationality',
        legend_title='Cancellation Indicator',
        xaxis={'categoryorder': 'total descending'}
    )

    # Show the plot
    fig.show()


if __name__ == "__main__":
    preprocess_train_task2(DATA_PATH)
