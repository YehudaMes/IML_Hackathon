import pandas as pd


def booking_to_checkin_feature(df):
    df["booking_to_checkin"] = pd.to_datetime(df["checkin_date"]) - pd.to_datetime(df["booking_datetime"])
    df["booking_to_checkin"] = df["booking_to_checkin"].fillna(pd.Timedelta(0)).dt.days.astype(int)
    df["booking_to_checkin"] = df['booking_to_checkin'].apply(lambda x: max(0, x))
    df["checkin_to_checkout"] = pd.to_datetime(df["checkout_date"]) - pd.to_datetime(df["checkin_date"])
    df["checkin_to_checkout"] = df["checkin_to_checkout"].fillna(pd.Timedelta(1)).dt.days.astype(int)


def common_column_edit(df, cols_to_drop, cols_to_dummies):
    booking_to_checkin_feature(df)
    df["is_first_booking"] = df["is_first_booking"].astype(int)
    df = df.drop(cols_to_drop, axis=1)
    df = pd.get_dummies(df, columns=cols_to_dummies)
    return df
