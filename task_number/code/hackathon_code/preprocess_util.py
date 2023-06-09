from copy import copy

import pandas as pd
import numpy as np




def booking_to_checkin_feature(df):
    df["booking_to_checkin"] = pd.to_datetime(df["checkin_date"]) - pd.to_datetime(df["booking_datetime"])
    df["booking_to_checkin"] = df["booking_to_checkin"].fillna(pd.Timedelta(0)).dt.days.astype(int)
    df["booking_to_checkin"] = df['booking_to_checkin'].apply(lambda x: max(0, x))
    df["checkin_to_checkout"] = pd.to_datetime(df["checkout_date"]) - pd.to_datetime(df["checkin_date"])
    df["checkin_to_checkout"] = df["checkin_to_checkout"].fillna(pd.Timedelta(1)).dt.days.astype(int)

def cancellation_policy_cost_function(x: list, func):
    vecay_len, booking_time, policies=int(x[0]), int(x[1]), x[2:]
    if not policies:
        return 0
    costs=[]
    for policy in policies:
        if "D" in policy:
            days, cost=policy.split("D")
            days=min(int(days),booking_time)
        else:
            cost=policy
            days=0
        if "N" in cost:
            cost_in_days=min(int(cost[:-1]), vecay_len)
        else:
            cost_in_days=vecay_len*int(cost[:-1])/100
        costs.append(cost_in_days*(days+1))
    return func(costs)



def cancellation_cost_feature(df):
    df.cancellation_policy_code = df.cancellation_policy_code.replace({"UNKNOWN": "0D0N"})
    df.cancellation_policy_code = df.checkin_to_checkout.astype(str) + "_" + df.booking_to_checkin.astype(
        str) + "_" + df.cancellation_policy_code
    df["max_cancellation_penalty"] = df.cancellation_policy_code.apply(
        lambda x: cancellation_policy_cost_function(x.split("_"), np.max))
    df["min_cancellation_penalty"] = df.cancellation_policy_code.apply(
        lambda x: cancellation_policy_cost_function(x.split("_"), np.min))


def common_column_edit(df, cols_to_drop, cols_to_dummies):
    booking_to_checkin_feature(df)
    cancellation_cost_feature(df)
    df["is_first_booking"] = df["is_first_booking"].astype(int)
    df = pd.get_dummies(df, columns=cols_to_dummies, prefix="dummy")
    cols_to_drop=copy(cols_to_drop)
    for col in df.columns:
        if "dummy" in col and df[col].sum()<=18:
            cols_to_drop.append(col)
    df = df.drop(cols_to_drop, axis=1)
    return df
