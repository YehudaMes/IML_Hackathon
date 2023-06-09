# Churn prediction Model - Find minimal features set which predict the best that person going cancel
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
import pandas as pd
import plotly.express as px

from preproccer_task1 import load_train_agoda_data_task1

TRAIN_AGODA_PATH = "./hackathon_code/agoda_data/agoda_cancellation_train.csv"


def lasso(df, lam):
    # Separate the features (X) and the target variable (y)
    X = df.drop('cancellation_indicator', axis=1)
    y = df['cancellation_indicator']

    lasso = Lasso(alpha=lam)
    lasso.fit(X, y)

    # Get the coefficients and corresponding feature names
    coefficients = lasso.coef_
    feature_names = X.columns

    # Create a DataFrame to store the coefficients and feature names
    coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

    # Sort the DataFrame by the absolute value of the coefficients in descending order
    coefficients_df = coefficients_df.reindex(coefficients_df['Coefficient'].abs().sort_values(ascending=False).index)

    # Print the top contributing features
    top_features = coefficients_df.head(10)['Feature'].tolist()
    print("\n")
    print(f"Top contributing features of Lasso with Lambda={lam}:")
    for feature in top_features:
        print(feature)


def uni(df):
    X = df.drop('cancellation_indicator', axis=1)
    y = df['cancellation_indicator']

    # Assuming X is your feature matrix and y is the corresponding target variable (cancellation)
    # X should be a pandas DataFrame or a numpy array

    k = 5  # Select the number of top features (k) you want to keep

    print("Using SelectKBest, first is with chi2 and second with f_classif. both with k=5 (features)")

    for selector in [SelectKBest(score_func=chi2, k=k), SelectKBest(score_func=f_classif, k=k)]:
        print()
        # Fit the selector to the features and target variable
        selector.fit(X, y)

        # Get the scores and corresponding feature names
        scores = selector.scores_
        feature_names = X.columns

        # Create a dictionary to store feature names and their corresponding scores
        feature_scores = {feature: score for feature, score in zip(feature_names, scores)}

        # Sort the features based on their scores in descending order
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        # Print the top k features with the highest scores
        top_k_features = sorted_features[:k]
        for feature, score in top_k_features:
            print(f"Feature: {feature}, Score: {score}")


def ratio_plots(df):
    cols = ['cancellation_policy_code', 'customer_nationality', 'charge_option']
    col2 = 'cancellation_indicator'

    for col1 in cols:
        # Calculate the count of col1 for each category
        counts = df[col1].value_counts().reset_index()
        counts.columns = [col1, 'Count']
        threshold = 30
        # Filter out col1 categories with less than 20 rows
        counts_filtered = counts[counts['Count'] > threshold]

        # Filter the original DataFrame based on the filtered col1 categories
        df_filtered = df[df[col1].isin(counts_filtered[col1])]

        # Group the filtered DataFrame by col1 and calculate the value counts of col2
        grouped_counts = df_filtered.groupby(col1)[col2].value_counts(normalize=True).unstack()

        # Create a new column with the ratio of 1s
        grouped_counts['Ratio of 1s'] = grouped_counts[1] * 100

        # Create a new column with the ratio of 0s
        grouped_counts['Ratio of 0s'] = grouped_counts[0] * 100

        # Reset the index to have col1 as a regular column
        grouped_counts = grouped_counts.reset_index()

        # Create a bar plot
        fig = px.bar(grouped_counts, x=col1, y=['Ratio of 1s', 'Ratio of 0s'],
                     title='Cancellation Indicator by ' + col1 + f' (Categories with >{threshold} Rows)',
                     labels={'value': 'Ratio', 'variable': 'Cancellation Indicator'})

        # Update layout
        fig.update_layout(
            yaxis=dict(tickformat='.1f', title='Ratio (%)'),
            xaxis_title=col1,
            legend_title='Cancellation Indicator',
            xaxis={'categoryorder': 'total descending'},
            bargap=0.2  # Increase the value to add more space between the histograms
        )

        # Add the counts as annotations
        for i, row in counts_filtered.iterrows():
            fig.add_annotation(
                x=row[col1],
                y=0,  # Place the annotation below the graph
                text=str(row['Count']),
                showarrow=False,
                font=dict(size=9),
                xanchor='center',  # Center the text under each category
                yshift=-40,  # Adjust the vertical position of the annotation
                textangle=90  # Rotate the text by 180 degrees
            )

        # Show the plot
        fig.show()


def plots_and_info():
    X, y = load_train_agoda_data_task1(True)
    X['cancellation_indicator'] = y
    uni(X)
    lasso(X, 0.1)

    X, y = load_train_agoda_data_task1(False)
    X['cancellation_indicator'] = y
    ratio_plots(X)