# Suggest the optimal cancellation and pricing policy
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

from task_number.code.hackathon_code.preproccer_task1 import load_train_data_task1


# Assuming df is your preprocessed dataframe with relevant features
def cluster(df):
    # Extract relevant features for clustering
    features = ['original_selling_amount', 'booking_to_checkin', 'hotel_star_rating']
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    # Apply K-means clustering
    n_clusters = 4  # Choose the desired number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    # Add the cluster labels to the original dataframe
    df['cluster'] = clusters
    # Analyze the characteristics of each cluster
    cluster_analysis = df.groupby('cluster').mean()
    # Implement pricing strategy based on the cluster analysis
    for cluster_id, cluster_data in cluster_analysis.iterrows():
        # Access cluster-specific information such as cancellation rate, booking lead time, etc.
        cancellation_rate = cluster_data['cancellation_rate']
        booking_lead_time = cluster_data['booking_lead_time']
        # Implement pricing policy based on the cluster-specific information
    # Monitor and evaluate the performance of the pricing strategy
    # Continuously assess cancellation rates, revenue, and customer satisfaction metrics
    # Make adjustments and refinements as needed


def plot(df):

    # Assuming df is your classified dataframe with the relevant features and cluster labels

    # Define the features for visualization
    x_feature = 'booking_to_checkin'
    y_feature = 'original_selling_amount'
    z_feature = 'hotel_star_rating'
    color_feature = 'cluster'

    # Create a 3D scatter plot with color-coded clusters
    fig = px.scatter_3d(df, x=x_feature, y=y_feature, z=z_feature, color=color_feature)

    # Customize the plot layout
    fig.update_layout(
        title="Customer Segmentation based on Booking, Selling Amount, and Hotel Star Rating",
        scene=dict(
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            zaxis_title=z_feature
        )
    )

    # Show the plot
    fig.show()
if __name__ == "__main__":

    plot(load_train_data_task1()[0])