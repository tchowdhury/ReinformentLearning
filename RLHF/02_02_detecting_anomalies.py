import numpy as np
from sklearn.cluster import KMeans
import warnings
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "8" 

# 1. Get Low confidence predictions
texts = ['This movie was fantastic!', "I don't know if I liked it or not.", 'The book was incredibly boring.']

# probability distributions for each feedback text
prob_dists = np.array([[0.4942085, 0.5057915],[0.5551105 , 0.44488952],[0.46784133, 0.5321586 ]])

def least_confidence(prob_dist):
	simple_least_conf = np.nanmax(prob_dist)
	num_labels = float(prob_dist.size)
	# number of labels 
	least_conf = (1 - simple_least_conf) * (num_labels / (num_labels - 1))
	return least_conf

# Define the function to filter the indices of probability distributions for which the confidence is below a given threshold.
def filter_low_confidence_predictions(prob_dists, threshold=0.5):
    filtered_indices = [i for i, prob_dist in enumerate(prob_dists) if least_confidence(prob_dist) > threshold]
    return filtered_indices

# Find the indices
filtered_indices = filter_low_confidence_predictions(prob_dists)

high_confidence_texts = [texts[i] for i in filtered_indices]
print("High-confidence texts:", high_confidence_texts)

# 2. Clustering the feedback texts using KMeans

confidences = np.array([[0.34], [0.72], [0.51], [0.68]])

def detect_anomalies(data, n_clusters=3):
    # Initialize k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_

    # Calculate distances from cluster centers
    distances = np.linalg.norm(data - centers[clusters], axis=1)
    return distances

anomalies = detect_anomalies(confidences)
print(anomalies)