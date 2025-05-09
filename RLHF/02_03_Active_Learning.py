from sklearn.linear_model import LogisticRegression
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
import numpy as np


# 1. Implementing an active learning pipeline

### 
# Initialize an ActiveLearner object.
# Use LogisticRegression as the estimator.
# Use uncertainty sampling as the query strategy.
# Initialize the learner with labeled training data.
###

y_labeled = [0,1,1,0,0,0,0,1,1,0,0,0,1,0,1,0,0,1,1,1,0,1,0,1,0,0,1,1,1,0,0,1,1,0,0,1,1
,1,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0,1,0,1,0,0,0,1,1,0,1,0,0,0,0,0,1,1,0,0,1,1
,0,1,0,0,0,0,0,1,0,1,0,0,1,0,1,0,1,0,1,1,0,1,0,1,1,1]


X_labeled = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9],
 [0.9, 1.0], [1.0, 1.1], [1.1, 1.2], [1.2, 1.3], [1.3, 1.4], [1.4, 1.5], [1.5, 1.6], [1.6, 1.7],
[1.7, 1.8], [1.8, 1.9], [1.9, 2.0], [2.0, 2.1], [2.1, 2.2], [2.2, 2.3], [2.3, 2.4], [2.4, 2.5],
[2.5, 2.6], [2.6, 2.7], [2.7, 2.8], [2.8, 2.9], [2.9, 3.0], [3.0, 3.1], [3.1, 3.2], [3.2, 3.3],
[3.3, 3.4], [3.4, 3.5], [3.5, 3.6], [3.6, 3.7], [3.7, 3.8], [3.8, 3.9], [3.9, 4]]

X_unlabeled = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8], [0.8, 0.9],
 [0.9, 1.0], [1.0, 1.1], [1.1, 1.2], [1.2, 1.3], [1.3, 1.4], [1.4, 1.5], [1.5, 1.6], [1.6, 1.7],
[1.7, 1.8], [1.8, 1.9], [1.9, 2.0], [2.0, 2.1], [2.1, 2.2], [2.2, 2.3], [2.3, 2.4], [2.4, 2.5],
[2.5, 2.6], [2.6, 2.7], [2.7, 2.8], [2.8, 2.9], [2.9, 3]]

learner = ActiveLearner(
    estimator=LogisticRegression(),
    query_strategy=uncertainty_sampling,
    X_training=X_labeled,  # Initial labeled data
    y_training=y_labeled   # Initial labels
)

# 2. Active learning loop

# Implement a loop that will run 10 queries.
# In each iteration, have the learner teach itself using the current labeled data.
# Use the learner to query the most uncertain data points from the unlabeled data, setting the number of instances to 5.
# Update the unlabeled dataset accordingly.

# Set the number of queries
n_queries = 10
for _ in range(n_queries):
    # Use the current labeled data
    learner.teach(X_labeled, y_labeled)
    # Query from unlabeled data
    query_idx, _ = learner.query(X_unlabeled, n_instances=5)  
    X_new, y_new = X_unlabeled[query_idx], y_labeled[query_idx]  
    X_labeled = np.vstack((X_labeled, X_new))  
    y_labeled = np.append(y_labeled, y_new)  
    # Update the unlabeled dataset
    X_unlabeled = np.delete(X_unlabeled, query_idx, axis=0) 

