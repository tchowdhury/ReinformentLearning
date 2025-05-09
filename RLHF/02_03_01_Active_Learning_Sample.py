import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner
import matplotlib.pyplot as plt

# 1. Create a toy dataset
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, 
                           n_redundant=0, random_state=42)

# 2. Start with a tiny labeled dataset
n_initial = 10
initial_idx = np.random.choice(range(len(X)), size=n_initial, replace=False)
X_initial, y_initial = X[initial_idx], y[initial_idx]

# Remaining data (unlabeled pool)
X_pool = np.delete(X, initial_idx, axis=0)
y_pool = np.delete(y, initial_idx, axis=0)

# 3. Define the learner
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy=lambda classifier, X_pool: (
        np.argmax(classifier.predict_proba(X_pool).max(axis=1)),
    ),
    X_training=X_initial, y_training=y_initial
)

# 4. Active learning loop
n_queries = 10
for i in range(n_queries):
    query_idx, query_instance = learner.query(X_pool)

    # "Oracle" gives the correct label (simulate human labeling)
    learner.teach(X_pool[query_idx].reshape(1, -1), y_pool[query_idx].reshape(1, ))

    # Remove the newly labeled sample from the pool
    X_pool = np.delete(X_pool, query_idx, axis=0)
    y_pool = np.delete(y_pool, query_idx, axis=0)

# 5. Plotting
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm', edgecolor='k', s=30)
plt.title('Active Learning Toy Example')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
