import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report

# Define the minimum and maximum values for X and Y
# that will be used in the mesh grid
def visualize_classifier(classifier, X, y):
  min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
  min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

print("\nLoad file")

# Load input data
input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Separate input data into three classes based on labels
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])
class_2 = np.array(X[y==2])

# Split the data into training and testing datasets 
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

# Define the parameter grid 
parameter_grid = [ {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
                   {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
                 ]

metrics = ['precision_weighted', 'recall_weighted']

for metric in metrics:
    print("\n##### Searching optimal parameters for", metric)

    classifier = GridSearchCV(
            ExtraTreesClassifier(random_state=0), 
            parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)
    
    print("\nGrid scores for the parameter grid:")
    for results in classifier.cv_results_:
        print(results)

    print("\nBest parameters:", classifier.best_params_)

    y_pred = classifier.predict(X_test)
    print("\nPerformance report:\n")
    print(classification_report(y_test, y_pred))
