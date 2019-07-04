from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


classificadors = [
                  {'title': 'Gradient Boosting',
                   'params': {'max_depth': range(5, 21, 2), 'min_samples_split': [5, 10, 100, 500, 1000],
                              'n_estimators': [10, 50, 100, 150]},
                   'clf': GradientBoostingClassifier(learning_rate=0.1, max_features='sqrt',
                                                     subsample=0.8,
                                                     random_state=42),
                   'enabled': True},
                  {'title': 'Nearest Neighbours',
                   'params': {'n_neighbors': [5, 10, 15, 20], 'weights': ['uniform', 'distance'],
                              'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute']},
                   'clf': neighbors.KNeighborsClassifier(),
                   'enabled': True},

                  {'title': 'Support Vector Machine',
                   'params': { 'kernel': ['linear'],'C': [1000], 'gamma': [ 0.1, 10, 100]},
                   'clf': SVC(),
                   'enabled': True},
                  {'title': 'Multi Layer Perceptron',
                   'params': {'hidden_layer_sizes': [(20,40,20), (60,80,60)], 'alpha': [1e-3,1e-4, 1e-5, 1e-6, 1e-7]},
                   'clf': MLPClassifier(solver='adam', alpha=1e-5, activation='logistic',random_state=42),
                   'enabled': True},
                  {'title': 'Random Forest',
                   'params': {'n_estimators': [10, 50, 100, 150, 200, 250], 'max_depth': [1, 5, 10, 15, 20, 50, 100, 200]},
                   'clf': RandomForestClassifier(random_state=42),
                   'enabled': True, }
                  ]

