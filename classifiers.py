from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


classificadors = [{'title': 'Gradient Boosting',
                   'params': {'max_depth': range(5, 16, 2), 'min_samples_split': [10, 100, 500, 1000],
                              'n_estimators': [10, 50, 100]},
                   'clf': GradientBoostingClassifier(learning_rate=0.1, max_features='sqrt',
                                                     subsample=0.8,
                                                     random_state=42),
                   'enabled': True, 'pca': False},
                  {'title': 'Support Vector Machine',
                   'params': {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1, 10]},
                   'clf': SVC(kernel='rbf'),
                   'enabled': True, 'pca': False},
                  # {'title': 'Multi Layer Perceptron',
                  #  'params': {'hidden_layer_sizes': [(2, 2), (2, 3), (3, 3), (3, 4), (2, 4), (2, 5), (4, 2), (4, 3),
                  #                                    (4, 4), (4, 5)], 'alpha': [1e-4, 1e-5, 1e-6, 1e-7]},
                  #  'clf': MLPClassifier(solver='adam', alpha=1e-5, random_state=42),
                  #  'enabled': True, 'pca': False},
                  {'title': 'Random Forest',
                   'params': {'n_estimators': [10, 50, 100, 150], 'max_depth': [1, 5, 10, 15, 20, 50]},
                   'clf': RandomForestClassifier(random_state=42),
                   'enabled': True, 'pca': False}]

# Dividir el conjunt en entrenament i test
#
# d = pd.DataFrame(data=X)
# print(X.shape, d.shape )
#
# # Compute the correlation matrix
# corr = d.corr()
#
# # Generate a mask for the upper triangle
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
#
# # Set up the matplotlib figure
# f = plt.figure(figsize=(11, 9))
#
# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(120, 10, as_cmap=True)
#
# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap) # vmax=.3, center=0,                square=True, linewidths=.5, cbar_kws={"shrink": .5})
#
# plt.savefig(str(size) + "_heatmap.png")