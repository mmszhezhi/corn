from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

X,y = load_boston(return_X_y=True)

param_grid = {
    # 'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3,5,6,7,8,10],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [10,50,100,150]
}

rf = GradientBoostingRegressor()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,
                          cv = 2, n_jobs = -1, verbose = 2,scoring=["neg_mean_squared_error"],return_train_score=True,refit="neg_mean_squared_error")
grid_search.fit(X, y)
# grid_search.best_estimator_
print(grid_search.cv_results_)