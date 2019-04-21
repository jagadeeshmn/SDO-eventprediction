from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso


def Perform_Lasso(X,Y):
    lasso = Lasso()
    parameters = {'alpha':[1e-15,1e-10,1e-8,1e-4,1e-3,1e-2,1,5,10,20]}
    lasso_regressor = GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
    lasso_regressor.fit(X,Y)
    print(lasso_regressor.best_params_)
    print(lasso_regressor.best_score_)
