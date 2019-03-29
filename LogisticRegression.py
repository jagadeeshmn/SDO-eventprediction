import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

def Perform_Logistic_Regression(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X.T,Y.T,test_size=0.7)
    logreg = LogisticRegression(fit_intercept=True,solver='lbfgs')
    for i in range(y_train.shape[0]):
		logreg.fit(x_train,y_train[:,i])
		print(logreg.coef_)
	logreg.predict(x_test)
    print(logreg.score(x_test,y_test))