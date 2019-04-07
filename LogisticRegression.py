import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

def Perform_Logistic_Regression(X,Y):
    x_train, x_test, y_train, y_test = train_test_split(X.T,Y.T,test_size=0.7)
    logreg = LogisticRegression(fit_intercept=True,solver='lbfgs')
    for i in range(y_train.shape[0]):
        logreg.fit(x_train,y_train[:,i])
        print(logreg.coef_)
        logreg.predict(x_test)
        predicted_classes = logreg.predict(x_test)
        accuracy = accuracy_score(y_test[:,i],predicted_classes)
        f1 = f1_score(y_test[:,i],predicted_classes)
        print(logreg.coef_)
        print("Accuracy: "+str(accuracy))
        print("F1 Score: "+str(f1))
        cm = confusion_matrix(y_test[:,i],predicted_classes)
        conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
        plt.figure(figsize = (8,5))
        sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")