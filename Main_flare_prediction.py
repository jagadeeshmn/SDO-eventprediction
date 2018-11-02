from Flare_prediction import Load_data
from LogisticRegression import Perform_Logistic_Regression
from datetime import datetime

start_time = datetime(2013,2,16,1,00,00)
end_time = datetime(2013,2,16,1,1,00)

N = 10

X,Y = Load_data(start_time,end_time,N)
print(Y)
Perform_Logistic_Regression(X,Y)