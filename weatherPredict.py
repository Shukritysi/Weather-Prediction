import pandas as pd
import matplotlib as plt
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression

#Reading Dataset from the file "data_set.csv"
data_set=pd.read_csv("data_set.csv")

x=data_set.iloc[:,:-1].values
y=data_set.iloc[:,-1].values

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
#print(data_set.shape)
#print(data_set.head(20))
#print(data_set.describe())
#data_set.hist()
#plt.show()

logistic=LogisticRegression()
logistic.fit(x_train,y_train)
predict=logistic.predict(x_test)

print()
print("Accuracy: ",accuracy_score(y_test,predict))
print()
t=int(input("Total no. of times you want to predict: "))
for _ in range(t):
    print("Enter your own inputs")
    print()
    print("TempAvgF , DewPointAvgF, HumidityAvgPercent , SeaLevelPressureAvgInches",end="")
    print("VisibiltyAvgMiles , WindAvgMPH , WindGustMPH :")
    l=list(map(float,input().split()))
    print(logistic.predict([l]))