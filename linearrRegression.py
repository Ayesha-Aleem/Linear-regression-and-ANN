from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
df = pd.read_csv('train.csv')
# print(df)
data=df.drop(columns=['Id'])




train_data=data.select_dtypes(include=[np.number]).dropna()
print(train_data.shape)
print(train_data['SalePrice'])



train_x=data.iloc[:,0:35]
train_y=train_data.iloc[:,36]
print(train_y)
Error=[]
X_train, X_test, y_train, y_test = train_test_split(train_data, train_y, test_size=0.3, random_state=0)
y_pred = LinearRegression().fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

Error.append((y_test != y_pred).sum()/len(y_test)*100)

y_pred2=LogisticRegression(random_state=0).fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred2).sum()))
Error.append((y_test != y_pred2).sum()/len(y_test)*100)

y_pred3=svm.SVC().fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred3).sum()))
Error.append((y_test != y_pred3).sum()/len(y_test)*100)

print(Error)

y_pred4= RandomForestClassifier().fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred4).sum()))
Error.append((y_test != y_pred4).sum()/len(y_test)*100)

c = ["red", "green", "orange", "black"]
name=["LinearRegression","LogisticRegression","svm","RandomForestTree"]
ax.bar(name,Error,color=c)
plt.show()




