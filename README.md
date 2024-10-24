# Ex-8: Implementation-of-SVM-For-Spam-Mail-Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Split the data into training and testing sets.
5. Convert the text data into a numerical representation using CountVectorizer.
6. Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7. Finally, evaluate the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRAKASH R
RegisterNumber: 212222240074
*/
```
```Python
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
```
## Output:
![image](https://github.com/RahulM2005R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/166299886/44799fa3-ece7-4462-be26-90ecb6fd4836)
![image](https://github.com/RahulM2005R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/166299886/66d76051-4607-4d8a-bb69-47877a51c4ef)
![image](https://github.com/RahulM2005R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/166299886/ab54471e-e31c-4eca-a501-e35f11d1f8da)
![image](https://github.com/RahulM2005R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/166299886/72308b35-ee91-4cbc-b8f8-ef14466fbaa1)
![image](https://github.com/RahulM2005R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/166299886/a830ff21-4db0-4dbd-a14a-d71fd74df8ec)
![image](https://github.com/RahulM2005R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/166299886/c8430281-3a88-4acd-a387-304ff1003b67)
![image](https://github.com/RahulM2005R/Implementation-of-SVM-For-Spam-Mail-Detection/assets/166299886/b94818ae-7a21-4cb7-9f65-9b5773a47f8c)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
