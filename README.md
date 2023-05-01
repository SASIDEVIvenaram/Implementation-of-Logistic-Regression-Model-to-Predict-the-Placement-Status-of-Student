# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1. Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SASIDEVI V
RegisterNumber: 212222230136
*/
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()


data1=data.copy()
data1=data1.drop(["sl_no",'salary'],axis=1)
print("Salary data:")
data1.head()

print("Null Data:")
data1.isnull().sum()

print("Duplicate Data:")
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1['gender']=le.fit_transform(data1['gender'])
data1['ssc_b']=le.fit_transform(data1['ssc_b'])
data1['hsc_b']=le.fit_transform(data1['hsc_b'])
data1['hsc_s']=le.fit_transform(data1['hsc_s'])
data1['degree_t']=le.fit_transform(data1['degree_t'])
data1['workex']=le.fit_transform(data1['workex'])
data1['specialisation']=le.fit_transform(data1['specialisation'])
data['status']=le.fit_transform(data1['status'])
data1

x=data1.iloc[:,:-1]
print("x Data:")
x

y=data["status"]
print("y Data:")
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear')
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
print("Confusion array:")
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print("Classification Report:")
print(classification_report1)
```

## Output:
### data.head()
![ML41](https://user-images.githubusercontent.com/118707332/235418739-cc35229d-4f31-46ec-9f4a-805a1a0c4220.png)

![ML42](https://user-images.githubusercontent.com/118707332/235418746-64c1f4dd-10e3-448b-b613-52b331fdad5c.png)

![ML43](https://user-images.githubusercontent.com/118707332/235418762-d0531905-4ba7-4ebf-8fc0-c41a63ad1393.png)

![ML44](https://user-images.githubusercontent.com/118707332/235418772-b1cf0cb0-0b5c-4565-8434-ab376b0f21af.png)
### Data after Encoding:
![ML45](https://user-images.githubusercontent.com/118707332/235418780-e6a7bb30-cb96-434f-9ffa-44f3b3ff5f64.png)
### X data:
![ML46](https://user-images.githubusercontent.com/118707332/235418787-17c0854d-ff36-4d8b-a9e6-1d80e1140aea.png)
### Y data:
![ML47](https://user-images.githubusercontent.com/118707332/235418793-696f47bc-880d-46e0-be1a-8354233e4526.png)
### Predicted Values:
![ML48](https://user-images.githubusercontent.com/118707332/235418804-19baea8b-a9ee-44f3-a124-5aff965569bd.png)
### Confusion array:
![ML49](https://user-images.githubusercontent.com/118707332/235418811-92f2f2bc-d0fe-4203-b0eb-ed635971a5ae.png)
### Classsification report:
![ML410](https://user-images.githubusercontent.com/118707332/235418836-5cb8e447-7016-45e5-81d6-71788bd12037.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
