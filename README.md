# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

#Load Dataset
data=pd.read_csv('CarPrice_Assignment.csv')
print(data.head())
print(data.info())

#Data Preprocessing
#Dropping unnecessary columns and handling categorial variables
data=data.drop(['CarName','car_ID'],axis=1)
data=pd.get_dummies(data,drop_first=True)


#Splitting the data into features and target variable
x=data.drop('price',axis=1)
y=data['price']

#Standardizing the data
scaler=StandardScaler()
x=scaler.fit_transform(x)
y=scaler.fit_transform(np.array(y).reshape(-1,1))

#Splitting the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#Creating the SGD Regressor model
sgd_model=SGDRegressor(max_iter=1000,tol=1e-3)

#Fitting the model on the training data
sgd_model.fit(x_train,y_train)

#Making predictions
y_pred=sgd_model.predict(x_test)

#Evaluating model performance
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)


#Print Evaluation Metrics
print("Name:Ponsriram P")
print("Reg no:212225240105")
print("Mean Squared Error:",mse)
print("Mean Absolute Error:",mae)
print("R-squared Score:",r2)


#Print model coefficients
print("\nModel Coefficients")
print("Coefficients:",sgd_model.coef_)
print("Intercept:",sgd_model.intercept_)


#Visualizing actual vs predicted prices
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color='red')
plt.show()

print(y.shape)


```

## Output:
<img width="849" height="624" alt="image" src="https://github.com/user-attachments/assets/82bee41b-49b8-4b33-939b-0aeffb045abb" />
<img width="536" height="721" alt="image" src="https://github.com/user-attachments/assets/7cbe309d-f067-4438-b6af-7e18fbc9e678" />
<img width="861" height="340" alt="image" src="https://github.com/user-attachments/assets/fac84aa9-0a39-4b44-955c-82183aa1736c" />
<img width="1257" height="722" alt="image" src="https://github.com/user-attachments/assets/42e02fd1-03c9-4c22-ad73-faf88ff4a74c" />





## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
