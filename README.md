# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Load Data** – Import dataset using pandas and explore features & target.
2. **Preprocess** – Handle missing values, encode categorical variables, and split into features (X) & target (y).
3. **Train Model** – Split data and train a `DecisionTreeClassifier` from `sklearn.tree`.
4. **Evaluate & Visualize** – Predict on test data, check accuracy/confusion matrix/precision/recall.



## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Pranav Bhargav M
RegisterNumber:  212224040239
```

```
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()

le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level", "last_evaluation", "number_project", "average_montly_hours", 
          "time_spend_company", "Work_accident", "promotion_last_5years", "salary"]]
x.head() #no departments and no left
y=data["left"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['stayed','left'],filled=True)

```


## Output:


<img width="651" height="482" alt="image" src="https://github.com/user-attachments/assets/05ba81c9-3b82-4d88-9547-6158232d3542" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
