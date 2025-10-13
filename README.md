# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Dataset: Import the dataset using pandas and understand its structure (features, target).

Preprocess Data:

Handle missing values

Encode categorical variables (Label Encoding / One-Hot Encoding)

Split the dataset into features (X) and target (y)

Split Dataset: Divide data into training set and test set (e.g., 80:20 ratio).

Build Decision Tree Classifier:

Import DecisionTreeClassifier from sklearn.tree

Train the model on the training set

Predict & Evaluate:

Make predictions on the test set

Evaluate using accuracy, confusion matrix, precision, recall

Visualize Decision Tree (Optional):

Use plot_tree or export_graphviz to visualize decision paths
2. 
3. 
4. 

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
