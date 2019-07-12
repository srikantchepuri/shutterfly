
#Exploration and understanding of the data sets

'''

order.csv contains the columns such as [custno],[ordno],[orderdate],[prodcat2],[prodcat1],[revenue]
online.csv contains the columns such as   [session],[visitor],[dt],[custno],[category],[event1],[event2]
One of the common common columns is the custno. With this we can try to corelate the onine data 
for the customer and lead it to the order he/she may make.  

'''

#Feature engineering

'''
Upon review of the data, for simplicity, I have decided to consider online data for all the customers that made orders. 
The assumption here is that the user's online data for the same day is going to determine the prodcat1 he/she may order in that perticular month.

I have used the below query to join both the data sources to get final data. 

select distinct ol.session,ol.visitor,ol.custno,ol.dt,isNull(ol.event1,0)event1,
ol.event2,ol.category,od.ordno,od.orderdate,od.prodcat1  
from online_tmp ol  left join order_tmp od on ol.custno=od.custno 
and convert(date,od.orderdate)=convert(date,ol.dt) 
where od.prodcat1 is not null
order by ol.custno,ol.dt

'''


#Feature selection

'''
The independent variables are : category,event1,event2,dt
The dependent variable is : prodcat1

All the other columns such as visitor,session,prodcat2,revenue are ignored as there is no apparent corelation from them to the dicision.

For the sake of simplicity I had considered the following assumptions:
I considered the online data for each customer mapped it with the order he/she made in the corresponding month.  

'''


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset=pd.read_csv("OnlineOrders.csv")
dataset['dt'] = pd.to_datetime(dataset['dt']).values.astype(np.int64) // 10**6

X = dataset.iloc[:,[3,4,5,6]].values
y = dataset.iloc[:, [9]].values




#Taking care of missing data
'''I have assumed zero - no event for those records that have null as event1'''

#Encoding Categorical data
'''The categorical data is already encoded in the files, hence not needed.''' 

# Model design and sampling
#Spliting the data set into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 3)



#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Model generation


#I am considering random forest classification to the training data

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


#Model evaluation
#Predicting the accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

'''

I have tried to work with independent variables such as event1,even2,dt (online date),category. 
I could not get a combination of independent variables that could predict the values with better accuracy. 
I have tried to take few different combinations to get better accuracy, but I always ended up with accuracy ranging from 20% - 30%.

Hence, I concluded that there can be different reasons for the same:
    Data may not be balanced.
    There was no accurate corelation between online data and order data.
    
'''
