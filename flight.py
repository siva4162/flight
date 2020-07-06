import pandas as pd 
import numpy as np
import pickle
import seaborn  as sns
from sklearn.model_selection import train_test_split,RandomizedSearchCV 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
train_data=pd.read_excel("Data_Train.xlsx")
pd.set_option('display.max_columns',None)
#print(train_data.head(2))
#print(train_data.info())
train_data.dropna(inplace=True)
#print(train_data.isnull().sum())
train_data['Journey_data']=pd.to_datetime(train_data.Date_of_Journey,format= '%d/%m/%Y').dt.day
train_data['Journey_month']=pd.to_datetime(train_data.Date_of_Journey,format= '%d/%m/%Y').dt.month
#print(train_data['Journey_month'])
train_data.drop(['Date_of_Journey'],axis=1,inplace=True)
train_data['Dep_hour']=pd.to_datetime(train_data['Dep_Time']).dt.hour
train_data['Dep_min']=pd.to_datetime(train_data['Dep_Time']).dt.minute
train_data["Arrival_hour"]=pd.to_datetime(train_data['Arrival_Time']).dt.hour
train_data["Arrival_min"]=pd.to_datetime(train_data['Arrival_Time']).dt.minute
train_data.drop(['Dep_Time'],axis=1,inplace=True)
train_data.drop(['Arrival_Time'],axis=1,inplace=True)
#print(train_data.head(2))
duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))
train_data["Duration_hours"] = duration_hours
train_data["Duration_mins"] = duration_mins    
train_data.drop(['Duration'],axis=1,inplace=True)    
#print(train_data["Airline"].value_counts())
Airline = train_data[["Airline"]]
Airline= pd.get_dummies(Airline,drop_first=True)
#
#print(Airline.head(2))
#print(train_data["Source"].value_counts())
Source = train_data[["Source"]]

Source = pd.get_dummies(Source, drop_first= True)

#print(Source.head())
Destination = train_data[["Destination"]]

Destination = pd.get_dummies(Destination, drop_first = True)

# print(Destination.head())
train_data.drop(['Route','Additional_Info'],axis=1,inplace=True)
train_data.replace({"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4},inplace=True)
#print(train_data['Total_Sxtops'])
data_train = pd.concat([train_data, Airline, Source, Destination], axis = 1)
data_train.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
test_data = pd.read_excel("Test_set.xlsx")
print(data_train.shape)
# Preprocessing

#print("Test data Info")
#print("-"*75)
#print(test_data.info())

#print()
#print()

#print("Null values :")
#print("-"*75)
test_data.dropna(inplace = True)
#print(test_data.isnull().sum())

# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

# Duration
duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration

# Adding Duration column to test set
test_data["Duration_hours"] = duration_hours
test_data["Duration_mins"] = duration_mins
test_data.drop(["Duration"], axis = 1, inplace = True)


# Categorical data

#print("Airline")
#print("-"*75)
#print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(test_data["Airline"], drop_first= True)

#print()

#print("Source")
#print("-"*75)
#print(test_data["Source"].value_counts())
Source = pd.get_dummies(test_data["Source"], drop_first= True)

#print()

#print("Destination")
#print("-"*75)
#print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(test_data["Destination"], drop_first = True)

# Additional_Info contains almost 80% no_info
# Route and Total_Stops are related to each other
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
data_test = pd.concat([test_data, Airline, Source, Destination], axis = 1)

data_test.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)

#print()
#print()

#print("Shape of test data : ", data_test.shape)
X = data_train.drop(['Price'],axis=1)
#print(X.columns)
y=data_train.iloc[:,1]
#print(y.head())
ex=ExtraTreesRegressor()
ex.fit(X,y)
print(ex.feature_importances_)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
reg_rf=RandomForestRegressor()
reg_rf.fit(X_train,y_train)
print(reg_rf.score(X_train,y_train))
y_pred= reg_rf.predict(X_test)
print(y_pred)
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
print(rf_random.best_params_)
print(rf_random.score(X_train,y_train))
file=open('flight_rf.pkl','wb')
pickle.dump(rf_random,file)
model=open('flight_rf.pkl','rb')
forest=pickle.load(model)
y_prediction= forest.predict(X_test)
