#--------------------------------------------------#




#1) IMPORT LIBRARIES

#Computation and Structuring:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#Modeling:

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Testing:

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#--------------------------------------------------#

#2) DATA IMPORT AND PRE-PROCESSING

#import full data set
# data=pd.read_csv("D:\\dataset\\Crime\\Merged_data_crime.csv")

df = pd.read_csv('D:\\ved\\dataset\\Crime\\Merged_data_crime.csv') 


# Lets understand how's our data
# columns and datatypes
print(df.dtypes)

# to get more idea about na values
print(df.isna().sum())

# once we get idea about na values we can drop it.it would give results
df = df.dropna(axis=0)



# to know min,max,mean about columns

print(df.describe())



# This is bar plot which tells about how occurrencehour is related to crime
# on x axis--> grouped hours
# on Y-axis-->crime count

hour_freq=pd.DataFrame(df.groupby(['occurrencehour']).size())
count_column = list(hour_freq.iloc[:, 0])
second_column=[str(i-2)+"-"+str(i) for i in range(2,24,3)]
count_frq=[]
for i in range(0,22,3):
    temp=count_column[i]+count_column[i+1]+count_column[i+2]
    count_frq.append(temp)




import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([1,1,1,1])
day_time = second_column
commied_crime = count_frq
ax.bar(day_time,commied_crime)
plt.show()




# This is bar plot which tells about how occurrencemonth is related to crime
# on x axis--> months
# on Y-axis-->crime count

month_freq=pd.DataFrame(df.groupby(['occurrencemonth']).size())
month_count = list(month_freq.iloc[:, 0])
mon_second_column=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sept","Oct","Nov","Dec"]



import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([1,1,1,1])
# day_time = second_column
# commied_crime = count_frq
ax.bar(mon_second_column,month_count)
plt.show()


# This is bar plot which tells about how occurrenceyear is related to crime
# on x axis--> year
# on Y-axis-->crime count

# we can use linear regression to predict crime count in current year

year_freq=pd.DataFrame(df.groupby(['occurrenceyear']).size())
year_count = list(year_freq.iloc[:, 0])
year_second_column=[str(i) for i in range(2000,2020)]



import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([2,2,2,2])
# day_time = second_column
# commied_crime = count_frq
ax.bar(year_second_column,year_count)
# ax.bar(year_count,year_second_column)
plt.show()







#list of relevant columns for model
col_list = ['occurrenceyear',	'occurrencemonth','occurrenceday','occurrencedayofyear','occurrencedayofweek','occurrencehour','MCI',	'Division',	'Hood_ID','premisetype']

#dataframe created from list of relevant columns

df2 = df[col_list]
df2 = df2[df2['occurrenceyear'] > 2013] #drop "stale" crimes, where occurence is before 2014. Since data set is filtered based on reported date, we're ignoring these old crimes.

#Factorize dependent variable column:

crime_var = pd.factorize(df2['MCI']) #codes the list of crimes to a int64 variable
df2['MCI'] = crime_var[0]
definition_list_MCI = crime_var[1] #create an index reference so we know which crimes are coded to which factors

#factorize independent variables:

#factorize premisetype:

premise_var = pd.factorize(df2['premisetype'])
df2['premisetype'] = premise_var[0]
definition_list_premise = premise_var[1] 

#factorize occurenceyear:

year_var = pd.factorize(df2['occurrenceyear'])
df2['occurrenceyear'] = year_var[0]
definition_list_year = year_var[1] 

#factorize occurencemonth:

month_var = pd.factorize(df2['occurrencemonth'])
df2['occurrencemonth'] = month_var[0]
definition_list_month = month_var[1] 

#factorize occurenceday:

day_var = pd.factorize(df2['occurrenceday'])
df2['occurenceday'] = day_var[0]
definition_list_day = day_var[1] 

#factorize occurencedayofweek:

dayweek_var = pd.factorize(df2['occurrencedayofweek'])
df2['occurrencedayofweek'] = dayweek_var[0]
definition_list_day = dayweek_var[1] 

#factorize division:

division_var = pd.factorize(df2['Division'])
df2['Division'] = division_var[0]
definition_list_division = division_var[1] 

#factorize HOOD_ID:

hood_var = pd.factorize(df2['Hood_ID'])
df2['Hood_ID'] = hood_var[0]
definition_list_hood = hood_var[1] 

#factorize occurencehour:

hour_var = pd.factorize(df2['occurrencehour'])
df2['occurrencehour'] = hour_var[0]
definition_list_hour = hour_var[1] 

#factorize occurencedayofyear:

dayyear_var = pd.factorize(df2['occurrencedayofyear'])
df2['occurrencedayofyear'] = dayyear_var[0]
definition_list_dayyear = dayyear_var[1] 

#set X and Y:

X = df2.drop(['MCI'],axis=1).values #sets x and converts to an array
# print(X.head())

y = df2['MCI'].values #sets y and converts to an array

#split the data into train and test sets for numeric encoded dataset:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)

#need to OneHotEncode all the X variables for input into the classification model:

binary_encoder = OneHotEncoder(sparse=False)
encoded_X = binary_encoder.fit_transform(X)

X_train_OH, X_test_OH, y_train_OH, y_test_OH = train_test_split(encoded_X, y, test_size = 0.25, random_state = 21)


#--------------------------------------------------#

#3) MODELING AND TESTING:

#Numeric Encoded Model w/ SKLEARN:

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test) # Predicting the Test set results

print(accuracy_score(y_test, y_pred)) #accuracy at 0.63
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test,y_pred, target_names=definition_list_MCI)) 

#theft over is pulling down results. Pretty good on Assault (largest sample size) and break and enter 


#One Hot Encoded Model w/ SKLEARN:

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
classifier.fit(X_train_OH, y_train_OH)
y_pred_OH = classifier.predict(X_test_OH) # Predicting the Test set results

print(accuracy_score(y_test_OH, y_pred_OH)) #modest improvement to 0.648
print(confusion_matrix(y_test_OH, y_pred_OH)) 
print(classification_report(y_test_OH,y_pred_OH, target_names=definition_list_MCI)) #modest improvement

#Balanced Class Weight doesn't make a big difference for results:

classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42, class_weight='balanced')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test) 
print(accuracy_score(y_test, y_pred)) #accuracy at 0.63
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test,y_pred, target_names=definition_list_MCI)) 

#--------------------------------------------------#

#gradientboost performs poorly relative to randomforest

grad_class = GradientBoostingClassifier(learning_rate=0.1,n_estimators = 10, random_state = 42)
grad_class.fit(X_train_OH, y_train_OH)
y_pred_OH = grad_class.predict(X_test_OH) # Predicting the Test set results

print(accuracy_score(y_test_OH, y_pred_OH)) #modest improvement to 0.648
print(confusion_matrix(y_test_OH, y_pred_OH)) 
print(classification_report(y_test_OH,y_pred_OH, target_names=definition_list_MCI)) 



# We have Homicides dataset too ,so we can take that dataset also.
# we can take mode of all model result ,It probably give the better accuracy .
# In short it would be great if we calculate results by considering all models.
# Thanking you
