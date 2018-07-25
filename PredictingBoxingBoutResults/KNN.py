import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.linear_model import LinearRegression
#0.8103 accuracy when only adjustment is removal of irrelevant columns.



df = pd.read_csv('bouts_out_new.csv')


#df.drop(['judge1_A'],['judge1_B'],['judge2_A'],['judge2_B'],['judge3_A'],['judge3_B'],['decision'], 1, inplace=True)


df.fillna(value=-9999, inplace=True) #Ensure that NaN entries are treated as outliers. 

#Give integer values to bout results.
df.replace('win_A', 1, inplace=True) 
df.replace('win_B', -1, inplace=True)
df.replace('draw', 0, inplace=True)

#In this case, the fighters' physical measurements relative to his/her opponents' measurements are clearly a much more accorate indicator of an advatange.  
#
df['height_diff_A'] = df['height_A']-df['height_B']
df['height_diff_B'] = df['height_B']-df['height_A']
df['reach_diff_A'] = df['reach_A']-df['reach_B']
df['reach_diff_B'] = df['reach_B']-df['reach_A']
df['weight_diff_A'] = df['weight_A']-df['weight_B']
df['weight_diff_B'] = df['weight_B']-df['weight_A']

#Create a new dataframe with only the relevant columsns. 
df = df[['age_A', 'age_B', 'height_diff_A', 'height_diff_B', 'reach_diff_A', 'reach_diff_B', 'weight_diff_A', 'weight_diff_B', 'won_A', 'won_B', 'lost_A', 'lost_B', 'drawn_A', 'drawn_B', 'kos_A', 'kos_B' , 'result']]

print(df)

X = np.array(df.drop(['result'], 1)) #The features are the X value. 
y = np.array(df['result']) #The result is the y variable.


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#The K Neighbors method will be used as the classifer for this predictive model. 
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)
