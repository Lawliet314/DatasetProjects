import math
import numpy as np
import pandas as pd
import csv
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#Just for the sake of implementing linear regression to this dataset, I will atempt to build a predictive model of the resulting score using the features given.
#With linear regression, I thought the approriate prediction would be the competitiveness of the fight (how close the bout was) rather than a binary output such as win/loss.
#Bouts ending in knockouts will be excluded from the dataset for this model. 
#The judges' scores will be used to represent competitiveness of the fight quantitatively. A zero represents a draw, a positive score is in favor of fighter A, and a negative score is in favor of fighter B.

df = pd.read_csv('bouts_out_new.csv')
#df.drop(columns=['judge1_A','judge1_B','judge2_A','judge2_B','judge3_A','judge3_B'])
df.fillna(value=0, inplace=True)
df.dropna()
#df.drop(columns=['stance_A','stance_B', 'result', 'decision'])
df['score'] = (df['judge1_A']+df['judge2_A']+df['judge3_A'])-(df['judge1_B']+df['judge2_B']+df['judge3_B'])
#Positve score is in favor of A, negative score is in favor of B. Zero is a draw.

df = df[['age_A' ,'age_B', 'height_A', 'height_B', 'reach_A', 'reach_B', 'weight_A', 'weight_B', 'won_A', 'won_B', 'lost_A', 'lost_B', 'drawn_A', 'drawn_B', 'score']] 
pred_col = 'score'
pred_out = int(math.ceil(0.01*len(df)))
df['score'] = df[pred_col].shift(-pred_out)
df.dropna(inplace=True)
#df['height_diff'] = (df['height_A')-df('height_B'))
#df['weight_diff'] = (df['weight_A')-df('weight_B'))

#df['loss_pct_A'] = -(df['lost_A']/(df['won_A']+df['lost_A']+df['drawn_A']))
#df['loss_pct_B'] = (df['lost_B']/(df['won_B']+df['lost_B']+df['drawn_B']))                    
#df['ko_pct_A'] = (df['kos_A']/(df['won_A']+df['lost_A']+df['drawn_A']))                    
#df['ko_pct_B'] = -(df['kos_B']/(df['won_B']+df['lost_B']+df['drawn_B']))                       

X = np.array(df.drop(['score'], 1))
y = np.array(df['score'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf = svm.SVR()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

