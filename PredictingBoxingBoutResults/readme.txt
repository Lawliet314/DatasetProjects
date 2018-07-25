The dataset was posted by user Alexander Slonsky on kaggle.com https://www.kaggle.com/slonsky/boxing-bouts
Using scikit-learn and pandas, I implemented a k-Nearest Neighbors algorithm to build a binary predictive model using the features included in the data set. 
Simply for the sake of implementation, I also built a  regression model using support vector regression to predict the closeness of the resuling score. The "closeness" of the score was first simply derived the difference in judges' scorecards but there was room for flexibility in handling the bouts ending in a knockout. 
