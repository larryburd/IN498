import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

#Ignoring warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



#Widen the column display
pd.set_option('max_colwidth',500)

#Read data into a DataFrame using these columns
##"Date","Package_Name","Country","Store_Listing_Visitors",
##"Installers","Visitor-to-Installer_conversion_rate",
##"Installers_retained_for_1_day","Installer-to-1_day_retention_rate",
##"Installers_retained_for_7_days","Installer-to-7_days_retention_rate",
##"Installers_retained_for_15_days","Installer-to-15_days_retention_rate",
##"Installers_retained_for_30_days","Installer-to-30_days_retention_rate"
col_names = ["Date","Package_Name","Country","Store_Listing_Visitors","Installers","Visitor-to-Installer_conversion_rate","Installers_retained_for_1_day","Installer-to-1_day_retention_rate","Installers_retained_for_7_days","Installer-to-7_days_retention_rate","Installers_retained_for_15_days","Installer-to-15_days_retention_rate","Installers_retained_for_30_days","Installer-to-30_days_retention_rate"]
data = pd.read_csv('final_retentions_parsed.csv', names=col_names)

########### EXPLORE THE ORIGINAL DATA SET #######################

#Plot the installer retention rates for 1 day using a bar chart


#Plot the installer retention rates for 7 days using a bar chart


#Plot the installer retention rates for 15 days using a bar chart


#Plot the installer retention rates for 30 days using a bar chart


#Plot the retention of users at 30 days per date using a line chart


#Add a month column by converting the Date column for month only
data['Month'] = pd.DatetimeIndex(data['Date']).month

#Plot the installs per month using a bar chart


########### EARLY SEABORN PLOTS #######################
#Seaborn simple regression plot of installers retained for 1 day


#Seaborn simple regression plot of installers retained for 7 day


#Seaborn simple regression plot of installers retained for 30 day


############ FIX MISSING DATA #######################
#Replace NaN with 0 for Installers_retained_for_30_days
data[np.isnan(data.Installers_retained_for_30_days)] = 0

############ SEABORN PLOT AFTER FIXING MISSING DATA #######################
#Seaborn pairplot for installers retained for 1, 7, 15, and 30 days



########### LOGISTIC REGRESSION #######################
#Get logistic regression model
logreg = LogisticRegression()

#Set feature_cols to Installers column data only
feature_cols = ['Installers']

#Set X to feature_cols
X = data[feature_cols]

#Set y to installer retained for 30 days
y = data.Installers_retained_for_30_days

#Fit X and y to logistic regression model
logreg.fit(X, y)

#Predict classes using X and save as a variable called assorted_pred_class



#Plot the predictions in a scatter plot using
#Scatter uses data.Installers and data.Installers_retained_for_30_days
#X label of Installer
#y label of predicted retained
#Plot using data.Installer and the prediction results (assorted_pred_class) on X


#Get the predicted probabilites of class 1
#Save as a variable called assorted_pred_prob



#Scatter plot the predicted probabilities
#Scatter uses data.Installers and data.Installers_retained_for_30_days
#X label of Installer
#y label of predicted probabilities
#Plot using data.Installer and the probability results (assorted_pred_prob) on X



#################################### INSTALL_30 ##########################################
#Add a new column for installers retained for 30 days
#Call the new column Install_30.
#  If greater than 0, put 1, if 0, put 0



########### LOGISTIC REGRESSION #######################

#Perform logistic regression using Install_30 column for y
# X = Installers column
logreg = LogisticRegression()
feature_cols = ['Installers']
X = data[feature_cols]
y = data.Install_30

#Fit X and y for logistic regression
logreg.fit(X, y)


#Predict on X and capture the result to assorted_pred_class



#Plot the class predictions
#Scatter uses data.Intsallers and data.Install_30
#Plot uses data.Installers and assorted_pred_class
#X label is Installer
#Y label is Predicted Retained (Install_30)


#Get the predicted probabilites of class 1 and save to assorted_pred_prob


#Plot the predicted probabilities
#Scatter uses data.Intsallers and data.Install_30
#Plot uses data.Installers and assorted_pred_prob
#X label is Installer
#Y label is Predicted Probabilities (Install_30)



########### DECISION TREE #######################
#Train/test split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Build a decision tree model
dt = DecisionTreeClassifier(random_state=1)

#Fit the tree using X_train and y_train
dt.fit(X_train, y_train)


#Plot the decision tree



########### RANDOM FOREST #######################
#Build a random forest tree model
# n_estimators = 1000
rf=RandomForestClassifier(n_estimators=1000)

#Fit the random tree model with X_train and y_train
rf.fit(X_train,y_train)

#Get the predicitons using X_test and save to rf_probs


#Plot the random forest predictions using rf_preds





