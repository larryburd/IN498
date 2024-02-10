import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import sys

# Ignoring warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


# Set the writeFunction boolean
# True = write to concole
# False = write to file for assignment turnin
PRINT = False

# Open a file handle for assignment results.
if PRINT == False:
    f = open("IN498_Unit4_Burden.txt", "a")
###############################################
#
# PURPOSE: Write to console or file based on
#  the writeFunction variable
#      True = write to console
#      False = write to file for assignment turnin
#
# INPUT: Message to write (message1)
#   Optional: message2
#
# OUTPUT: None
#
###############################################
def writeFunction(message1, *message2):

    # Print to console
    if PRINT:
        print(message1)
        print(message2)
        print()
    # Print to file for assignment
    else:
        f.write(str(message1))
        f.write(str(message2))
        f.write("\n\n")


# Widen the column display
pd.set_option('max_colwidth', 500)

# Read data into a DataFrame using these columns
# "Date","Package_Name","Country","Store_Listing_Visitors",
# "Installers","Visitor-to-Installer_conversion_rate",
# "Installers_retained_for_1_day","Installer-to-1_day_retention_rate",
# "Installers_retained_for_7_days","Installer-to-7_days_retention_rate",
# "Installers_retained_for_15_days","Installer-to-15_days_retention_rate",
# "Installers_retained_for_30_days","Installer-to-30_days_retention_rate"
col_names = ["Date", "Package_Name", "Country", "Store_Listing_Visitors",
             "Installers", "Visitor-to-Installer_conversion_rate", "Installers_retained_for_1_day",
             "Installer-to-1_day_retention_rate", "Installers_retained_for_7_days",
             "Installer-to-7_days_retention_rate", "Installers_retained_for_15_days",
             "Installer-to-15_days_retention_rate", "Installers_retained_for_30_days",
             "Installer-to-30_days_retention_rate"]
data = pd.read_csv('final_retentions_parsed.csv', names=col_names)

############ FIX MISSING DATA #######################
# Replace NaN with 0 for Installers_retained_for_30_days
data[np.isnan(data.Installers_retained_for_30_days)] = 0

#################################### INSTALL_30 ##########################################
# Add a new column for installers retained for 30 days
#  If greater than 0, put 1, if 0, put 0
data['Install_30'] = np.where(data['Installers_retained_for_30_days'] > 0, 1, 0)

# Create X using Installers and y using Install_30 columns
X = pd.DataFrame({'Installers': data['Installers']})
y = pd.DataFrame({'Install_30': data['Install_30']})

# print('data shape: ', data.shape)
# print('shape: ', data['Installers'].shape)
# print('X Shape: ', X.shape)
# print('y Shape: ', y.shape)

# Train/test split 80% train, 20% test
# Save into these variables: X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

########### DECISION TREE #######################
# Build a decision tree model
dt = DecisionTreeClassifier()

# Fit the tree using X_train and y_train
dt.fit(X_train, y_train)

# Print the accuracy of the tree using X_train and y_train
writeFunction('Decision Tree Accuracy with Training Data: ', dt.score(X_train, y_train))

writeFunction('Decision Tree Accuracy with Test Data: ', dt.score(X_test, y_test))

########### RANDOM FOREST #######################
#Build a random forest tree model
# n_estimators = 1000
rf = RandomForestClassifier(n_estimators=1000)

#Fit the random tree model with X_train and y_train
rf.fit(X_train, y_train)

#Predict using X_test and the random forest model and store in y_pred
y_pred = rf.predict(X_test)

#Print the random forest results using y_pred
writeFunction('Random Forest Prediction: ', y_pred)

#Print the accuracy for the random forest model using y_test and y_pred
writeFunction('Random Forest Accuracy: ', metrics.accuracy_score(y_test, y_pred))

#Predict retention over 30 days for number of installs
#  Use 1,2,4
writeFunction('Random Forest Prediction of Keeping 1 User 30 Days with 1 Install: ',
              rf.predict([[1]]))
writeFunction('Random Forest Prediction of Keeping 1 User 30 Days with 2 Install: ',
              rf.predict([[2]]))
writeFunction('Random Forest Prediction of Keeping 1 User 30 Days with 4 Install: ',
              rf.predict([[4]]))

# Get the absolute errors for the random forest model
# Use y_pred and y_test
errors = abs(y_pred - y_test['Install_30'])

# Print the absolute errors for the random forest model
# Use y_pred and y_test
writeFunction('Absolute Errors for Random Forest Model', errors)

# Print out the mean absolute error
writeFunction('Mean Absolute Error', round(np.mean(errors), 2))

# Get the predictions using X_test and save to rf_probs
rf_probs = rf.predict_proba(X_test)[:, 1]

# Print the predictions for Random Forest using rf_probs
writeFunction('Random Forest Probabilities X_test', rf_probs)

# Compute Area Under the Receiver Operating Characteristic Curve
# Get the ROC AUC score for the random forest model for y_test and rf_probs
roc_value = roc_auc_score(y_test, rf_probs)

# Print the ROC AUC for the random forest model
writeFunction('ROC AUC for Random Forest', roc_value)

# Get the mean absolute percentage error (MAPE) using y_test
mape = 100 * (errors / y_test['Install_30'])

# Print the mean absolute percentage error (MAPE) using y_test
writeFunction('Random Forest MAPE', mape)

# Get the accuracy for the random forest model using mape
accuracy = 100 - np.mean(mape)

# Print the accuracy for the random forest model
writeFunction('Random Forest Accuracy Percentage Using Error Percentage', accuracy)

# Close the file handle
if PRINT == False:
    f.close()


