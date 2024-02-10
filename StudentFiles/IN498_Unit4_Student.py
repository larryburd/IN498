import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import sys

#Ignoring warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


#Set the writeFunction boolean
#True = write to concole
#False = write to file for assignment turnin
PRINT = True

#Open a file handle for assignment results.
if PRINT == False:
    f = open("IN499_Unit4.txt", "a")
###############################################
##
##PURPOSE: Write to console or file based on
##  the writeFunction variable
##      True = write to concole
##      False = write to file for assignment turnin
##
##INPUT: Message to write (message1)
##   Optional: message2
##
##OUTPUT: None
##
###############################################
def writeFunction(message1, *message2):

    #Print to console
    if PRINT:
        print(message1)
        print(message2)
        print()
    #Print to file for assignment
    else:
        f.write(str(message1))
        f.write(str(message2))
        f.write("\n\n")


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

############ FIX MISSING DATA #######################
#Replace NaN with 0 for Installers_retained_for_30_days


#################################### INSTALL_30 ##########################################
#Add a new column for installers retained for 30 days
#  If greater than 0, put 1, if 0, put 0


#Create X using Installers and y using Install_30 columns


#Train/test split 80% train, 20% test
#Save into these variables: X_train, X_test, y_train, y_test


########### DECISION TREE #######################
#Build a decision tree model


#Fit the tree using X_train and y_train


#Print the accuracy of the tree using X_train and y_train



########### RANDOM FOREST #######################
#Build a random forest tree model
# n_estimators = 1000


#Fit the random tree model with X_train and y_train


#Predict using X_test and the random forest model and store in y_pred


#Print the random forest results using y_pred


#Print the accuracy for the random forest model using y_test and y_pred


#Predict retention over 30 days for number of installs
#  Use 1,2,4


#Get the absolute errors for the random forest model
# Use y_pred and y_test


#Print the absolute errors for the random forest model
# Use y_pred and y_test


# Print out the mean absolute error


#Get the predicitons using X_test and save to rf_probs


#Print the predictions for Random Forest using rf_probs


#Compute Area Under the Receiver Operating Characteristic Curve
#Get the ROC AUC score for the random forest model for y_test and rf_probs


#Print the ROC AUC for the random forest model


#Get the mean absolute percentage error (MAPE) using y_test


#Print the mean absolute percentage error (MAPE) using y_test


#Get the accuracy for the random forest model using mape


#Print the accuracy for the random forest model


#Close the file handle
if PRINT == False:
    f.close()


