import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
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
    f = open("IN499_Unit3.txt", "a")
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
data = pd.read_csv('', names=col_names)

########### EXPLORE THE DATA SET #######################

#Print the top 10 rows


#Print the shape


#Print the description of the data



############ FIX MISSING DATA #######################
#Replace NaN with 0 for Installers_retained_for_30_days



########### LINEAR REGRESSION #######################
#Create a linear regression model


#Set feature_cols to Installers column data only


#Set X to feature_cols


#Set y to installer retained for 30 days


#Print the shape of X


#Print the description of X


#Print the shape of y


#Print the description of X


#Fit the linear regression model with X and y


#Get predictions for retained for 30 days with 1 install


#Get predictions for retained for 30 days with 2 installs


#Get predictions for retained for 30 days with 4 installs


#Print the intercept


#Print the coefficient


########### LOGISTIC REGRESSION #######################
#Get logistic regression model


#Fit X and y to logistic regression model


#Predict classes using X


#Print the predictions using X



#Get the predicted probabilities of class 1


#Print the probabilities using X


#Predict the probablity of maintaining
# number of users for 30 days (0, 1, 2 users)

#Get probability for retained users for 30 days with 1 install


#Get probability for retained users for 30 days with 2 installs


#Get probability for retained users for 30 days with 4 installs



#################################### INSTALL_30 ##########################################
#Add a new column for installers retained for 30 days. Call it Install_30.
#  If greater than 0, put 1, if 0, put 0


#Print the top 10 rows of the new data set


#Print the shape of the new data set


#Print the description of the new data set


########### LOGISTIC REGRESSION #######################

#Perform logistic regression using Install_30 column for y
# X = Installers column


#Fit X and y for logistic regression


#Print the top 10 rows of X


#Print the shape rows of X


#Print the description of X


#Print the top 10 rows of y


#Print the shape of y


#Print the description of y


#Predict on X and capture the result to assorted_pred_class


#Print the predictions using assorted _pred_class (X predictions)


#Get the predicted probabilities of class 1 and save to assorted_pred_prob


#Print the probabilities using assorted_pred_prob


#Predict the probability of maintaining
# a user for 30 days (0, 1)

#Get probability for retained users for 30 days with 1 install


#Get probability for retained users for 30 days with 2 installs


#Get probability for retained users for 30 days with 4 installs


#Print the intercept


#Print the coefficient

