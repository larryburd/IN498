import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import sys

# Ignoring warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Widen the column display
pd.set_option('max_colwidth',500)

# Read data into a DataFrame using these columns
# "Date","Package_Name","Country","Store_Listing_Visitors",
# "Installers","Visitor-to-Installer_conversion_rate",
# "Installers_retained_for_1_day","Installer-to-1_day_retention_rate",
# "Installers_retained_for_7_days","Installer-to-7_days_retention_rate",
# "Installers_retained_for_15_days","Installer-to-15_days_retention_rate",
# "Installers_retained_for_30_days","Installer-to-30_days_retention_rate"
col_names = ["Date", "Package_Name", "Country", "Store_Listing_Visitors", "Installers",
             "Visitor-to-Installer_conversion_rate", "Installers_retained_for_1_day",
             "Installer-to-1_day_retention_rate", "Installers_retained_for_7_days",
             "Installer-to-7_days_retention_rate", "Installers_retained_for_15_days",
             "Installer-to-15_days_retention_rate", "Installers_retained_for_30_days",
             "Installer-to-30_days_retention_rate"]
data = pd.read_csv('final_retentions_parsed.csv', names=col_names)

########### EXPLORE THE DATA SET #######################
# Print the top 10 rows
print('### First 10 Rows ###')
print(data.head(10), '\n')

# Print the shape
print('### Data Shape ###')
print(data.shape, '\n')

# Print the description of the data
print('### Data Description ###')
print(data.describe(), '\n')

############ FIX MISSING DATA #######################
# Replace NaN with 0 for Installers_retained_for_30_days
data[np.isnan(data.Installers_retained_for_30_days)] = 0

########### LINEAR REGRESSION #######################
# Create a linear regression model
linreg = LinearRegression()

# Set feature_cols to Installers column data only
feature_cols = ['Installers']

# Set X to feature_cols
X = data[feature_cols]

# Set y to installer retained for 30 days
#y = data['Installers_retained_for_30_days']
y = data.Installers_retained_for_30_days

# Print the shape of X
print('### Shape of X ###')
print(X.shape, '\n')

# Print the description of X
print('### Description of X ###')
print(X.describe, '\n')

# Print the shape of y
print('### Shape of y ###')
print(y.shape, '\n')

# Print the description of y
print('### Description of y ###')
print(y.describe(), '\n')

# Fit the linear regression model with X and y
linreg.fit(X[:], y[:])

# Get predictions for retained for 30 days with 1 install
pred_30_days_1_install = linreg.predict([[1]])
print('### Prediction for number retained for 30 days with 1 install ###')
print(pred_30_days_1_install, '\n')

# Get predictions for retained for 30 days with 2 installs
pred_30_days_2_installs = linreg.predict([[2]])
print('### Prediction for number retained for 30 days with 2 installs ###')
print(pred_30_days_2_installs, '\n')

# Get predictions for retained for 30 days with 4 installs
pred_30_days_4_installs = linreg.predict([[4]])
print('### Prediction for number retained for 30 days with 3 installs ###')
print(pred_30_days_4_installs, '\n')

# Print the intercept
print('### Y Intercept ###')
print(linreg.intercept_, '\n')

# Print the coefficient
print('### Coefficient ###')
print(linreg.coef_, '\n')

########### LOGISTIC REGRESSION #######################
# Get logistic regression model
logreg = LogisticRegression()

# Fit X and y to logistic regression model
logreg.fit(X, y)

# Predict classes using X
predictions = logreg.predict(X)

# Print the predictions using X
print('### Logistic Regression Predictions ###')
print(predictions)


# Get the predicted probabilities of class 1
assorted_pred_prob = logreg.predict_proba(X)[:, 1]

# Print the probabilities using X
print('### Log Regression Probabilities ###')
print(assorted_pred_prob, '\n')

# Predict the probablity of maintaining
# number of users for 30 days (0, 1, 2 users)

# Get probability for retained users for 30 days with 1 install
print('### Prediction for number retained for 30 days with 1 install ###')
print(logreg.predict_proba([[1]]), '\n')

# Get probability for retained users for 30 days with 2 installs
print('### Prediction for number retained for 30 days with 2 installs ###')
print(logreg.predict_proba([[2]]), '\n')

# Get probability for retained users for 30 days with 4 installs
print('### Prediction for number retained for 30 days with 4 installs ###')
print(logreg.predict_proba([[4]]), '\n')


#################################### INSTALL_30 ##########################################
# Add a new column for installers retained for 30 days. Call it Install_30.
#  If greater than 0, put 1, if 0, put 0
data['Install_30'] = np.where(data['Installers_retained_for_30_days'] > 0, 1, 0)

# Print the top 10 rows of the new data set
print('### Top 10 of Install_30 Column ###')
print(data['Install_30'].head(10))

# Print the shape of the new data set
print('### Install_30 Shape ###')
print(data['Install_30'].shape)

# Print the description of the new data set
print('### Install_30 Description ###')
print(data['Install_30'].describe())



########### LOGISTIC REGRESSION #######################

# Perform logistic regression using Install_30 column for y
# X = Installers column
# Set y to numpy array with the right shape
y = data['Install_30']

# Fit X and y for logistic regression
logreg.fit(X, y)

# Print the top 10 rows of X
print('### Top 10 Rows of X ###')
print(X.head(10), '\n')

# Print the shape rows of X
print('### Shape of X ###')
print(X.shape, '\n')

# Print the description of X
print('### Description of X ###')
print(X.describe(), '\n')

# Print the top 10 rows of y
print('### Top 10 Rows of y ###')
print(y.head(10), '\n')

# Print the shape of y
print('### Shape of y ###')
print(y.shape, '\n')

# Print the description of y
print('### Description of y ###')
print(y.describe(), '\n')

# Predict on X and capture the result to assorted_pred_class
assorted_pred_class = logreg.predict(X)

# Print the predictions using assorted _pred_class (X predictions)
print('### Predictions Based on X ###')
print(assorted_pred_class, '\n')

# Get the predicted probabilities of class 1 and save to assorted_pred_prob
assorted_pred_prob = logreg.predict_proba([[1]])

# Print the probabilities using assorted_pred_prob
print('### Predictions for Class 1 ###')
print(assorted_pred_prob, '\n')

# Predict the probability of maintaining
# a user for 30 days (0, 1)

# Get probability for retained users for 30 days with 1 install
print('### Prediction for number retained for 30 days with 1 install ###')
print(logreg.predict_proba([[1]]), '\n')

# Get probability for retained users for 30 days with 2 installs
print('### Prediction for number retained for 30 days with 2 installs ###')
print(logreg.predict_proba([[2]]), '\n')

# Get probability for retained users for 30 days with 4 installs
print('### Prediction for number retained for 30 days with 4 installs ###')
print(logreg.predict_proba([[4]]), '\n')


# Print the intercept
print('### Log Regression Y Intercept ###')
print(logreg.intercept_, '\n')

# Print the coefficient
print('### Log Regression Coefficient ###')
print(logreg.coef_)
