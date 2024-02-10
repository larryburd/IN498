import pandas as pd
import numpy as np
import seaborn
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

# Ignoring warnings
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

# Widen the column display
pd.set_option('max_colwidth', 500)

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
data = pd.read_csv('retained_installers_com.foo.bar_Combined.csv', names=col_names)

########### EXPLORE THE ORIGINAL DATA SET #######################
# sns.barplot(data['Installers_retained_for_1_day'].value_counts())
data['Installers_retained_for_1_day'].value_counts().plot(title='Installers Retained for One Day',
                                                          kind='bar')
plt.show()

# Plot the installer retention rates for 7 days using a bar chart
data['Installers_retained_for_7_days'].value_counts().plot(title='Installers Retained for Seven Days',
                                                           kind='bar')
plt.show()

# Plot the installer retention rates for 15 days using a bar chart
data['Installers_retained_for_15_days'].value_counts().plot(title='Installers Retained for Fifteen Days',
                                                            kind='bar')
plt.show()

# Plot the installer retention rates for 30 days using a bar chart
data['Installers_retained_for_30_days'].value_counts().plot(title='Installers Retained for Thirty Days',
                                                            kind='bar')
plt.show()

# Plot the retention of users at 30 days per date using a line chart
# data['Installers_retained_for_30_days'].value_counts().plot(title='Installers Retained for Thirty Days',
#                                                            kind='line')
data.plot(x='Date', y='Installers_retained_for_30_days', figsize=(10, 5), grid=True)
plt.title('30-Day Retention by Date')
plt.show()

# Add a month column by converting the Date column for month only
data['Month'] = pd.DatetimeIndex(data['Date']).month

# Plot the installs per month using a bar chart
data['Month'].value_counts().plot(title='Installs by Month', kind='bar')
plt.show()

########### EARLY SEABORN PLOTS #######################
# Seaborn simple regression plot of installers retained for 1 day
sns.lmplot(x='Installers', y='Installers_retained_for_1_day', data=data, ci=None)
plt.title('Linear Regression Installer and Retained for 1 Day')
plt.show()

# Seaborn simple regression plot of installers retained for 7 day
sns.lmplot(x='Installers', y='Installers_retained_for_7_days', data=data, ci=None)
plt.title('Linear Regression Installer and Retained for 7 Days')
plt.show()

# Seaborn simple regression plot of installers retained for 30 day
sns.lmplot(x='Installers', y='Installers_retained_for_30_days', data=data, ci=None)
plt.title('Linear Regression Installer and Retained for 30 Days')
plt.show()

############ FIX MISSING DATA #######################
# Replace NaN with 0 for Installers_retained_for_30_days
data[np.isnan(data.Installers_retained_for_30_days)] = 0

############ SEABORN PLOT AFTER FIXING MISSING DATA #######################
# Seaborn pairplot for installers retained for 1, 7, 15, and 30 days
sns.pairplot(data,
             x_vars=['Installers_retained_for_1_day', 'Installers_retained_for_7_days',
                     'Installers_retained_for_15_days', 'Installers_retained_for_30_days'],
             y_vars='Installers', size=6, aspect=0.7, kind='reg')
plt.title('Linear Regression Pair Plot')
plt.show()

########### LOGISTIC REGRESSION #######################
# Get logistic regression model
logreg = LogisticRegression()

# Set feature_cols to Installers column data only
feature_cols = ['Installers']

# Set X to feature_cols
X = data[feature_cols]

# Set y to installer retained for 30 days
y = data.Installers_retained_for_30_days

# Fit X and y to logistic regression model
logreg.fit(X, y)

# Predict classes using X and save as a variable called assorted_pred_class
assorted_pred_class = logreg.predict(X)

# Plot the predictions in a scatter plot using
# Scatter uses data.Installers and data.Installers_retained_for_30_days
# X label of Installer
# y label of predicted retained
# Plot using data.Installer and the prediction results (assorted_pred_class) on X
plt.scatter(data.Installers, data.Installers_retained_for_30_days)
plt.plot(data.Installers, assorted_pred_class, color='red')
plt.xlabel('Installer')
plt.ylabel('Predicted Retained')
plt.title('Logistic Regression Predicted Retained')
plt.show()

# Get the predicted probabilites of class 1
# Save as a variable called assorted_pred_prob
assorted_pred_prob = logreg.predict_proba(X)[:, 1]

# Scatter plot the predicted probabilities
# Scatter uses data.Installers and data.Installers_retained_for_30_days
# X label of Installer
# y label of predicted probabilities
# Plot using data.Installer and the probability results (assorted_pred_prob) on X
plt.scatter(data.Installers, data.Installers_retained_for_30_days)
plt.plot(data.Installers, assorted_pred_prob, color='red')
plt.xlabel('Installer')
plt.ylabel('Predicted Probabilities')
plt.title('Logistic Regression Predicted Probabilities')
plt.show()

#################################### INSTALL_30 ##########################################
# Add a new column for installers retained for 30 days
# Call the new column Install_30.
#  If greater than 0, put 1, if 0, put 0
data['Install_30'] = np.where(data['Installers_retained_for_30_days'] > 0, 1, 0)

########### LOGISTIC REGRESSION #######################

# Perform logistic regression using Install_30 column for y
# X = Installers column
logreg = LogisticRegression()
feature_cols = ['Installers']
X = data[feature_cols]
y = data.Install_30

# Fit X and y for logistic regression
logreg.fit(X, y)

# Predict on X and capture the result to assorted_pred_class
assorted_pred_class = logreg.predict(X)

# Plot the class predictions
# Scatter uses data.Intsallers and data.Install_30
# Plot uses data.Installers and assorted_pred_class
# X label is Installer
# Y label is Predicted Retained (Install_30)
plt.scatter(data.Installers, data.Install_30)
plt.plot(data.Installers, assorted_pred_class, color='red')
plt.xlabel('Installer')
plt.ylabel('Predicted Retained (Install_30)')
plt.title('Logistic Regression Predicted Retained (Install_30)')
plt.show()

# Get the predicted probabilites of class 1 and save to assorted_pred_prob
assorted_pred_prob = logreg.predict_proba(X)[:, 1]

# Plot the predicted probabilities
# Scatter uses data.Intsallers and data.Install_30
# Plot uses data.Installers and assorted_pred_prob
# X label is Installer
# Y label is Predicted Probabilities (Install_30)
plt.scatter(data.Installers, data.Install_30)
plt.plot(data.Installers, assorted_pred_prob, color='red')
plt.xlabel('Installer')
plt.ylabel('Predicted Probabilities (Install_30)')
plt.title('Logistic Regression Predicted Probabilities (Install_30)')
plt.show()

########### DECISION TREE #######################
# Train/test split 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build a decision tree model
dt = DecisionTreeClassifier(random_state=1)

# Fit the tree using X_train and y_train
dt.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(10,4))
tree.plot_tree(dt, filled=True)
plt.title('Decision Tree')
plt.show()

print(tree.export_text(dt))

########### RANDOM FOREST #######################
# Build a random forest tree model
# n_estimators = 1000
rf = RandomForestClassifier(n_estimators=1000)

# Fit the random tree model with X_train and y_train
rf.fit(X_train, y_train)

# Get the predicitons using X_test and save to rf_probs
rf_probs = rf.predict(X_test)

# Plot the random forest predictions using rf_probs
plt.figure(figsize=(14,4))
plt.plot(rf_probs)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.show()
