'''

To justify using simple linear regression, you'll want to check the four linear regression assumptions are not violated.
These assumptions are:

Linearity
Independent observations
Normality
Homoscedasticity


𝑌=Intercept+Slope∗𝑋

Ordinary Least Squares(OLS) method of linear regression.
Introduction :
A linear regression model establishes the relation between a dependent variable(y) and at least one independent variable(x) as :
\hat{y}=b_1x+b_0
In OLS method, we have to choose the values of b_1  and b_0  such that, the total sum of squares of the difference between the calculated and observed values of y, is minimised.
Formula for OLS:
S=\sum\limits_{i=1}^n (y_i - \hat{y_i})^2 = \sum\limits_{i=1}^n (y_i - b_1x_1 - b_0)^2 = \sum\limits_{i=1}^n (\hat{\epsilon_i})^2 = min
Where,
\hat{y_i}  = predicted value for the ith observation
y_i  = actual value for the ith observation
\epsilon_i  = error/residual for the ith observation
n = total number of observations
To get the values of b_0  and b_1  which minimise S, we can take a partial derivative for each coefficient and equate it to zero.


'''



# Import the statsmodel module
import statsmodels.api as sm

# Import the ols function from statsmodels
from statsmodels.formula.api import ols

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#Read data into a DataFrame using these columns
##"Date","Package_Name","Country","Store_Listing_Visitors",
##"Installers","Visitor-to-Installer_conversion_rate",
##"Installers_retained_for_1_day","Installer-to-1_day_retention_rate",
##"Installers_retained_for_7_days","Installer-to-7_days_retention_rate",
##"Installers_retained_for_15_days","Installer-to-15_days_retention_rate",
##"Installers_retained_for_30_days","Installer-to-30_days_retention_rate"
col_names = ["Date", "Package_Name", "Country", "Store_Listing_Visitors", "Installers",
             "Visitor-to-Installer_conversion_rate", "Installers_retained_for_1_day",
             "Installer-to-1_day_retention_rate", "Installers_retained_for_7_days",
             "Installer-to-7_days_retention_rate", "Installers_retained_for_15_days",
             "Installer-to-15_days_retention_rate", "Installers_retained_for_30_days",
             "Installer-to-30_days_retention_rate"]


# Load the data
data = pd.read_csv('final_retentions_parsed.csv', names=col_names)

# Display the first five rows
print(data.head())


############ FIX MISSING DATA #######################

def findMissingPercentage():
    # Calculate the average missing rate in the intstallers retianed for 30 days column
    missing_30 = subdf.Installers_retained_for_30_days.isna().mean()

    # Convert the missing_30 from a decimal to a percentage and round to 2 decimal places
    missing_30 = round(missing_30*100, 2)

    # Display the results
    print('Percentage missing installers 30 days: ' +  str(missing_30) + '%')


# Get the columns to compare
subdf = data[['Installers', 'Installers_retained_for_1_day', "Installers_retained_for_7_days", "Installers_retained_for_15_days", 'Installers_retained_for_30_days']]

findMissingPercentage()

#Drop NaN rows in Installers_retained_for_30_days
subdf = subdf.dropna(subset = ['Installers_retained_for_30_days'], axis = 0)

findMissingPercentage()



# Subset the data
#subdf = subdf.loc[subdf['Installers_retained_for_30_days'] <= 30]
#subdf = subdf.loc[subdf['Installers'] <= 30]



# Create a pairplot of the data
sns.pairplot(subdf)
plt.show()



#######Build and fit the model##################
# Define the OLS formula
ols_formula = 'Installers_retained_for_30_days ~ Installers'
#x = subdf['Installers']
#y = subdf['Installers_retained_for_30_days']
#model = sm.OLS(y,x).fit()

# Create an OLS model
OLS = ols(formula = ols_formula, data = subdf)

# Fit the model
model = OLS.fit()

# Save the results summary
model_results = model.summary()

# Display the model results
print(model_results)





'''
Model Assumption - Linearity
The linearity assumption requires a linear relationship between the independent and dependent variables. 
A great way to check this assumption is to create a scatter plot comparing the independent variable 
with the dependent variable.

Create a scatterplot comparing the X variable you selected above with the dependent variable.
'''
# Create a scatter plot comparing X and Y
sns.scatterplot(x = subdf['Installers'], y = subdf['Installers_retained_for_30_days'])
plt.show()


'''
Model Assumption - Independence
The independent observation assumption states that each observation in the dataset is independent. 
As each install (i.e., row) is independent from one another, the independence 
assumption is not violated.



Model Assumption - Normality
The normality assumption is that the errors are normally distributed.

Create two plots to check this assumption:

Plot 1: Histogram of the residuals
Plot 2: Q-Q plot of the residuals
'''

# Calculate the residuals
residuals = model.resid

# Create a 1x2 plot figure
fig, axes = plt.subplots(1, 2, figsize = (8,4))

# Create a histogram with the residuals
sns.histplot(residuals, ax=axes[0])

# Set the x label of the residual plot
axes[0].set_xlabel("Residual Value")

# Set the title of the residual plot
axes[0].set_title("Histogram of Residuals")

# Create a Q-Q plot of the residuals
sm.qqplot(residuals, line='s',ax = axes[1])

# Set the title of the Q-Q plot
axes[1].set_title("Normal Q-Q Plot")

# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance
plt.tight_layout()

# Show the plot
plt.show()


'''
Model Assumption - Homoscedasticity
The homoscedasticity (constant variance) assumption is that the residuals have a constant variance for all values of X.

Check this assumption is not violated by creating a scatter plot with the fitted values and residuals. 
Add a line at 𝑦=0 to visualize the variance of residuals above and below 𝑦=0.
'''

# Create a scatter plot with the fitted values from the model and the residuals
fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)

# Set the x axis label
fig.set_xlabel("Fitted Values")

# Set the y axis label
fig.set_ylabel("Residuals")

# Set the title
fig.set_title("Fitted Values v. Residuals")

# Add a line at y = 0 to visualize the variance of residuals above and below 0.
fig.axhline(0)

# Show the plot
plt.show()
