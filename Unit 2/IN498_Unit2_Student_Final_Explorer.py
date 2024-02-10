import pandas as pd
import numpy as np
import sys

# Ignoring warnings
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# Widen the column display
pd.set_option('max_colwidth', 500)

# Files to analyze in a list
files = ["retained_installers_com.foo.bar_201904_country_parsed.csv",
         "retained_installers_com.foo.bar_201905_country_parsed.csv",
         "retained_installers_com.foo.bar_201906_country_parsed.csv",
         "retained_installers_com.foo.bar_201907_country_parsed.csv",
         "retained_installers_com.foo.bar_201908_country_parsed.csv",
         "retained_installers_com.foo.bar_201909_country_parsed.csv",
         "retained_installers_com.foo.bar_201910_country_parsed.csv",
         "retained_installers_com.foo.bar_201911_country_parsed.csv",
         "retained_installers_com.foo.bar_201912_country_parsed.csv"]

# Columns for the upcoming dataframe
col_names = ["Date","Package_Name", "Country", "Store_Listing_Visitors", "Installers",
             "Visitor-to-Installer_conversion_rate", "Installers_retained_for_1_day",
             "Installer-to-1_day_retention_rate", "Installers_retained_for_7_days",
             "Installer-to-7_days_retention_rate", "Installers_retained_for_15_days",
             "Installer-to-15_days_retention_rate", "Installers_retained_for_30_days",
             "Installer-to-30_days_retention_rate"]


# Function to explore the data with snippets, shapes, descriptions, etc.
def dataexplorer(df):
    # Print the top 10 rows
    print("###########Top 10 Rows###########")
    print(df.head(10))
    print()

    # Print the shape
    print("###########Data Shape###########")
    print(df.shape)
    print()

    # Print the description of the entire data set
    print("###########Data Description###########")
    print(df.describe())
    print()

    # Print the counts of the data
    print("###########Data Counts###########")
    print(df.count())
    print()

    # Get the mean for installers retained 1 day
    print('Installers retained 1 day mean: ' + str(df['Installers_retained_for_1_day'].mean()))

    # Get the mode for installers retained 1 day1
    print('Installers retained 1 day mode: ' + str(df['Installers_retained_for_1_day'].mode()), '\n')

    # Get the mean for installers retained 7 days
    print('Installers retained 7 days mean: ' + str(df['Installers_retained_for_7_days'].mean()))

    # Get the mode for installers retained 7 days
    print('Installers retained 7 days mode: ' + str(df['Installers_retained_for_7_days'].mode()), '\n')

    # Get the mean for installers retained 15 days
    print('Installers retained 15 days mean: ' + str(df['Installers_retained_for_15_days'].mean()))

    # Get the mode for installers retained 15 days
    print('Installers retained 15 days mode: ' + str(df['Installers_retained_for_15_days'].mode()), '\n')

    # Get the mean for installers retained 30 days
    print('Installers retained 30 days mean: ' + str(df['Installers_retained_for_30_days'].mean()))

    # Get the mode for installers retained 30 days
    print('Installers retained 30 days mode: ' + str(df['Installers_retained_for_30_days'].mode()), '\n')


# Function to replace missing dates
def fill_missing_dates(dfdates, missingvallocs):
    # Loop through each missing value location
    for loc in missingvallocs:
        if loc > 0:
            # Get the value to the left and right of the missing value
            nextDate = dfdates[loc - 1]
            prevDate = dfdates[loc + 1]

            # Replace the date if both values are equal (We would add more logic if this didn't fix all missing values,
            # but it does in this case)
            if prevDate == nextDate:
                dfdates[loc] = prevDate

    return dfdates


# Function to replace missing country codes (uses global variable countries as list we find missing values from)
def fill_missing_countries(dfcountries, missingvallocs):
    # Loop through each missing value location
    for loc in missingvallocs:
        # Get the country in the location one to the left
        prevCountry = dfcountries[loc - 1]

        # Get the index of the country we just found and add the next country to the return list
        index = np.where(countries == prevCountry)[0][0]
        dfcountries[loc] = countries[index + 1]

    return dfcountries


# Function to fill missing retention numbers at 30 day mark by finding mean of other values
def fill_missing_installs_retained(df15dayinstalls, df30dayinstalls, missingvallocs):
    # dataframe with dropped missing vals
    dfInstallsDroppedNA = df30dayinstalls.dropna()
    # Get mean of non-missing values and round to an int
    imputedMean = round(dfInstallsDroppedNA.mean())

    # if the 15 day install number is 0, then fill with that
    for loc in missingvallocs:
        if df15dayinstalls[loc] == 0:
            df30dayinstalls[loc] = int(round(0))

    # Fill remaining missing values and return corrected data frame
    df30dayinstalls.fillna(int(imputedMean), inplace=True)
    return df30dayinstalls


# Function to fill missing retention rate
def fill_missing_retention_rates(dfinitinstalls, df30dayinstalls, df30dayrates, missingvallocs):
    # Loop through each missing value location
    for loc in missingvallocs:
        # quickly add 0 to the value if no installs remain
        if df30dayinstalls[loc] == 0.0:
            df30dayrates[loc] = 0.0
        else:
            # calculate the retention rate by finding the quotient of the 30 day install and initial install numbers
            rRate = df30dayinstalls[loc] / dfinitinstalls[loc]
            df30dayrates[loc] = rRate

    return df30dayrates

# Combined CSV File
combinedFile = 'retained_installers_com.foo.bar_Combined.csv'

# Print the file name
print("##########File#########")
print(combinedFile)
print()

# Get the data frame from the file
dfInstallerStats = pd.read_csv(combinedFile, names=col_names)

# Print initial Values
print("################## INITIAL VALUES ##################")
dataexplorer(dfInstallerStats)

# START DATA MUNGING/CLEANING
# Get the highest number of counts, so we know how many are missing in each column
correctDataCount = dfInstallerStats.count().max()

# Columns that need to be cleaned and countries list (we assume no countries are completely missing from the data)
colsToClean = []
countries = dfInstallerStats['Country'].unique()
# drop value at index 60 due to being nan
countries = np.delete(countries, 60)

# Counts of the data
dataCounts = dfInstallerStats.count()


# Add columns with less than the max data count to the columns that need to cleaned
for col, count in dataCounts.items():
    if count < correctDataCount:
        colsToClean.append(col)


# Clean each column that is missing values
for col in colsToClean:
    # Save location of missing values
    missingValLocs = dfInstallerStats[dfInstallerStats[col].isnull()].index.tolist()

    # Perform cleaning based on column
    if col == 'Date':
        correctedVals = fill_missing_dates(dfInstallerStats[col], missingValLocs)
        dfInstallerStats[col] = correctedVals
    elif col == 'Country':
        correctedVals = fill_missing_countries(dfInstallerStats[col], missingValLocs)
        dfInstallerStats[col] = correctedVals
    elif col == 'Installers_retained_for_30_days':
        fifteenDayInstalls = dfInstallerStats['Installers_retained_for_15_days']
        correctedVals = fill_missing_installs_retained(fifteenDayInstalls, dfInstallerStats[col], missingValLocs)
        dfInstallerStats[col] = correctedVals
    else:
        initInstalls = dfInstallerStats['Installers']
        thirtyDayInstalls = dfInstallerStats['Installers_retained_for_30_days']
        correctedVals = fill_missing_retention_rates(initInstalls, thirtyDayInstalls, dfInstallerStats[col], missingValLocs)
        dfInstallerStats[col] = correctedVals

# Print the date exploration of the cleaned data
print("################## CLEANED VALUES ##################")
dataexplorer(dfInstallerStats)