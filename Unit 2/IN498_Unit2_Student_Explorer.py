import pandas as pd
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

    print('Installers retained for 1 day: ' + str(df['Installers_retained_for_1_day'].sum()))
    print('Installers retained for 7 days: ' + str(df['Installers_retained_for_7_days'].sum()))
    print('Installers retained for 15 days: ' + str(df['Installers_retained_for_15_days'].sum()))
    print('Installers retained for 30 days: ' + str(df['Installers_retained_for_30_days'].sum()))

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


# Parse files to analyze structure of CSV files and data
for file in files:

    # Print the file
    print("##########File#########")
    print(file)
    print()

    # Get the data frame from the file
    dfInstallerStats = pd.read_csv(file, names=col_names)

    # Print the data exploration
    dataexplorer(dfInstallerStats)



