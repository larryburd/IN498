import pandas as pd
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
    f = open("IN499_Explorer.txt", "a")
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

#Files to analyze in a list
''' EXAMPLE
files = ["retained_installers_com.foo.bar_201904_country_parsed.csv",
         "retained_installers_com.foo.bar_201905_country_parsed.csv",
         "retained_installers_com.foo.bar_201906_country_parsed.csv",
         "retained_installers_com.foo.bar_201907_country_parsed.csv",
         "retained_installers_com.foo.bar_201908_country_parsed.csv",
         "retained_installers_com.foo.bar_201909_country_parsed.csv",
         "retained_installers_com.foo.bar_201910_country_parsed.csv",
         "retained_installers_com.foo.bar_201911_country_parsed.csv",
         "retained_installers_com.foo.bar_201912_country_parsed.csv"]      
'''


#Read data into a DataFrame using these columns
##"Date","Package_Name","Country","Store_Listing_Visitors",
##"Installers","Visitor-to-Installer_conversion_rate",
##"Installers_retained_for_1_day","Installer-to-1_day_retention_rate",
##"Installers_retained_for_7_days","Installer-to-7_days_retention_rate",
##"Installers_retained_for_15_days","Installer-to-15_days_retention_rate",
##"Installers_retained_for_30_days","Installer-to-30_days_retention_rate"
col_names = ["Date","Package_Name","Country","Store_Listing_Visitors","Installers","Visitor-to-Installer_conversion_rate","Installers_retained_for_1_day","Installer-to-1_day_retention_rate","Installers_retained_for_7_days","Installer-to-7_days_retention_rate","Installers_retained_for_15_days","Installer-to-15_days_retention_rate","Installers_retained_for_30_days","Installer-to-30_days_retention_rate"]

#Parse files to analyze structure of CSV files and data
for file in files:

    #Print the file
    writeFunction("##########File#########", file)

    #Get the data frame from the file
    df = pd.read_csv(file, names=col_names)

    ########### EXPLORE THE DATA SET #######################

    #Print the top 10 rows


    #Print the shape


    #Print the description of the entire data set


    #Print the counts of the data


    #Get the mean for installers retained 1 day


    #Get the mode for installers retained 1 day1


    #Get the mean for installers retained 7 days


    #Get the mode for installers retained 7 days


    #Get the mean for installers retained 15 days


    #Get the mode for installers retained 15 days


    #Get the mean for installers retained 30 days


    #Get the mode for installers retained 30 days



#Close the file handle
if PRINT == False:
    f.close()

