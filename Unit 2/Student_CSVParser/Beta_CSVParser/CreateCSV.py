import csv
import sys
import os
import re

#Load an input file and return a list
def loadInputFile(fileName):
    #Holds file values
    fileList = []
    # Open CSV file.
    with open(fileName, newline='') as f:
        #Specify delimiter for reader.
        r = csv.reader(f)
        # Loop over rows and display them.
        for row in r:
            print(row)
            fileList.append(row)

    f.close()
    return fileList

#Save the output file from a newly created list
def saveOutputFile(fileList, fileName):

    with open(fileName, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        #Write to the parsed .csv file
        for row in fileList:
            print(row)
            writer.writerow(row)
    csv_file.close()

#Menu for opening one file and saving
# it as another with properly arranged values
while True:
    print("What would you like to do:")
    print(" 1. Open input file")
    print(" 2. Save output file")
    print(" 3. Multiple files from a directory")
    print(" 4. Quit program")
    print("Please enter a number (1-4)")
    choice = input()

    #Enter the path of the input .csv
    if(choice == '1'):
        inputFileName = input("Enter the file path and name for the input file: ")
        inputFileList = loadInputFile(inputFileName)

        print("Hmmm...may have worked")

    #Save the new file with proper list values
    if(choice == '2'):
            outputFileName = input("Enter the file path and name for the output file: ")
            saveOutputFile(inputFileList,outputFileName)
            print("Output File Saved!")

    #Create parsed files from a directory
    if(choice == '3'):

        #Put all the raw .csv files in one directory.
        #Enter the ocmplete path to the directory.
        directoryName = input("Enter the directory with multiple raw files: ")
        #get the directory listing
        listFiles = os.listdir(directoryName)

        #Loop through the files from the directory
        for file in listFiles:
            #Skip any files starting with a dot
            if re.match("^\.", file):
                continue
            print("Working on file", file)

            #Set the directory path and file name to be parsed
            myList = loadInputFile(directoryName + "/" + file)

            #Split out the file ending with .csv to retain the file name
            newName = file.split(".csv")

            #Save the output to the same file name with _parsed added
            saveOutputFile(myList, newName[0] + "_parsed.csv" )

        #Print the finished message
        print("All files processed and parsed")

    #Quit the program - buh bye
    if(choice == '4'):
        sys.exit("Buh Bye")
