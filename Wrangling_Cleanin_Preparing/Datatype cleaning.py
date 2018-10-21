import pandas as pd
import xlrd

file = xlrd.open_workbook(".\CleanData.xlsx","rw")
sheet = file.sheet_by_name("Sheet1")

strList = []
numList = []
deciList = []

#Iterating over cells and adding into data type list
for i in range(sheet.nrows):
    row = sheet.row_values(i)
    for cell in row:
        if type(cell) is str:
            strList.append(cell)
        elif cell % 1 == 0:
            numList.append(cell)
        elif type(cell) is float:
            deciList.append(cell)

# Finding length of cell to be imputed
maxList = 0
if strList.__len__() > numList.__len__():
    if strList.__len__() > deciList.__len__():
        maxList = strList.__len__()
elif numList.__len__() > deciList.__len__():
    maxList = numList.__len__()
else: maxList = deciList.__len__

# Imputing data with 0 or NAN values
for i in range(maxList):
    if strList.__len__() < maxList:
        strList.insert(strList.__len__(),"NAN")
    if numList.__len__() < maxList:
        numList.insert(numList.__len__(),0)
    if deciList.__len__() < maxList:
        deciList.insert(deciList.__len__(),0)


# Creating data set for dataframe
data = { 'String': strList,
         'Numbers': numList,
         'Float': deciList }

dframe = pd.DataFrame(data)
print(dframe)