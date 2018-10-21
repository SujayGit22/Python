import csv
import pandas as pd
csvfile = open('.\cars.csv', 'r')
reader = csv.reader(csvfile)

data = []
for row in reader:
    row = row[0].split(';')
    data.append(row)
csvfile.close()

dataframe = pd.DataFrame(data)
print(dataframe.head(),dataframe.shape)
