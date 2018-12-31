
import numpy as np
import pandas as pd

series_obj = pd.Series(np.arange(8),index=['a','b','c','d','e','f','g','h'])
print(series_obj)
print(series_obj[[0,5]])

np.random.seed(25)
df_obj = pd.DataFrame(np.random.rand(36).reshape((6,6)),index=['r1','r2','r3','r4','r5','r6'],
columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
print(df_obj)
print(df_obj.ix[['r2','r4'],['c1','c4']])
# Data slicing
print(series_obj['a':'f'])
#comparing with scalars
print(df_obj < 0.5)

# Filtering with scalars
print(series_obj[series_obj > 6])
series_obj['a','d','f'] = 8
print(series_obj)

from pandas import Series,DataFrame
# MISSING VALUES
#Discovering what data is missing
missing = np.nan
series_obj = Series(['r1','r2',missing,'r4','r5','r6',
                      missing,'r8'])
print(series_obj)
print(series_obj.isnull())

# Filling missing values
np.random.seed(25)
df_obj = DataFrame(np.random.rand(36).reshape(6,6))
print(df_obj)
df_obj.ix[3:5,0]=missing
df_obj.ix[1:4,5] =missing
print(df_obj)

filled_ob = df_obj.fillna(0)
print(filled_ob)

filled_ob = df_obj.fillna({0: 0.1,5:1.25})
print(filled_ob)
# Forward filling
f_d = df_obj.fillna(method='ffill')
print(f_d)

# Counting mising values
np.random.seed(25)
df_obj = pd.DataFrame(np.random.rand(36).reshape((6,6)),index=['r1','r2','r3','r4','r5','r6'],
columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
df_obj.ix[3:5,0]=missing
df_obj.ix[1:4,5] =missing
print(df_obj)
print(df_obj.isnull().sum())
# Call dropna to drop

df_no_nan = df_obj.dropna(axis=1)
print(df_no_nan)
