import numpy as np
import pandas as pd
from pandas import Series,DataFrame

'''
Hierarchical Indexing-It comes into picture when multiple index levels (one or more) on the axis.
we separte the dependency by unstack().
It provides a way to work higher-dimensional data on lower dimensional form 
'''

dt = pd.Series(np.random.randn(9),index=[['a', 'a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                                         [1, 2, 3, 1, 3, 1, 2, 2, 3]])
print(dt)
print(dt.index)
print(dt.unstack())

data = pd.DataFrame(np.arange(12).reshape(4,3),
                    index=[['a','a','b','b'],[1,2,1,2]],
                    columns=[['Ohio','Ohio','Colorado'],['Green','Red','Green']])
print(data.unstack())
print(data['Ohio'].unstack())


# Naming the indices
data.index.names = ['key1','key2']
data.columns.names = ['state','color']
print(data)
print(data['Ohio'])


#Reordering and Sorting
print(data.swaplevel('key1','key2'))
print(data.sort_index(level=1))
print(data.swaplevel(0,1).sort_index())


# Summary statistics by level
print(data.sum(level='key2'))
print(data.sum(level='color',axis=1))


# Indexing with DataFrame's column
df = pd.DataFrame({'a':range(7),'b':range(7,0,-1),
                   'c':['one','one','one','two','two','two','two'],
                   'd':[0,1,2,0,1,2,3]})
print(df.set_index(['c','d']))
print(df.set_index(['a','d'],drop=False))


'''
Combining and merging data sets 
- pandas.merge connects rows in dataframe based on one or more keys
- pandas.concat concatenates or 'stacks' together objects along the axis
- pandas combine_first instance method enables slicing together overlapping data to fill missing values
 in one object from the values from another object
  
Database style - DataFrame joins
'''
df1 = pd.DataFrame({'data1':range(7),'key':['b','b','a','c','a','a','b']})
print(df1)
df2 = pd.DataFrame({'data2':range(3),'key':['a','b','d']})
print(df2)

# MERGE
# By default merge happens with union keys of two sets or frames
print(pd.merge(df1,df2,on='key'))

# on mentioning how='inner' merge happen on intersection of two sets or frames
print(pd.merge(df1,df2,how='inner'))

# on mentioning how='outer' merge happens on union of two sets or frames, unfilled values
# are mentioned as NAN
print(pd.merge(df1,df2,how='outer'))

# Left merge happens, on every element of left set with the each element of right set
print(pd.merge(df1,df2,how='left'))

#Right merge happens, on every element of right set with the each element of left set
print(pd.merge(df1,df2,how='right'))

'''
Merge with multiple keys
'''

left_df = pd.DataFrame({'key1':['foo', 'foo', 'bar'],
                        'key2':['one', 'two', 'one'],'lval':[1,2,3]})
right_df = pd.DataFrame({'key1':['foo', 'foo', 'bar', 'bar'],
                        'key2':['one', 'one', 'one', 'two'],'rval':[4,5,6,7]})
print(left_df)
print(right_df)

print(pd.merge(left_df,right_df,on=['key1','key2'],how='inner'))  # Intersection
print(pd.merge(left_df,right_df,on=['key1','key2'],how='outer'))  # Union + common

#For overlapping columns the merge has the suffix options for specifiying strings to overlapping names
# for left and right DataFrame objects
print(pd.merge(left_df,right_df,on='key1',suffixes=('_left','_right')))


# Merge on index
# Here to merge on index, either we should pass left_index=True or right_index=True to
# indicate index should be used as key to merge

left1 = pd.DataFrame({'key':['a', 'b', 'a', 'a', 'b', 'c'],'value': range(6)})
right1 = pd.DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])
print(left1);print(right1)
print(pd.merge(left1,right1,left_on='key',right_index=True))

# left_on(left side columns) + right_index could merge
# right_on(right side columns) +left_index could merge
lefth = pd.DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio','Nevada', 'Nevada'],
                      'key2': [2000, 2001, 2002, 2001, 2002],'data': np.arange(5.)})
righth = pd.DataFrame(np.arange(12).reshape((6, 2)),index=[['Nevada', 'Nevada', 'Ohio', 'Ohio','Ohio', 'Ohio'],
                        [2001, 2000, 2000, 2000, 2001, 2002]],columns=['event1', 'event2'])

print(lefth);print(righth)
print(pd.merge(lefth,righth,left_on=['key1','key2'],right_index=True))
print(pd.merge(lefth,righth,left_on=['key1','key2'],right_index=True,how='outer'))

# Using the indexes of both sides of the merge is also possible
left2 = pd.DataFrame([[1., 2.], [3., 4.], [5., 6.]],index=['a', 'c', 'e'],columns=['Ohio', 'Nevada'])
right2 = pd.DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],index=['b', 'c', 'd', 'e'],columns=['Missouri', 'Alabama'])

print(pd.merge(left2,right2,left_index=True,right_index=True,how='outer'))
#Same result by using join
print(left2.join(right2,how='outer'))

another = pd.DataFrame(DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],index=['a', 'c', 'e', 'f'],
                                    columns=['New York', 'Oregon']))
print(another)
print(left2.join([right2,another],how='outer'))

#Concatenating along X axis

arr = np.arange(12).reshape(3,4)
print(arr)
print(np.concatenate([arr,arr],axis=1))
s1 = pd.Series([0, 1], index=['a', 'b'])
s2 = pd.Series([2, 3, 4], index=['c', 'd', 'e'])
s3 = pd.Series([5, 6], index=['f', 'g'])

print(pd.concat([s1,s2,s3],axis=1))

# Combining data with Overlap
a = pd.Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan],index=['f', 'e', 'd', 'c', 'b', 'a'])
b = pd.Series(np.arange(len(a), dtype=np.float64),index=['f', 'e', 'd', 'c', 'b', 'a'])
b[-1] =np.nan
print(a);print(b)
# if is null()? fill it up with b value
print(np.where(pd.isnull(a),b,a))

df1 = pd.DataFrame({'a': [1., np.nan, 5., np.nan],'b': [np.nan, 2., np.nan, 6.],'c': range(2, 18, 4)})
df2 = pd.DataFrame({'a': [5., 4., np.nan, 3., 7.],'b': [np.nan, 3., 4., 6., 8.]})

print(df1);print(df2)
print(df1.combine_first(df2))

''' Reshaping
Stack() converts columns into rows
unstack() converts rows into columns
'''
data = pd.DataFrame(np.arange(6).reshape((2, 3)),index=pd.Index(['Ohio', 'Colorado'], name='state'),
                    columns=pd.Index(['one', 'two', 'three'],name='number'))
print(data)
result= data.stack()
print(result.unstack(0))

