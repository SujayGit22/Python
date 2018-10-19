import numpy as np
import pandas as pd
from pandas import Series,DataFrame

# Series is one dimensional array object with index values

obj = pd.Series([3,5,2,5])
print(obj)
print(obj.values)
print(obj.index)

obj1 = pd.Series([11,22,33,44],index=['a','b','c','d'])
print(obj1)
# Operations
print(obj1*2)
print(obj1[obj1>15])
print(np.exp(obj1))

sdata = {1:"sam",2:"ram",3:'som',4:'bhim'}
sda= pd.Series(sdata,index=obj.values)
print(sda,sda.values,sda.index)
print(pd.isnull(sda))

# Essential reindexing
obj = pd.Series([4.4,5.4,-3.6,2.4],index=['d','s','f','g'])
print(obj)
print(obj.reindex(['a','f','d','s','g','s']))

# Dropping entries  from an axis

# Essential re-indexing
obj = pd.Series([4.5,3.3,2.3,5.6],index=['d','c','b','a'])
print(obj)
print(obj.reindex(['a','b','c','d','e']))

ob3 = pd.Series(['yellow','red','black','green'],index=[0,2,4,5])
print(ob3)
print(ob3.reindex(range(6),method='ffill'))

frame = pd.DataFrame(np.arange(9).reshape(3,3),
                     index=['a','c','d'],columns=['Ohio','Texas','California'])
print(frame)
fr2 = frame.reindex(['a','b','c','d'])
print(fr2)
states = ['Texas','Ohio','Oman']
print(fr2.reindex(columns=states))
print(frame.loc[['a','b','c','d'],states])

# Dropping entries, index selection,
print(frame.drop(['d']))
print(frame.drop(['Texas'],axis=1))
print(frame.drop(['a'],axis=0))
frameTemp = frame.drop(['Ohio','Texas'],axis=1)
print(frameTemp)
frameTemp.drop('c',inplace=True)
print(frameTemp)
frame['c':]=5
print(frame)

data = pd.DataFrame(np.arange(16).reshape(4,4),
                    index=['ohio','texas','Austin','California'],columns=['one','two','three','four'])
print(data)
print(data['three'] > 4)
data['two'] = data['one']+data['two']
data['three'] = data['two']+data['three']
data['four']= data['three']+data['four']
print(data)
print(data.columns.size)
print(data.values)
data = pd.DataFrame(data.values,index=['ohio','texas','Austin','California'],columns=[np.arange(4)])
print("line 71")
i=0
'''for i in np.arange(4):
    data[i] = data[i].values + data[i].values
'''
print(data)
print("========loc and iloc")
'''Selection loc and iloc
 They enable you to select a subset of the rows and columns from a
 DataFrame with NumPy-like notation using either axis labels (loc) or integers (iloc).
'''
data = pd.DataFrame(np.arange(16).reshape(4,4),
                    index=['ohio','texas','Austin','California'],columns=['one','two','three','four'])

print(data.loc['texas',['one','four']])
print(data.loc[['texas','Austin'],['one']])
dataDp = pd.DataFrame(data.values,index=['ohio','texas','Austin','California'],columns=[np.arange(4)])
print(dataDp.iloc[0:3,2:])

# Arithametic and data alignment
s1 = pd.Series([2.4,6.4,-6.7,7.0],index=['a','b','c','d'])
s2 = pd.Series([5.5,8.3,-2.4,-4,-3.0],index=['a','b','c','f','g'])
print(s1);print(s2)
print(s1+s2)

# Arithametic values with fill
df1 = pd.DataFrame(np.arange(12.).reshape(3,4),columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape(4,5),columns=list('abcde'))
print("-------Df1")
print(df1)
print(df2)
print(df1+df2)
print(df1.add(df2,fill_value=0))

# Substrating frame and series
frame = pd.DataFrame(np.arange(12).reshape((4,3)),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'],columns=list('bde'))
series = frame.iloc[0]
print(frame)
print(series)
print(frame - series)

# Function application and mapping
frame = pd.DataFrame(np.random.randn(4,3),
                     columns=list('edb'),
                     index=['Utah', 'Ohio', 'Texas', 'Oregon'])
print(frame)
print(np.abs(frame))
f = lambda x: x.max() - x.min()

print(frame.apply(f))
print(frame.apply(f,axis='columns'))

format = lambda x: '%.2f' % x
print(frame.applymap(format))
print(frame['e'].map(format))

#Sorting and ranking
print(frame.sort_index())
print(frame.sort_index(axis=1))
print("line 128")
print(frame.sort_values(by=['b'],ascending=False))

ob4 = pd.Series([7,-5,4,7,4,3,1,2])
print(ob4.rank())
print(ob4.rank(method='first'))
print(ob4.rank(ascending=False,method='max'))

# Axis indexing and duplicates
ob = pd.Series(range(5),index=['a','b','b','a','c'])
print(ob.index.is_unique)

# Computing Descriptive Statistics
df = pd.DataFrame([[1.4, np.nan], [7.1, -4.5],[np.nan, np.nan], [0.75, -1.3]],
                    index=['a', 'b', 'c', 'd'],columns=['one', 'two'])
print(df)
print(df.sum())
print(df.sum(axis='columns'))
print(df.mean(axis=1,skipna= False))
print(df.describe())
print("line 150")
obj = pd.Series(['a', 'a', 'b', 'c'] * 4)
print(ob.describe())

''' Corelation and Covariance
Covariance is measured wheather the 2 variable are positively or negatively realted
It is calculated by the formula. It is used to measure whether the value is increasing or decreasing at a point
COV(x,y) = [i=0 -> n] Sigma  (Xi - Xmean)(Yi - Ymean) / (n-1)
--------------------------------------------------------------
Corelation coeff is the degree to which they vary, value lies between -1 to 1
Cor r = Cov(x,y) / Sx Sy
Sx Sy are standard deviation
'''
returns =pd.DataFrame([[-0.000680,0.001837, 0.002072, -0.003483],
       [-0.002979, 0.007846 ,0.003583, -0.002255],
       [-0.000512, -0.005652, 0.001719, -0.004867],
       [-0.003930, 0.003011, -0.012474 ,0.042096]],columns=['AAPL', 'IBM', 'MSFT', 'GOOG'])
print(returns)
print(returns['MSFT'].corr(returns['AAPL']))
print(returns['MSFT'].cov(returns['AAPL']))
print(returns.corr())
print(returns.cov())
print(returns.corrwith(returns.IBM))
