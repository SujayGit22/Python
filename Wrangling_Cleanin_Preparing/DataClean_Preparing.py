import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from numpy import nan as NA

# NA handling methods
st_data = pd.Series(['aadrvark','artichoke',np.nan,'avocado'])
print(st_data)
print(st_data.isnull())
print(st_data.dropna())
print(st_data.fillna(0))
print(st_data.notnull())

# Filtering out missing data
data = pd.Series([1,NA,2.3,NA,5,None,7])
print(data.isnull())
print(data.dropna())

data = pd.DataFrame([[1.,6.5, 3.],[1.,NA, NA],[NA, NA, NA],[NA, 6.5, 3.]])
cleaned = data.dropna()
print(data);print(cleaned)
print(data.dropna(how='all'))
data[4] = NA
print(data)
print(data.dropna(how='all',axis=1))

#  Futher cleaning
frame = pd.DataFrame(np.random.randn(7,3))
print(frame)
frame.iloc[2:,1] = NA
frame.iloc[:2,2] = NA
print(frame)
print(frame.dropna())
print(frame.dropna(thresh=2))

# Filling missing data
print(frame.fillna(0))
print(frame.fillna({1:0.55,2:0.33}))

frame.iloc[2:,1] = NA
frame.iloc[4:,2] = NA
print(frame.fillna(method='ffill',limit=2))

data = pd.Series([1.,NA,3.5,NA,7])
print(data.fillna(data.mean()))

# Data Duplicates, removing duplicates
data = pd.DataFrame({'k1':['one','two']*3+['two'],
                     'k2':[1,1,2,3,3,4,4]})
print(data.duplicated())
print(data.drop_duplicates())
data['v1']= range(7)
print(data.drop_duplicates(['k1','k2'],keep='last'))

# Transforming data using a function or mapping
data = pd.DataFrame({ 'food':['bacon', 'pulled pork', 'bacon','Pastrami', 'corned beef', 'Bacon',
                                'pastrami', 'honey ham', 'nova lox'],
                      'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]
                    })
meat_to_animal = {
'bacon': 'pig',
'pulled pork': 'pig',
'pastrami': 'cow',
'corned beef': 'cow',
'honey ham': 'pig',
'nova lox': 'salmon'
}
lowercased = data['food'].str.lower()
data['animal'] = lowercased.map(meat_to_animal)
print(data)

# print(data['food'].map(lambda x: meat_to_animal[x.lower()]))

# Renaming axis indexes
data = pd.DataFrame(np.arange(12).reshape(3,4),index=['Ohio','Colorado','NewYork'],
                    columns=['one','two','three','four'])
transform = lambda x: x[:].upper()
data.index = data.index.map(transform)
print(data)

print(data.rename(index=str.title,columns=str.upper))
data.rename({'OHIO':'Indiana'},inplace=True)
print(data)

# Discretization and Binning
ages = [18,34,56,22,78,90,51,77,21,93,12,67,18]
bins = [10,25,35,60,100]
cats  = pd.cut(ages,bins)
print(cats);print(cats.codes);print(cats.categories)
print(pd.value_counts(cats))

# Quartiles
data = np.random.randn(10007)
cats= pd.qcut(data,3) # Quartiles
print(cats)
print(pd.value_counts(cats))

# Detecting and filtering outliers
print("---------------------------Detecting and filtering Outliers---------------------")
data = pd.DataFrame(np.random.randint(0,10,size=(500,4),dtype=int))
print(data)
col = data[1]
d2 = col[np.abs(col) > 1]
no_cnt = pd.value_counts(d2)
print(no_cnt)

percent_abs ={
    2:no_cnt[2]*400/data.size,
    3:no_cnt[3]*400/data.size,
    4:no_cnt[4]*400/data.size,
    5:no_cnt[5]*400/data.size,
    6:no_cnt[6]*400/data.size,
    7:no_cnt[7]*400/data.size,
    8:no_cnt[8]*400/data.size,
    9:no_cnt[9]*400/data.size
}
perDf = pd.DataFrame(no_cnt)
print(perDf)
print(percent_abs)
perDf['percentage'] = perDf.index.map(percent_abs)
print(perDf)
print(data.size)
#[lambda x: no_cnt.values/1000]

def pp(st,data):
    print(data)
    print("----"+st)
# Permutation and random sampling
print('------------Permutation and random sampling------')
df = pd.DataFrame(np.arange(5 *4).reshape(5,4))
sampler = np.random.permutation(5)
#print("sampler".format(sampler))
pp("sampler",sampler)
pp("df take",df.take(sampler))
print(df.sample(n=3))
print("df sample axis 1",df.sample(n=3,axis=1))

choices = pd.Series([5,1,-7,3,4])
draws = choices.sample(n=10,replace=True)
print(draws)

# computing indicator / Dummy variables
# This we do because of categorical data

df = pd.DataFrame({'keys': ['b', 'b', 'a', 'c', 'a', 'b'],
                   'data1': [3,2,76,89,43,2]})
print(df)
print(pd.get_dummies(df['keys']))
print(df['data1'])
dummies = pd.get_dummies(df['keys'],prefix='key')
df_with_dummy = df[['data1']].join(dummies)
print(df_with_dummy)

# Creating dummy variable for movie data
#fetching file
file = pd.read_excel('.\movie_data.xlsx')
movies = pd.DataFrame(file)
print(movies.head())
# Storing all genres
all_generes = []
for x in movies.genres:
    all_generes.extend(x.split('|'))
#Getting only unique genres
genres = pd.unique(all_generes)
genres

# creating empty dummy values
zero_matrix = np.zeros((len(movies),len(genres)))
dummies = pd.DataFrame(zero_matrix,columns=genres)
print(dummies)
#filling values in the zero_matrix
gen = movies.genres[0]
for i,gen in enumerate(movies.genres):
    indices = dummies.columns.get_indexer(gen.split('|'))
    dummies.iloc[i,indices] =1
movies_wind = movies.join(dummies.add_prefix('Genre_'))
print(movies_wind.iloc[0])