import numpy as np
import pandas as pd
from pandas import Series,DataFrame

#Removing duplicates
df = DataFrame({'c1':[1,1,2,2,3,3,3],'c2':['a','a','b','b','c','c','c'],
                'c3': ['A','A','B','B','C','C','C']})
print(df)
print(df.duplicated())
print(df.drop_duplicates())
df = DataFrame({'c1':[1,1,2,2,3,3,3],'c2':['a','a','b','b','c','c','c'],
                'c3': ['A','A','B','B','C','D','C']})
print(df)
print(df.drop_duplicates(['c3']))

#--------------------------------------
# Data Concatenation and transformation
#--------------------------------------

df = DataFrame(np.arange(36).reshape(6,6))
print(df)

df2 = DataFrame(np.arange(15).reshape(5,3))
print(df2)
print(pd.concat([df,df2],axis=1))
print("--line 26")
print(pd.concat([df,df2]))
# Adding data
sobj = Series(np.arange(6))
sobj.name = "added_var"
print(sobj)

variable_added = DataFrame.join(df,sobj)
print(variable_added)
added_table = variable_added.append(variable_added,ignore_index=True)
print(added_table)

#Sorting data
df_sorted = df.sort_values(by=[5],ascending=[False])
print(df_sorted)

# Data Grouping and Aggregation
