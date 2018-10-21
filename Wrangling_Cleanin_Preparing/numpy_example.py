import numpy as np

# Boolean indexing
names = np.array(['a','b','c','d','c','a','d'])
data = np.random.randn(7,4)
print(names)
print(data)
print(names == 'b')
print(data[names == 'b',2:])
print(data[~(names == 'b')])
mask = (names == 'b') | (names == 'd')
print("line 12")
print(data[mask])
print("line 14")
print(data[data < 0])

# Fancy indexing
arr = np.empty((8,4))
for i in range(8):
    arr[i] = i
print(arr)
print("line 22")
print(arr[[-3,-5,-6]])

# Transposing array and swapping axis
arr = np.arange(15).reshape(3,5)
print("line 29")
print(arr)
print(arr.T)
arr= np.random.randn(6,3)
print(np.dot(arr.T,arr))
print("line 32")
arr = np.arange(16).reshape(2,2,4)
print(arr)
print(arr.transpose(1,0,2))
print(arr.swapaxes(1,2))

# Universal fnctions
arr = np.arange(10)
x = np.random.randn(10)
y = np.random.randn(10)
print("line 40")
print(np.sqrt(arr))
print(np.exp(arr))
print(x)
print(y)
print(np.maximum(x,y))

# array oriented programming with array
arr = np.random.randn(5,4)
print("Line 51")
print(arr)
print(arr.mean(axis=1))
print(arr.mean(axis=0))
print(arr.cumprod(axis=1))

# Arrays sorting
print("line 58")
arr = np.random.randn(10)
print(arr)

# Unique values
arr = np.array(['bob','cat','bat','cat'])
print("line 65")
print(np.unique(arr))
print(sorted(set(arr)))
values = np.array([6,2,5,0,4])
bools= np.in1d(values,[3,5,0])
print(bools)

#  file input and put with arrays
arr = np.arange(10)
print("line 73")
print(arr)
print(x)
np.save(x,arr)
print(arr)