import numpy as np

a = 1.0
print(type(a))
a1 = np.float32(a)
print(type(a1))
a2 = [a1, a, 4.6, 2, 3, 4]
array_after = np.array(a2, dtype=np.float32)
print(array_after.shape)
for number in array_after:
    print(type(number))
