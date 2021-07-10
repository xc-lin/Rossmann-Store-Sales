import numpy as np
import pandas

x = {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]}
y = {"c": [1, 2, 3, 4]}
xd = pandas.DataFrame(x)
yd = pandas.DataFrame(y)
print(yd[xd["a"] == 2])
k = [0.1, 0.2, 0.3, 0.4]
k2=np.array(k)
print(k*0.5)
