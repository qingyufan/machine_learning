
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

labels = np.zeros(shape=(7190,1),dtype=int)
for i in range(10):
    for j in range(719):
        labels[719*i+j][0] = i
print(labels)
