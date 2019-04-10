import numpy as np
import matplotlib.pyplot as plt
import os
#size(487, 880)
re = [[v for i in range(880)]  for v in range(1000,1588)]
re.append(np.ones(880,)*0.01)
re = np.array(re)
#re = re + np.ones(re.shape)*200
plt.imshow(re)
plt.savefig("./haha.png")
print(re.shape)