import sys
import numpy as np

b = np.loadtxt(sys.argv[1], skiprows=1);

print("Norm of the vector = ", np.linalg.norm(b))
print("Sum Reduce of the vector = ", np.sum(b))

