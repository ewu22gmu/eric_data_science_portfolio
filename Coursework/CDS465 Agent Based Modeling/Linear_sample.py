print('hello world!')

import numpy as np
import pandas as pd

Ab = pd.read_csv('/Users/eric/Downloads/AbMat.csv').fillna(0.00)

A = Ab.iloc[:,1:].copy()
b = Ab['pre mm price'].copy()

# 1. Calculate the Moore-Penrose Pseudoinverse of A
A_pinv = np.linalg.pinv(A)

# 2. Calculate the minimum-norm solution x = A_pinv @ b
x_pinv = A_pinv @ b

print(f"Pseudoinverse (A+):\n{A_pinv}")
print(f"\nMinimum-Norm Solution (x):\n{x_pinv}")
print(f"\nCheck: A * x = {A @ x_pinv}") # This should equal b



# Solve using numpy.linalg.lstsq
# 'rcond=None' suppresses a warning for older versions of numpy
x_min_norm, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

print(f"Matrix A:\n{A}")
print(f"\nVector b: {b}")
print(f"\nMinimum-Norm Solution (x):\n{x_min_norm}")
print(f"\nCheck: A * x = {A @ x_min_norm}") # This should equal b


x_min_norm_solution = np.insert(x_min_norm,0,9999.0)
x_min_norm_solution.shape

Ab.loc[len(Ab)] = x_min_norm_solution
#Ab.to_csv('/Users/eric/Downloads/AbMat_solved.csv')
