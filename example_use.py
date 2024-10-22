from pprint import pprint
from statistics import mean

import numpy as np

from vectorial_subspace import VectorialSubspace

similarity_value = 0.85
maxiter = 1000
window_size = 16
step_window = int(window_size / 2)
interval_width = 1

vectorial_subspace = VectorialSubspace(
    metric="cosine",
    threshold=similarity_value,
    maxiter=maxiter,
    window_size=window_size,
    window_step=step_window,
    interval_width=interval_width,
    method="COBYLA",
    intervals_reducing_type="union",
    expand_factor=0,
    verbose=1
)

N = 100
tensor_target = np.random.randn(N)
print(tensor_target)
vectorial_subspace.optimize(tensor_target)
print()
pprint(vectorial_subspace.intervals)
print(len(vectorial_subspace.intervals))

oracle_ = [True if vectorial_subspace.intervals[i][0][0] <= tensor_target[i] <= vectorial_subspace.intervals[i][0][1] else False for i in range(N)]
print(oracle_)

differences = [tuple_[0][1] - tuple_[0][0] for tuple_ in vectorial_subspace.intervals]
mean_differences = mean(differences)
print(differences)
print(mean_differences)
