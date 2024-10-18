from pprint import pprint

import numpy as np

from vectorial_subspace import VectorialSubspace

similarity_value = 0.95
minimization_step = (1 - similarity_value) / 2
window_size = 4
step_window = int(window_size / 2)

vectorial_subspace = VectorialSubspace(
    metric="cosine",
    threshold=similarity_value,
    minimization_step=minimization_step,
    window_size=window_size,
    window_step=step_window,
    method="COBYLA",
    intervals_reducing_type="hard",
    verbose=1
)

N = 10
tensor_target = np.random.randn(N)
print(tensor_target)
vectorial_subspace.optimize(tensor_target)
pprint(vectorial_subspace.intervals)
print(len(vectorial_subspace.intervals))

