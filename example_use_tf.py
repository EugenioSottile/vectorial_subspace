from pprint import pprint
from statistics import mean

import numpy as np

from vectorial_subspace import VectorialSubspace
from vectorial_subspace_tensorflow import VectorialSubspaceTF

similarity_value = 0.8
maxiter = 10000
window_size = 16
step_window = 8
#step_window = 6
interval_width = 0.0
random_step = 0

"""constraints = []
for i in range(0, N, 4):
    constraints.append(i)"""

epochs = 15
learning_rate = 0.01

vectorial_subspace = VectorialSubspaceTF(
    epochs=epochs,
    learning_rate=learning_rate,
    metric="cosine",
    threshold=similarity_value,
    interval_width=interval_width,
    window_size=window_size,
    window_step=step_window,
    intervals_reducing_type="union",
    random_steps=random_step,
    add_penalty=True,
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
