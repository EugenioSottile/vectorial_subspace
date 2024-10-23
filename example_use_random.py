from pprint import pprint
from statistics import mean

import numpy as np

from random_vectorial_subspace import RandomVectorialSubspace
from vectorial_subspace import VectorialSubspace


similarity_value = 0.95
random_step = 1000
shift_value = 0.2

vectorial_subspace = RandomVectorialSubspace(
    metric="cosine",
    threshold=similarity_value,
    intervals_reducing_type="disjunction",
    random_step=random_step,
    shift_value=shift_value,
    verbose=1
)

N = 768
tensor_target = np.random.randn(N)
print(tensor_target)
vectorial_subspace.optimize(tensor_target)
print()
pprint(vectorial_subspace.intervals)
print(len(vectorial_subspace.intervals))


differences = [tuple_[0][1] - tuple_[0][0] for tuple_ in vectorial_subspace.intervals]
mean_differences = mean(differences)
print(differences)
print(mean_differences)
