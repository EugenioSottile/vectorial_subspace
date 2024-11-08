import random
from pprint import pprint
from statistics import mean

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from vectorial_subspace import VectorialSubspace
from vectorial_subspace_deep import VectorialSubspaceDeep

N = 300
tensor_target = np.random.randn(N)
#tensor_target = [1.1, -2.6, 1.6]
tensor_target = np.array(tensor_target)
print(tensor_target)
to_plot = False
#tensor_target = np.random.randn(N)
"""tensor_target = [
    0.11, -1.43, -1.20, 0.09,
    -0.23, -1.37, -0.20, 0.06,
    0.11, -2.33, 0.33, 0.10,
    -0.17, -0.22, 0.47, 0.01,
    0.12, -0.23, 1.57, -0.02,
    -0.07, 1.12, -0.15, -0.05,
    0.04, 1.66, 0.14, -0.98,
    0.04, 1.74, 0.18, 0.83,
    0.04, 1.05, -1.15, -0.02
]"""
"""tensor_target = [
    0.1027, -1.7974, 0.2170, 0.1519,
    -0.1194, -2.3348, -1.0825, 0.0100,
    0.1068, -1.6865, -1.9371, -0.1648,
    -0.0451, -3.7241, -1.2843, 0.0652,
    0.0679, -4.1323, -2.2837, -0.0546,
    -0.0025, -4.5857, -0.1976, 0.2824,
    0.0606, -5.6592, -0.3550, 0.3247,
    -0.0451, -4.0583, 1.0939, 0.4328,
    0.0679, -4.7237, 1.9365, 0.5964,
    -0.1194, -2.6701, 1.2975, 0.3686,
    0.1068, -2.2654, 2.2979, 0.4923,
    -0.1814, -0.4012, 0.4833, 0.0939,
    -0.1814, 0.4006, -0.4821, -0.1005,
    0.1027, 1.7971, -0.2167, -0.1551,
    -0.1194, 2.6698, -1.2966, -0.3759,
    0.1068, 2.2652, -2.2968, -0.5016,
    -0.0451, 4.0583, -1.0932, -0.4373,
    0.0679, 4.7234, -1.9343, -0.6092,
    -0.0025, 4.5859, 0.1969, -0.2761,
    0.0606, 5.6595, 0.3544, -0.3176,
    -0.0451, 3.7245, 1.2828, -0.0556,
    0.0679, 4.1325, 2.2822, 0.0642,
    -0.1194, 2.3353, 1.0812, 0.0100,
    0.1068, 1.6868, 1.9365, 0.1681
]"""
#N = len(tensor_target)
"""for i in range(N):
    tensor_target[i] *= 10"""
#tensor_target = tensor_target * 10
#tensor_target = np.array(tensor_target)
#tensor_target = [-0.53, 0.53, -0.31, -0.32, 0.23, 0.35, 0.56, 0.94, 0.04, 0.91, -0.82, -0.09, 0.49, 0.57, 0.17, 0.32, 0.07, -0.59, 1.32, -0.41, -0.41, 0.27, 0.77, 0.58, 0.14, -0.4, 1.01, -0.57, 0.56, 0.12, 0.53, 0.08, -0.6, 0.15, -0.81, -0.47, -0.14, 0.08, -0.44, -0.09, -0.56, 0.38, 0.33, 0.02, 0.07, 0.11, -0.88, -0.3, 0.68, 0.76, 0.03, 1.01, -0.59, 0.21, 0.47, -0.26, 1.13, 0.31, -0.5, -1.96, 0.83, 0.63, 0.09, -0.0, 0.13, 1.3, 0.68, -0.01, -0.34, 0.28, -0.55, -0.19, 0.37, -1.35, -0.25, 0.15, -0.21, 0.13, -0.03, -1.03, 0.04, 0.2, 0.94, -0.58, -0.13, -0.93, 0.14, -0.15, 0.14, -0.77, 0.66, -0.48, 1.01, -0.2, -0.89, -0.27, 0.21, 0.44, -0.17, 0.08, -0.79, -2.08, 0.49, -0.86, -0.13, 0.76, -0.84, 0.03, 0.06, 0.4, 0.08, 0.95, 0.06, 0.72, -0.32, 0.27, 0.45, -0.41, -0.77, 0.11, -0.7, -0.83, -0.18, 0.16, 1.33, 0.01, 0.2, 0.97, -0.06, -0.14, 0.09, -0.18, 0.52, 0.36, 1.19, -0.27, 0.92, 0.56, -0.95, -0.77, 0.1, -0.62, 0.23, 0.09, -1.34, 0.34, 0.26, 0.85, -0.0, -0.2, -0.48, 0.85, 0.39, -1.68, -0.4, 1.24, -0.51, -0.53, -0.31, -0.12, 1.27, 0.44, -0.45, -0.62, -0.02, -0.24, 0.41, 0.66, -1.19, 0.28, -1.04, 0.03, -1.12, 0.6, -0.07, -1.01, 1.07, -0.65, 0.67, 0.05, -0.86, -0.18, -0.93, 0.05, -0.37, 0.88, -0.26, -0.07, -1.43, -0.1, -0.46, -0.1, 0.36, -0.02, 0.12, 0.81, -0.27, -1.7, 0.18, -0.69, -0.31, 0.57, -0.41, -0.11, 0.56, -0.58, 0.62, -1.5, 0.76, 0.47, -0.87, -0.41, -1.64, -0.08, 0.24, -0.63, -0.82, -0.23, -0.29, -0.1, -0.19, -0.37, 0.16, 0.17, 1.27, 0.2, 0.33, -0.37, -0.54, 0.13, 1.05, 0.05, -0.47, -0.84, -0.17, -0.53, 0.47, 0.14, -0.56, 0.45, -0.59, -0.21, 0.37, 0.27, -0.26, 0.48, -0.04, 0.75, 0.61, -0.67, -0.26, 0.63, 0.05, 0.42, -0.57, 0.02, 0.62, -0.11, -1.04, 0.02, -0.42, 0.57, -0.35, -0.34, -0.61, -1.23, 0.97, -0.11, 0.23, 0.18, -1.03, -0.21, -0.45, 1.04, -0.0, -0.66, -0.6, -0.29, 0.53, 0.83, 0.54, -1.58, 0.92, 0.37, 1.04, -0.63, -0.92, -0.73, -0.85, 0.15, 0.08, -0.15, 0.82, 0.55, 0.42, -0.55, 0.34, 0.91, 0.9, -0.94, 0.18, -0.64, 0.15, 0.61, 0.21, 0.09, 0.85, 0.94, 1.31, -0.13, -0.3, 0.61, 0.19, -0.85, -0.67, 0.07, -0.19, -0.59, -0.34, -0.19, 0.42, -0.2, -0.14, -0.25, 0.32, -0.22, -0.1, -0.38, -0.63, -0.29, -0.06, -0.29, -0.01, -0.2, -0.21, -0.55, 0.54, 0.07, 0.72, 0.41, -0.88, -0.33, -0.06, 0.43, 0.23, -0.46, -0.14, 1.2, -0.32, 1.71, 0.5, -0.42, 0.07, -0.41, -1.22, -0.64, 0.67, 1.09, 0.83, -1.1, 0.87, -0.12, 0.99, 0.72, 0.61, 0.17, 0.3, 0.66, 0.2, 0.14, -0.33, -0.64, -0.73, 0.38, -0.06, -0.4, -0.84, 0.87, 0.9, -0.02, -0.07, -0.83, 0.01, 0.11, 0.43, -0.6, -0.71, -1.21, 1.07, -0.61, 0.53, -0.08, -0.14, -1.59, -0.4, 0.08, -0.23, 0.1, -0.11, 0.28, 0.96, 0.39, -0.47, -0.11, -0.5, -1.99, 0.58, -0.12, -0.19, 0.52, 0.3, 0.29, -1.02, 0.02, -0.68, -0.28, -0.82, 1.3, 0.09, 0.31, 1.16, 1.74, 0.17, 0.14, 0.83, 0.83, 1.06, -0.51, -0.24, -0.29, -0.51, -0.38, -0.46, 0.49, -0.52, -0.81, 0.37, 0.43, -0.79, 0.48, -0.77, 0.8, 0.61, -0.91, -1.28, -0.12, -0.9, -0.38, -1.0, 0.59, 0.36, 0.48, -0.05, 0.32, -12.61, 0.65, -0.86, 0.55, 0.38, 0.66, -0.21, -0.06, 0.52, -0.19, -0.6, 1.62, 0.19, -0.85, 0.07, -1.02, -0.57, -0.13, 0.23, -0.97, 0.1, -1.63, 0.21, 0.11, 0.66, -0.62, 0.09, -0.08, 0.19, 0.16, 0.81, 0.56, 1.34, -1.25, 0.36, 0.06, -0.21, -0.16, 1.13, 0.02, 0.72, -0.31, -0.13, -1.51, -0.55, 0.82, 0.1, 0.13, -0.1, -0.98, -0.49, 0.69, 0.47, 0.61, 0.81, -0.04, -1.06, 0.19, -1.39, -0.9, 0.5, -0.04, -1.33, 0.6, -0.46, -0.03, 0.23, 0.47, -0.36, -0.28, 0.14, 0.95, 0.47, 0.2, 0.82, 0.24, 0.88, -1.31, 0.23, -0.16, 0.06, -0.57, -1.03, -0.95, -0.69, -0.31, -0.72, -1.26, 0.43, 0.18, 0.35, -0.72, -0.06, -0.11, 0.54, -0.47, -0.15, 0.62, -0.19, 0.41, 0.03, 0.4, -1.12, -0.82, 1.49, -0.16, -0.56, -0.08, 0.07, 0.52, -0.36, -0.31, 1.08, 0.98, -0.85, 1.32, 0.36, -0.13, 0.39, 0.85, 0.44, -1.04, 1.24, 0.58, -0.19, 0.4, -0.04, 0.69, -0.5, -0.93, 0.57, 0.24, 0.1, 0.33, -0.21, -0.65, -0.01, 0.78, -0.11, 1.12, -0.13, -1.29, -0.34, 1.14, -0.67, 0.05, -0.34, 0.08, 0.57, -0.76, 0.6, -0.38, 0.16, 0.51, 0.24, -0.17, 0.8, 0.48, -1.38, -0.09, 0.21, 0.11, -0.65, 0.22, 0.55, 1.16, 0.26, 0.71, -0.01, 0.27, -0.3, -0.76, 0.05, 0.0, 0.21, -1.12, 0.41, 0.64, -1.99, -0.36, -0.64, 0.25, 0.92, -0.57, -1.15, -0.34, 0.18, 0.67, -0.17, 0.23, -0.99, -0.8, -0.85, 0.72, 1.08, -0.85, 0.47, 1.35, -0.84, -0.19, -0.31, 0.32, 0.37, 0.26, 0.03, -0.18, -0.78, -0.42, 0.31, 0.85, -0.2, -0.01, 0.72, 0.22, -0.81, 0.24, 0.83, 1.18, 0.85, 0.36, 0.49, 0.41, -0.39, 0.04, 0.09, 1.31, 0.43, -0.83, 1.03, 0.12, -0.47, 0.14, -0.43, -0.39, 0.82, 0.23, 0.03, 1.61, -0.82, 0.18, 0.25, 0.23, 0.94, 1.34, -0.46, -0.53, -0.38, -0.81, -0.57, 0.41, -0.37, -0.4, 0.12, -0.08, 0.87, 0.02, 0.54, -0.39, 0.69, 1.29, 0.69, 0.26, 0.21, 0.18, -0.56, 0.67, -0.41, -0.04, 0.14, -1.0, -0.01, -0.42, -0.33, 0.26, -0.01, 0.08, -1.11, 0.02, -0.84, 0.65, 0.19, -0.11, -1.08, 0.92, 0.23, -0.22, 0.39, 0.31, 0.82, 0.13, 1.03, -0.29, -0.46, -0.44, -0.16, -0.28, 1.2, -0.4, -0.26, 0.39, 0.52, 0.09, 0.07, -0.03, 1.24, 0.55, 0.38, 1.06, 0.05, -0.22, 0.29, 0.95, 0.78, 0.81]


similarity_value = 0.85
maxiter = 10000
window_size = 8
step_window = 4
#step_window = 6
interval_width = 0.0
random_step = 0

deep_level = 1000

"""constraints = []
for i in range(0, N, 4):
    constraints.append(i)"""

vectorial_subspace = VectorialSubspaceDeep(
    deep_level=deep_level,
    metric="cosine",
    threshold=similarity_value,
    maxiter=maxiter,
    interval_width=interval_width,
    window_size=window_size,
    window_step=step_window,
    method="COBYLA",
    intervals_reducing_type="disjunction",
    random_steps=random_step,
    add_penalty=True,
    expand_factor=0,
    verbose=1
)

vectorial_subspace.optimize(tensor_target)
pprint(vectorial_subspace.intervals)
intervals = vectorial_subspace.intervals

"""differences = [tuple_[0][1] - tuple_[0][0] for tuple_ in vectorial_subspace.intervals]
mean_differences = mean(differences)"""
#print(differences)
#print(mean_differences)

flag = True
tensor_target = np.array([tensor_target, ])
num_error = 0
num_iter = 0
min_value = 100
max_value = 0
print()
for _ in range(100000):
    num_iter += 1
    random_tensor = []

    intervals_ = random.choice(intervals)

    for ranges in intervals_:
        range_ = random.choice(ranges)
        # random_value_range = random.uniform(range_[0], range_[1])
        mode = (range_[0] * 1 + range_[1] * 1) / 2
        random_value_range = random.triangular(range_[0], range_[1], mode=mode)
        random_tensor.append(round(random_value_range, 2))
    random_tensor = np.array([random_tensor, ])
    sim_random = cosine_similarity(tensor_target, random_tensor)[0][0]

    if sim_random < min_value:
        min_value = sim_random
    if sim_random > max_value:
        max_value = sim_random
    if round(sim_random, 2) < similarity_value:
        num_error += 1

    #print(random_tensor, sim_random)
    percentage_error = (num_error * 100) / num_iter
    print(
        f"\rNumbers generated vectors {num_iter}\t"
        f"Error percentage {round(percentage_error, 2)}%\t"
        f"Min value: {round(min_value, 2)}\t"
        f"Max value: {round(max_value, 2)}",
        end=""
    )

"""print(tensor_target.tolist())
bool_list = [True if intervals[i][0][0] <= tensor_target[0][i] <= intervals[i][0][1] else False for i in range(768)]
print(bool_list)"""
"""for i in range(N):
    random_tensor[0][i] /= 10"""
"""print()
print(sim_random)
for i in range(0, N, 4):
    bfac_index = i
    x_index = i + 1
    y_index = i + 2
    z_index = i + 3
    print(random_tensor[0][bfac_index], random_tensor[0][x_index], random_tensor[0][y_index], random_tensor[0][z_index])"""


if to_plot:

    plt.figure(figsize=(8, 6))

    # Plotta la regione (rettangolo)
    plt.fill_betweenx(intervals[1][0], intervals[0][0][0], intervals[0][0][1], color='lightblue', alpha=0.5, label='Regione')

    # Plotta il punto
    plt.plot(tensor_target[0][0], tensor_target[0][1], 'ro', label='Punto')

    min_ = 100
    max_ = -100
    for interval_list in intervals:
        min__ = min(interval_list, key=lambda x: x[0])[0]
        if min__ < min_:
            min_ = min__
        max__ = max(interval_list, key=lambda x: x[1])[1]
        if max__ > max_:
            max_ = max__
    print()
    print(min_, max_)

    plt.xlim(min_ - 1, max_ + 1)
    plt.ylim(min_ - 1, max_ + 1)
    plt.axhline(0, color='black',linewidth=0.5, ls='--')
    plt.axvline(0, color='black',linewidth=0.5, ls='--')
    plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
    plt.title('Grafico del Punto e della Regione')
    plt.xlabel('Asse x')
    plt.ylabel('Asse y')
    plt.legend()

    # Mostra il grafico
    plt.show()

