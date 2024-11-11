import itertools
import random
import time
from pprint import pprint
from typing import List, Tuple
import numpy as np
from absl.logging import level_warn
from numpy import arange
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity


class VectorialSubspaceDeep:

    def __init__(
            self,
            deep_level: int = 1,
            metric: str = "cosine",
            threshold: float = 0.95,
            maxiter: int = 1000,
            window_size: int = 16,
            window_step: int = 8,
            interval_width: float = 0.5,
            method: str = "COBYLA",
            intervals_reducing_type: str = "disjunction",
            expand_factor: float = 1,
            fixed_constraints: Tuple = (),
            random_steps: int = 1000,
            shift_value: float = 1.0,
            add_penalty: bool = True,
            verbose: int = 1
    ):

        """
        Constructor of the class. Initializes parameters for computation.

        :param metric: The type of metric to use (e.g., "cosine").
        :param threshold: Threshold for the minimum similarity.
        :param minimization_step: Step size for the optimization algorithm.
        :param window_size: The size of the data window to consider.
        :param window_step: The step size for moving between windows.
        :param method: The optimization method to use (e.g., "COBYLA").
        :param verbose: Verbosity level (0 = silent, 1 = informative output).
        """

        self.deep_level = deep_level
        self.metric = metric
        self.threshold = threshold
        self.maxiter = maxiter
        self.window_size = window_size
        self.window_step = window_step
        self.interval_width = interval_width
        self.method = method
        self.intervals_reducing_type = intervals_reducing_type
        self.expand_factor = expand_factor
        self.fixed_constraints = fixed_constraints
        self.random_step = random_steps
        self.shift_value = shift_value
        self.add_penalty = add_penalty
        self.verbose = verbose

        #self.tensor = np.array([])
        self.intervals = [[[(), ], ],]

        self.__minimization_step_count = 0
        self.__tensor_sliced = np.array([])
        self.__start = 0
        self.__end = 0
        self.__direction = 0
        self.__len_tensor = 0

    def optimize(
            self,
            tensor: np.ndarray = np.array([])
    ):

        #self.tensor = tensor
        len_tensor = len(tensor)
        self.__len_tensor = len_tensor
        """if self.verbose == 1:
            progress_bar = self.__ProgressBar(len_tensor)
            if self.window_size != self.window_step:
                progress_bar.update(self.window_step)"""

        level_constraints = []
        """level_constraints = [
            combination 
            for r in range(1, self.__len_tensor) 
            for combination in itertools.combinations(range(self.__len_tensor), r)
            if 
        ]"""
        count_combination = 0
        for r in range(1, self.__len_tensor):
            for combination in itertools.combinations(range(self.__len_tensor), r):
                level_constraints.append(combination)
                count_combination += 1
                if count_combination >= self.deep_level:
                    break
            if count_combination >= self.deep_level:
                break
        level_constraints_0 = [tuple(range(self.__len_tensor)), ]
        level_constraints = level_constraints_0 + level_constraints

        num_constraints = len(level_constraints)
        levels = self.deep_level
        if self.deep_level > num_constraints:
            levels = num_constraints

        intervals = []
        if self.verbose == 1:
            progress_bar = self.__ProgressBar(levels)
        for level in range(levels):

            self.__level_constraints = level_constraints[level]

            intervals_reduced_list_level = [list() for _ in range(len_tensor)]

            """step = self.window_step
            if self.window_step == 0 and self.window_size == self.__len_tensor:
                step = 1"""

            for i in range(0, len_tensor - self.window_size + 1, self.window_step):
                self.__start = i
                self.__end = i + self.window_size

                self.__tensor_sliced = tensor[self.__start: self.__end]

                to_do = any([index in self.__level_constraints for index in range(self.__start, self.__end)])

                if to_do:
                    intervals_slice = self.__get_window_subspace(self.__tensor_sliced)

                else:
                    intervals_slice = [
                        [
                            (
                                float(
                                    self.__tensor_sliced[index] - 0.01
                                ),
                                float(
                                    self.__tensor_sliced[index] + 0.01
                                )
                            )
                        ]
                        for index in range(self.window_size)
                    ]

                for j in range(self.__start, self.__end):
                    index_base = j - self.__end
                    intervals_reduced_list_level[j].extend(intervals_slice[index_base])

                """if self.verbose == 1:
                    progress_bar.update(self.window_step)"""

            if self.__end < len_tensor:
                start = len_tensor - self.window_size
                end = len_tensor
                self.__tensor_sliced = tensor[self.__start: self.__end]
                intervals_slice = self.__get_window_subspace(self.__tensor_sliced)

                for j in range(start, end):
                    index_base = j - start
                    intervals_reduced_list_level[j].extend(intervals_slice[index_base])

                """if self.verbose == 1:
                    progress_bar.update(self.window_step)"""

            for i in range(len_tensor):
                interval_list = intervals_reduced_list_level[i]

                min_ = min(interval_list, key=lambda x: x[0])[0]
                max_ = max(interval_list, key=lambda x: x[1])[1]

                if min_ > tensor[i]:
                    tuple_min = (float(tensor[i]), min_)
                    interval_list.append(tuple_min)
                elif max_ < tensor[i]:
                    tuple_max = (max_, float(tensor[i]))
                    interval_list.append(tuple_max)

            reduced_intervals_level = self.__reduce_intervals(
                intervals_reduced_list_level,
                intervals_reducing_type=self.intervals_reducing_type
            )

            """print(level_constraints[level])
            pprint(reduced_intervals_level[0][0])"""


            intervals.append(reduced_intervals_level)
            if self.verbose == 1:
                progress_bar.update(1)

        """if self.random_step > 0:
            intervals_reduced_list = self.__random_tuning(
                intervals_reduced_list,
                tensor=tensor
            )

        reduced_intervals = self.__reduce_intervals(
            intervals_reduced_list,
            intervals_reducing_type=self.intervals_reducing_type
        )"""

        self.intervals = intervals

    def __get_window_subspace(
            self,
            tensor: np.ndarray = np.array([])
    ):
        len_tensor = len(tensor)
        intervals = [list() for _ in range(len_tensor)]

        self.__direction = 0
        minimized_vector1 = self.__minimize_vector(tensor)

        self.__direction = 1
        minimized_vector2 = self.__minimize_vector(tensor)

        for i in range(self.window_size):

            global_index = i + self.__start
            if global_index in self.fixed_constraints:
                interval = [float(tensor[i]), float(tensor[i])]
            else:
                interval = [float(minimized_vector1[i]), float(minimized_vector2[i])]
            interval.sort()
            interval = tuple(interval)
            intervals[i].append(interval)
        return intervals

    def __reduce_intervals(
            self,
            intervals: List[List[Tuple[float]]],
            intervals_reducing_type: str = "disjunction"
    ) -> List[List[Tuple[float]]]:
        reduced_intervals = [[(), ], ]
        if intervals_reducing_type == "disjunction":
            reduced_intervals = self.__reduce_intervals_disjunction(intervals)
        elif intervals_reducing_type == "union":
            reduced_intervals = self.__reduce_intervals_union(intervals)
        else:
            pass  # launch exception
        return reduced_intervals

    def __reduce_intervals_disjunction(
            self,
            intervals: List[List[Tuple[float]]],
    ) -> List[List[Tuple[float]]]:
        reduced_intervals = []
        for interval_list in intervals:
            interval_list.sort(key=lambda x: x[0])
            merged_intervals = []
            for interval in interval_list:
                if not merged_intervals or merged_intervals[-1][1] < interval[0]:
                    merged_intervals.append(interval)
                else:
                    merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], interval[1]))
            reduced_intervals.append(merged_intervals)
        return reduced_intervals

    def __reduce_intervals_union(
            self,
            intervals: List[List[Tuple[float]]],
    ) -> List[List[Tuple[float]]]:
        reduced_intervals = []
        for interval_list in intervals:
            min_ = min(interval_list, key=lambda x: x[0])[0]
            max_ = max(interval_list, key=lambda x: x[1])[1]
            merged_intervals = (min_, max_)
            merged_intervals_list = [merged_intervals, ]
            reduced_intervals.append(merged_intervals_list)
        return reduced_intervals

    def __random_tuning(
            self,
            intervals: List[List[Tuple[float]]],
            tensor: np.ndarray = np.array([]),
    ) -> List[List[Tuple[float]]]:

        random.seed(self.__len_tensor)
        intervals_ = intervals.copy()
        tensor = np.array([tensor, ])
        if self.verbose == 1:
            progress_bar = self.__ProgressBar(self.random_step, string_message="Random tuning")
        for i in range(self.random_step):
            random_tensor = []
            for k in range(self.__len_tensor):
                interval_ = random.choice(intervals_[k])
                #random.seed(interval_[0] - self.shift_value + interval_[1] + self.shift_value)
                random_value_interval_ = random.uniform(interval_[0] - self.shift_value, interval_[1] + self.shift_value)
                """mode = random.choice(interval_)
                random_value_interval_ = random.triangular(
                    interval_[0] - self.shift_value,
                    interval_[1] + self.shift_value,
                    mode=mode
                )"""
                random_tensor.append(random_value_interval_)
            random_tensor = np.array([random_tensor, ])
            similarity = cosine_similarity(tensor, random_tensor)[0][0]
            if similarity >= self.threshold:
                for j in range(self.__len_tensor):
                    flag_not_included = False
                    for interval in intervals[j]:
                        if not (interval[0] <= random_tensor[0][j] <= interval[1]):
                            flag_not_included = True
                            break
                    if flag_not_included:
                        value_left = float(random_tensor[0][j]) - self.shift_value
                        value_right = float(random_tensor[0][j]) + self.shift_value
                        tuple_new_interval = (value_left, value_right)
                        intervals_[j].append(tuple_new_interval)
            if self.verbose == 1:
                progress_bar.update(1)
        return intervals_



    def __minimize_vector(
            self,
            tensor: np.ndarray = np.array([])
    ):

        if self.__direction == 0:

            constraints_limits = [
                {
                    'type': 'ineq',
                    'fun': lambda x, i=i: tensor[i] - x[i] - self.interval_width
                }
                for i in range(len(tensor)) if i in self.__level_constraints
            ]

        elif self.__direction == 1:

            constraints_limits = [
                {
                    'type': 'ineq',
                    'fun': lambda x, i=i: x[i] - tensor[i] + self.interval_width
                }
                for i in range(len(tensor)) if i in self.__level_constraints
            ]

        """constraints_limits_equals = [
            {
                'type': 'eq',
                'fun': lambda x, i=i: tensor[i] - x[i]
            }
            for i in range(len(tensor)) if i not in self.__level_constraints
        ]
        constraints_limits = constraints_limits + constraints_limits_equals"""

        constraints = constraints_limits

        result = minimize(
            self.__objective_function,
            tensor,
            method=self.method,
            options={
                'maxiter': self.maxiter,
                'tol': 0.01,
                #'rhobeg': 0.4,
                #'rhoend': 1e-6
            },
            constraints=constraints
        )
        return result.x

    def __objective_function(
            self,
            tensor: np.ndarray = np.array([])
    ):


        for index in self.fixed_constraints:
            if self.__start <= index < self.__end:
                sliced_index = index - self.__start
                tensor[sliced_index] = self.__tensor_sliced[sliced_index]

        similarity = 0
        if self.metric == "cosine":
            tensor1_ = np.array([self.__tensor_sliced, ])
            tensor2_ = np.array([tensor, ])
            similarity = cosine_similarity(tensor1_, tensor2_)[0][0]
        elif self.metric == "euclidian":
            tensor1_ = [self.__tensor_sliced, ]
            tensor2_ = tensor
            similarity = np.linalg.norm(tensor1_ - tensor2_)
        else:
            pass  # launch exception (to define)

        penalty = 0
        if self.add_penalty:
            if self.__direction == 0:
                for i in range(self.window_size):
                    global_index = i + self.__start
                    if global_index in self.__level_constraints:
                        if tensor[i] >= self.__tensor_sliced[i]:
                            difference = (tensor[i] - self.__tensor_sliced[i])
                            penalty += difference * 10
                            if difference < self.interval_width:
                                penalty += (self.interval_width - difference)
                    else:
                        difference = abs(tensor[i] - self.__tensor_sliced[i])
                        if difference >= 0.1:
                            penalty += 0.1 + (difference * 1)
            elif self.__direction == 1:
                for i in range(self.window_size):
                    global_index = i + self.__start
                    if global_index in self.__level_constraints:
                        if tensor[i] <= self.__tensor_sliced[i]:
                            difference = (self.__tensor_sliced[i] - tensor[i])
                            penalty += difference * 10
                            if difference < self.interval_width:
                                penalty += (self.interval_width - difference)
                    else:
                        difference = abs(tensor[i] - self.__tensor_sliced[i])
                        if difference >= 0.1:
                            penalty += 0.1 + (difference * 1)

        return abs(similarity - self.threshold) + penalty

    class __ProgressBar:
        def __init__(self, length: float, string_message=None, length_bar=50):
            self.completion_bar = 0
            self.completed_bar = 0
            self.length_bar = length_bar
            self.initial_time = time.time()
            if string_message is None:
                self.string_progress = "Progress:"
            else:
                self.string_progress = string_message
            self.reset(length)
            percentage = round((self.completion_bar / self.completed_bar) * 100, 2)
            time_ = 0.0
            bar = "[" + (self.length_bar * "-") + "]"
            print(f'\r{self.string_progress}\t{bar}\t{percentage}%\tTime: {time_}s', end="")

        def update(self, completion: float, print_mode="print"):
            self.completion_bar += completion
            if print_mode.__eq__("print"):
                percentage = round((self.completion_bar / self.completed_bar) * 100, 2)
                end_str = ""
                if percentage == 100:
                    end_str = "\n"
                num_bar = round((percentage * self.length_bar) / 100)
                actual_time = time.time()
                time_ = actual_time - self.initial_time
                time_ = round(time_, 2)
                self.time_ = actual_time
                bar = "[" + (num_bar * "#") + ((self.length_bar - num_bar) * "-") + "]"
                print(f'\r{self.string_progress}\t{bar}\t{percentage}%\tTime: {time_}s', end=end_str)

        def reset(self, length: float):
            self.completion_bar = 0
            self.completed_bar = length
