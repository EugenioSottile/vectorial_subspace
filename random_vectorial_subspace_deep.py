import itertools
import random
import time
from pprint import pprint
from tkinter.ttk import Progressbar
from typing import List, Tuple, Optional
import numpy as np
from numpy import arange
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity


class RandomVectorialSubspaceDeep:

    def __init__(
            self,
            metric: str = "cosine",
            deep_level: int = 1,
            threshold: float = 0.95,
            intervals_reducing_type: str = "disjunction",
            random_step: int = 1000,
            shift_value: float = 1.0,
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

        self.metric = metric
        self.deep_level = deep_level
        self.threshold = threshold
        self.intervals_reducing_type = intervals_reducing_type
        self.random_step = random_step
        self.shift_value = shift_value
        self.verbose = verbose

        #self.tensor = np.array([])
        self.intervals = [[(), ], ]

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

        len_tensor = len(tensor)
        self.__len_tensor = len_tensor

        level_constraints = []
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

        intervals_ = []

        if self.verbose == 1:
            progress_bar = self.__ProgressBar(levels)
        for level in range(levels):

            self.__level_constraints = level_constraints[level]

            intervals = []
            for value in tensor:
                list_intervals = []
                interval = (float(value - self.shift_value), float(value + self.shift_value))
                list_intervals.append(interval)
                intervals.append(list_intervals)


            intervals = self.__random_tuning(intervals, tensor)

            reduced_intervals = self.__reduce_intervals(
                intervals,
                intervals_reducing_type=self.intervals_reducing_type
            )

            intervals_.append(reduced_intervals)

            if self.verbose == 1:
                progress_bar.update(1)

        self.intervals = intervals_

    def __random_tuning(
            self,
            intervals: List[List[Tuple[float]]],
            tensor: np.ndarray = np.array([]),
    ) -> List[List[Tuple[float]]]:

        random.seed(self.__len_tensor)
        intervals_ = intervals.copy()
        tensor = np.array([tensor, ])
        """if self.verbose == 1:
            progress_bar = self.__ProgressBar(self.random_step, string_message="Random tuning")"""
        for i in range(self.random_step):
            random_tensor = []
            for k in range(self.__len_tensor):
                if k in self.__level_constraints:
                    interval_ = random.choice(intervals_[k])
                    # random.seed(interval_[0] - self.shift_value + interval_[1] + self.shift_value)
                    random_value_interval_ = random.uniform(interval_[0] - self.shift_value,
                                                            interval_[1] + self.shift_value)
                    """mode = random.choice(interval_)
                    random_value_interval_ = random.triangular(
                        interval_[0] - self.shift_value,
                        interval_[1] + self.shift_value,
                        mode=mode
                    )"""
                    random_tensor.append(random_value_interval_)
                else:
                    random_tensor.append(tensor[0][k])
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
            """if self.verbose == 1:
                progress_bar.update(1)"""
        return intervals_

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
