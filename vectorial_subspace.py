import random
import time
from pprint import pprint
from typing import List, Tuple, Optional
import numpy as np
from numpy import arange
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity


class VectorialSubspace:

    def __init__(
            self,
            metric: str = "cosine",
            threshold: float = 0.95,
            minimization_step: float = 0.01,
            window_size: int = 16,
            window_step: int = 8,
            method: str = "COBYLA",
            intervals_reducing_type: str = "disjunction",
            expand_factor: float = 1,
            fixed_constraints: Tuple = (),
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
        self.threshold = threshold
        self.minimization_step = minimization_step
        self.window_size = window_size
        self.window_step = window_step
        self.method = method
        self.intervals_reducing_type = intervals_reducing_type
        self.expand_factor = expand_factor
        self.fixed_constraints = fixed_constraints
        self.verbose = verbose

        #self.tensor = np.array([])
        self.intervals = [[(), ], ]

        self.__minimization_step_count = 0
        self.__tensor_sliced = np.array([])
        self.__start = 0
        self.__end = 0

    def optimize(
            self,
            tensor: np.ndarray = np.array([])
    ):

        #self.tensor = tensor
        len_tensor = len(tensor)
        if self.verbose == 1:
            progress_bar = self.__ProgressBar(len_tensor)
            progress_bar.update(self.window_step)

        intervals_reduced_list = [list() for _ in range(len_tensor)]
        for i in range(0, len_tensor - self.window_size + 1, self.window_step):
            self.__start = i
            self.__end = i + self.window_size

            self.__tensor_sliced = tensor[self.__start: self.__end]
            intervals_slice = self.__get_window_subspace(self.__tensor_sliced)

            for j in range(self.__start, self.__end):
                index_base = j - self.__end
                intervals_reduced_list[j].extend(intervals_slice[index_base])

            if self.verbose == 1:
                progress_bar.update(self.window_step)

        if self.__end < len_tensor:
            start = len_tensor - self.window_size
            end = len_tensor
            self.__tensor_sliced = tensor[self.__start: self.__end]
            intervals_slice = self.__get_window_subspace(self.__tensor_sliced)

            for j in range(start, end):
                index_base = j - start
                intervals_reduced_list[j].extend(intervals_slice[index_base])

            if self.verbose == 1:
                progress_bar.update(self.window_step)

        for i in range(len_tensor):
            min_tuple = min(intervals_reduced_list[i], key=lambda x: x[0])
            max_tuple = max(intervals_reduced_list[i], key=lambda x: x[1])
            if tensor[i] < min_tuple[0]:
                tuple_tensor = (tensor[i] - (max_tuple[1] - min_tuple[0]), tensor[i])
                intervals_reduced_list[i].append(tuple_tensor)
                """tuple_tensor = (tensor[i], tensor[i])
                intervals_reduced_list[i].append(tuple_tensor)"""
            elif tensor[i] > max_tuple[1]:
                tuple_tensor = (tensor[i], tensor[i] + (max_tuple[1] - min_tuple[0]))
                intervals_reduced_list[i].append(tuple_tensor)

            """if tensor[i] < min_tuple[0]:
                tuple_tensor = list(min_tuple)
                tuple_tensor[1] = tensor[i]
                tuple_tensor[0] -= abs(tuple_tensor[1] - tuple_tensor[0])
                tuple_tensor = tuple(tuple_tensor)
                intervals_reduced_list[i].append(tuple_tensor)
            elif tensor[i] > max_tuple[1]:
                tuple_tensor = list(max_tuple)
                tuple_tensor[0] = tensor[i]
                tuple_tensor[1] += abs(tuple_tensor[1] - tuple_tensor[0])
                tuple_tensor = tuple(tuple_tensor)
                intervals_reduced_list[i].append(tuple_tensor)"""

        reduced_intervals = self.__reduce_intervals(
            intervals_reduced_list,
            intervals_reducing_type=self.intervals_reducing_type
        )
        if self.expand_factor > 0:
            expanded_intervals = [
                [
                    self.__expand_interval(
                        interval[0],
                        self.expand_factor,
                        self.threshold
                    ),
                ]
                for interval in reduced_intervals
            ]
        else:
            expanded_intervals = reduced_intervals

        """for i in range(len_tensor):
            for j in range(len(expanded_intervals[i])):
                expanded_intervals[i][j] = list(expanded_intervals[i][j])
                if expanded_intervals[i][j][0] > tensor[i]:
                    expanded_intervals[i][j][0] = tensor[i]
                elif expanded_intervals[i][j][1] < tensor[i]:
                    expanded_intervals[i][j][1] = tensor[i] + 1
                expanded_intervals[i][j] = tuple(expanded_intervals[i][j])"""

        self.intervals = expanded_intervals

    def __get_window_subspace(
            self,
            tensor: np.ndarray = np.array([])
    ):
        len_tensor = len(tensor)
        intervals = [list() for _ in range(len_tensor)]

        start = self.threshold + self.minimization_step
        end = 1 + self.minimization_step
        arange_ = arange(start, end, self.minimization_step)
        for _ in arange_:
            minimized_vector1 = self.__minimize_vector(tensor)
            self.__minimization_step_count += 1
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
        self.__minimization_step_count = 0

        return intervals

    """
            for index in self.fixed_constraints:
            if self.__start <= index < self.__end:
                sliced_index = index - self.__start
                tensor[sliced_index] = self.__tensor_sliced[sliced_index]
    """

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

    def __minimize_vector(
            self,
            tensor: np.ndarray = np.array([])
    ):
        #tensors = np.concatenate((tensor, tensor), axis=0)
        method = self.method
        result = minimize(
            self.__objective_function,
            tensor,
            method=method
        )
        return result.x

    def __objective_function(
            self,
            tensor: np.ndarray = np.array([])
    ):

        threshold = (self.__minimization_step_count * self.minimization_step) + self.threshold

        if threshold == 1:
            threshold = 0.999

        #print(self.fixed_constraints)
        for index in self.fixed_constraints:
            if self.__start <= index < self.__end:
                sliced_index = index - self.__start
                tensor[sliced_index] = self.__tensor_sliced[sliced_index]

        """for i in range(len(tensor)):
            tensor[i] += random.uniform(-0.1, 0.1)"""

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

        return abs(similarity - threshold)

    def __expand_interval(
            self,
            interval: Tuple[float],
            factor: float,
            similarity: float,
    ) -> Tuple[float] :
        min_val, max_val = interval
        interval_length = max_val - min_val

        if interval_length == 0:
            interval_length = 1e-9

        expansion = (factor / interval_length) * (1 - similarity)
        min_val -= expansion
        max_val += expansion

        return min_val, max_val

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
