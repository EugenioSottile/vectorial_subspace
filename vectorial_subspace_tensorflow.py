import random
import time
from pprint import pprint
from tkinter.ttk import Progressbar
from typing import List, Tuple, Optional
import numpy as np
from numpy import arange
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf


class VectorialSubspaceTF:

    def __init__(
            self,
            epochs: int = 1,
            learning_rate: float = 0.1,
            metric: str = "cosine",
            threshold: float = 0.95,
            window_size: int = 16,
            window_step: int = 8,
            interval_width: float = 0.5,
            intervals_reducing_type: str = "disjunction",
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

        self.metric = metric
        self.threshold = threshold
        self.epochs = epochs
        self.window_size = window_size
        self.window_step = window_step
        self.interval_width = interval_width
        self.learning_rate = learning_rate
        self.intervals_reducing_type = intervals_reducing_type
        self.random_step = random_steps
        self.shift_value = shift_value
        self.add_penalty = add_penalty
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

        #self.tensor = tensor
        len_tensor = len(tensor)
        self.__len_tensor = len_tensor
        if self.verbose == 1:
            progress_bar = self.__ProgressBar(len_tensor)
            if self.window_size != self.window_step:
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
            interval_list = intervals_reduced_list[i]

            min_ = min(interval_list, key=lambda x: x[0])[0]
            max_ = max(interval_list, key=lambda x: x[1])[1]

            if min_ > tensor[i]:
                tuple_min = (float(tensor[i]), min_)
                interval_list.append(tuple_min)
            elif max_ < tensor[i]:
                tuple_max = (max_, float(tensor[i]))
                interval_list.append(tuple_max)

        if self.random_step > 0:
            intervals_reduced_list = self.__random_tuning(
                intervals_reduced_list,
                tensor=tensor
            )

        reduced_intervals = self.__reduce_intervals(
            intervals_reduced_list,
            intervals_reducing_type=self.intervals_reducing_type
        )
        """if self.expand_factor > 0:
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
            expanded_intervals = reduced_intervals"""



        self.intervals = reduced_intervals

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

        #print(minimized_vector1)
        #print(minimized_vector2)

        for i in range(self.window_size):
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
        x = tf.Variable(initial_value=tf.convert_to_tensor(tensor, dtype=tf.float32))

        learning_rate = self.learning_rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        epochs = self.epochs
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                loss = self.__objective_function(x)
            grads = tape.gradient(loss, [x])
            optimizer.apply_gradients(zip(grads, [x]))
        return x

    def __objective_function(self, tensor):

        # Calcola la similarità
        tensor1_ = tf.convert_to_tensor(self.__tensor_sliced, dtype=tf.float32)
        tensor2_ = tensor

        if self.metric == "cosine":
            similarity = 1 - tf.keras.losses.cosine_similarity(tensor1_[None, :], tensor2_[None, :])
        elif self.metric == "euclidean":
            similarity = tf.norm(tensor1_ - tensor2_)
        else:
            raise ValueError("Metric non supportata")

        # Penalità
        penalty = 0
        if self.add_penalty:
            if self.__direction == 0:
                for i in range(self.window_size):
                    if tensor[i] >= self.__tensor_sliced[i]:
                        difference = tensor[i] - self.__tensor_sliced[i]
                        penalty += difference + 0.1
                        if difference < self.interval_width:
                            penalty += (self.interval_width - difference)
            elif self.__direction == 1:
                for i in range(self.window_size):
                    if tensor[i] <= self.__tensor_sliced[i]:
                        difference = self.__tensor_sliced[i] - tensor[i]
                        penalty += difference + 0.1
                        if difference < self.interval_width:
                            penalty += (self.interval_width - difference)

        return tf.abs(similarity - self.threshold) + penalty

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