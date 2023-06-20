from typing import List
import os
import tensorflow as tf
import numpy as np
from utils.analysis import calculate_classifications
import math
import random

def create_trainings_data(all_data: dict, sub_dataset_percentage: float,
                          number_of_subdatasets: int, sort_key: str, subdataset_size: int) -> dict:
    subdatasets = create_all_subdatasets(all_data=all_data,
                                         sub_dataset_percentage=sub_dataset_percentage,
                                         number_of_subdatasets=number_of_subdatasets,
                                         subdataset_size=subdataset_size,
                                         sort_key=sort_key
                                         )
    return {
            'bilstm_input': np.array([np.array(get_bilstm_input_from_sorted_subdataset(
                sub_dataset=subdataset, sort_key=sort_key)) for subdataset in subdatasets]),
            "statistics": np.array([np.array(subdataset['statistics']) for subdataset in subdatasets]),
            "true_prevalence": np.array([np.array(subdataset['true_prevalence']) for subdataset in subdatasets])
        }


def create_all_subdatasets(all_data: dict, sub_dataset_percentage: float,
                           number_of_subdatasets: int, subdataset_size: int, sort_key: str):
    return [create_subdataset(all_data=all_data,
                              sub_dataset_percentage=sub_dataset_percentage, subdataset_size=subdataset_size, sort_key=sort_key) for
            subdataset in range(0, number_of_subdatasets)]


def create_subdataset(all_data: dict, sub_dataset_percentage: float, subdataset_size: int, sort_key: str) -> dict:
    ##y_prob = [ [p1, p2, p3], [p1, p2, p3], ...]
    n_total = len(all_data['y_true'])
    n_sub_dataset = subdataset_size
    sub_dataset_indeces = random.sample(range(0, n_total), n_sub_dataset)
    sub_dataset = {
        'y_true': [all_data['y_true'][index] for index in sub_dataset_indeces],
        'true_prevalence': get_prevalences_for_subdataset(y_true=[all_data['y_true'][index] for index in sub_dataset_indeces]),
        'y_pred': [all_data['y_pred'][index] for index in sub_dataset_indeces],
        'probabilities': [all_data['probabilities'][index] for index in sub_dataset_indeces],
        'embeddings': [all_data['embeddings'][index] for index in sub_dataset_indeces],
        'entropy': [calculate_entropy(probabilities=all_data['probabilities'][index]) for index
                    in sub_dataset_indeces],
        'gini': [calculate_gini_coefficient(probabilities=all_data['probabilities'][index]) for
                 index
                 in sub_dataset_indeces],
        'classification_error': [
            calculate_classification_error(probabilities=all_data['probabilities'][index]) for
            index
            in sub_dataset_indeces]
    }
    sub_dataset = add_statistics_to_subdataset(sub_dataset=sub_dataset)
    sub_dataset = sort_subdataset_on_key(sort_key=sort_key, sub_dataset=sub_dataset)
    return sub_dataset


def add_statistics_to_subdataset(sub_dataset: dict) -> dict:
    statistics = calculate_classifications(
        ty_train=sub_dataset['y_true'],
        py_train=sub_dataset['y_pred'],
        prob_train=sub_dataset['probabilities']
    )
    sub_dataset['statistics'] = statistics
    return sub_dataset


def calculate_entropy(probabilities: List[float]) -> float:
    return -sum([p * math.log(p) for p in probabilities])


def calculate_gini_coefficient(probabilities: List[int]) -> float:
    return sum([p * (1 - p) for p in probabilities])


def calculate_classification_error(probabilities: List[int]) -> float:
    return 1 - max(p for p in probabilities)


def sort_subdataset_on_key(sort_key: str, sub_dataset: dict) -> dict:
    sorted_indeces = sorted(range(250),
                            key=lambda k: sub_dataset[sort_key][k])
    for key, value in sub_dataset.items():
        if not key == 'true_prevalence' and not key == 'statistics':
            sub_dataset[key] = [value[index] for index in sorted_indeces]

    return sub_dataset


def get_bilstm_input_from_sorted_subdataset(sub_dataset: dict, sort_key: str) -> List[List[float]]:
    return [
        [sub_dataset[sort_key][index]] + sub_dataset['probabilities'][index] +
        sub_dataset['embeddings'][index] for index in
        range(0, len(sub_dataset[sort_key]))
    ]


def get_prevalences_for_subdataset(y_true: List[int]) -> List[float]:
    negative_sentiment = 0
    neutral_sentiment = 0
    positive_sentiment = 0
    for y in y_true:
        if y == 0:
            negative_sentiment += 1
        if y == 1:
            neutral_sentiment += 1
        if y == 2:
            positive_sentiment += 1
    n = len(y_true)
    return [negative_sentiment / n, neutral_sentiment / n, positive_sentiment / n]
