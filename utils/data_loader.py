import csv
import random
import os
from typing import Sequence
import dask.dataframe as dd
import numpy as np
import pandas as pd
import tensorflow as tf


def csv_to_input(inpath, column_names: Sequence[str]):
    df = dd.read_csv(inpath).compute()[column_names].fillna('')
    return df.T.values


def load_data(doc_path: str = None, asp_path: str = None):
    """
    Function that takes in a path to the document data and to the aspect data and returns a dictionary
    suitable for fitting the model.

    Args:
        doc_path: path to the document dataset
        asp_path: path to the aspect dataset

    Returns:
        x,y: dictionaries suitable for training and validating models

    """
    if doc_path is not None and asp_path is not None:
        doc_x, doc_y = csv_to_input(doc_path, ['text', 'polarity'])
        *asp_x, asp_y = csv_to_input(asp_path, ['context_left', 'target', 'context_right', 'polarity'])
        x = {'asp': asp_x, 'doc': doc_x}
        y = {'asp': tf.one_hot(asp_y + 1, 3, dtype='int32'), 'doc': tf.one_hot(doc_y + 1, 3, dtype='int32')}
        return x, y

    if doc_path is not None and asp_path is None:
        doc_x, doc_y = csv_to_input(doc_path, ['text', 'polarity'])
        x = {'doc': doc_x}
        y = {'doc': tf.one_hot(doc_y + 1, 3, dtype='int32')}
        return x, y

    if doc_path is None and asp_path is not None:
        *asp_x, asp_y = csv_to_input(asp_path, ['context_left', 'target', 'context_right', 'polarity'])
        x = {'asp': asp_x}
        y = {'asp': tf.one_hot(asp_y + 1, 3, dtype='int32')}
        return x, y


def train_val_split(x_data: dict, y_data: dict, val_percentage: float) -> (dict, dict, dict, dict):
    N_OBSERVATIONS = len(x_data['asp'][0])
    VAL_OBSERVATIONS = round(N_OBSERVATIONS * val_percentage)
    VAL_INDEX = random.sample(range(0, N_OBSERVATIONS), VAL_OBSERVATIONS)
    TRAIN_INDEX = [index for index in range(0, N_OBSERVATIONS) if index not in VAL_INDEX]
    x_train = []
    y_train = tf.gather(y_data['asp'], tf.constant([TRAIN_INDEX]))
    x_val = []
    y_val = tf.gather(y_data['asp'], tf.constant([VAL_INDEX]))
    for column in x_data['asp']:
        x_val.append(np.array([column[index] for index in VAL_INDEX]))
        x_train.append(np.array([column[index] for index in TRAIN_INDEX]))

    return {'asp': x_train}, {'asp': x_val}, {'asp': tf.squeeze(y_train)}, {'asp': tf.squeeze(y_val)}


def save_data_to_csv(data: dict, model: tf.keras.Model, text_path: str, embeddings_path: str, probabilities_path: str):
    x_text = []
    x_embeddings = []
    x_probabilities = []

    embeddings, probabilities = model.get_embeddings_and_probabilities(inputs=data)
    x_embeddings.append(embeddings)
    x_probabilities.append(probabilities)
    x_text.append(data)

    pd.DataFrame(x_text).to_csv(text_path)
    pd.DataFrame(x_embeddings[0]).to_csv(embeddings_path)
    pd.DataFrame(x_probabilities[0]).to_csv(probabilities_path)


def csv_to_input_results_LCRRothop(path: str, y_true: bool):
    with open(path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)
        if y_true:
            input = [list(int(i) for i in row[1:]) for row in reader]
        else:
            input = [list(float(i) for i in row[1:]) for row in reader]

    return input


def load_results_LCRRothop(y_true_path: str, y_pred_path: str, probabilities_path: str, embeddings_path: str) -> dict:
    y_true = csv_to_input_results_LCRRothop(y_true_path, y_true=True)
    y_pred = csv_to_input_results_LCRRothop(y_pred_path, y_true=False)

    y_true = [np.argmax(prob_vector) for prob_vector in y_true]
    y_pred = [np.argmax(prob_vector) for prob_vector in y_pred]

    all_data = {
        'y_true': y_true,
        'y_pred': y_pred,
        'probabilities': csv_to_input_results_LCRRothop(probabilities_path, y_true=False),
        'embeddings': csv_to_input_results_LCRRothop(embeddings_path, y_true=False)
    }
    return all_data
