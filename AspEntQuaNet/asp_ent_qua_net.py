import numpy as np

from models.layers.embedding import BERTEmbedding
import tensorflow as tf
from tensorflow_addons.layers import AdaptiveAveragePooling1D
from models.layers.attention import BilinearAttention, HierarchicalAttention
from typing import List
import os
import random
# make another class AspEntQuaNetNoBiLSTM(tf.keras.Model) which has as input in the call
# a dictionary with 1 vector with [entropy|prob|embeddings] and 1 vector with the
# statistics. Then with the model settings you can choose which statistics to use (see
# example from robbert below)

class AspEntQuaNet(tf.keras.Model):
    def __init__(self,
                 use_quantification_statistics: bool,
                 use_prevalence_statistics: bool,
                 bilstm_output_neurons: int,
                 hidden_layers: List[dict],
                 subdataset_size: int,
                 vector_size: int,
                 number_of_classes: int = 3,
                 ):
        """
        hidden_layers is a list with dictionaries. Each dictionary describes
        one hidden layer, and the hyperparameters you want to use there.

        Example:
            [{"neurons": 100, "activation_function": *some supported function name*, "drop_out": 0.2, etc.}, {...}]
        """
        super().__init__()
        self.number_of_classes = number_of_classes
        self.use_quantification_statistics = use_quantification_statistics
        self.use_prevalence_statistics = use_prevalence_statistics
        self.bilstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(bilstm_output_neurons, return_sequences=False, stateful=False),
            input_shape=(subdataset_size, vector_size)
        )

        size_input_vector = bilstm_output_neurons

        if self.use_quantification_statistics:
            size_input_vector += 12

        if self.use_prevalence_statistics:
            size_input_vector += 9

        self.input_dense_layer = tf.keras.layers.InputLayer(
            input_shape=(size_input_vector,)
        )

        self.custom_dense_layers = []
        for layer in hidden_layers:
            self.custom_dense_layers.append(
                tf.keras.layers.Dense(
                    units=layer["neurons"],
                    activation=layer["activation_function"]
                )
            )
            if layer["drop_out"] > 0:
                self.custom_dense_layers.append(
                    tf.keras.layers.Dropout(layer["drop_out"])
                )

        self.prediction_layer = tf.keras.layers.Dense(
            units=self.number_of_classes,
            activation="softmax"
        )

    def call(self, inputs: dict):
        """
        """
        bilstm_input = inputs['bilstm_input']
        bilstm_output = self.bilstm_layer(bilstm_input)

        # Add statistics, the first 9 are prevalence, last 12 are quantification
        if self.use_quantification_statistics:
            # may not be a list after the bilstm, fix typing if needed
            bilstm_output = self.concatenate_statistics_to_bilstm_output(
                complete_batch=bilstm_output,
                statistics=inputs['statistics'],
                statistics_range=(9,22),
                extra_added_statistics=12
            )

        if self.use_prevalence_statistics:
            bilstm_output = self.concatenate_statistics_to_bilstm_output(
                complete_batch=bilstm_output,
                statistics=inputs['statistics'],
                statistics_range=(0, 9),
                extra_added_statistics=9
            )

        hidden_state = self.input_dense_layer(bilstm_output)
        for layer in self.custom_dense_layers:
            hidden_state = layer(hidden_state)

        probabilites = self.prediction_layer(hidden_state)
        return probabilites

    def concatenate_statistics_to_bilstm_output(self, complete_batch: tf.Tensor, statistics: tf.Tensor, statistics_range: tuple, extra_added_statistics: int) -> tf.Tensor:
        result = np.empty(shape=(0, len(complete_batch[0])+extra_added_statistics))
        for index in range(0, len(complete_batch)):
            new_vector = tf.concat((complete_batch[0], statistics[index][statistics_range[0]:statistics_range[1]]), axis=0)
            new_vector = tf.reshape(new_vector, shape=(1, len(new_vector)))
            result = tf.concat((result, new_vector), axis=0)

        return tf.convert_to_tensor(result, np.float32)








