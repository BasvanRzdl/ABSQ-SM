from typing import List

from models.layers.embedding import BERTEmbedding
import tensorflow as tf
from tensorflow_addons.layers import AdaptiveAveragePooling1D
from models.layers.attention import BilinearAttention, HierarchicalAttention

class AspEntQuaNetNoBiLSTM(tf.keras.Model):
    def __init__(self,
                 use_quantification_statistics: bool,
                 use_prevalence_statistics: bool,
                 size_input_vector: int,
                 hidden_layers: List[dict],
                 number_of_classes: int = 3
                 ):

        super().__init__()
        self.number_of_classes = number_of_classes
        self.use_quantification_statistics = use_quantification_statistics
        self.use_prevalence_statistics = use_prevalence_statistics

        self.size_input_vector = size_input_vector

        if self.use_quantification_statistics:
            self.size_input_vector += 12

        if self.use_prevalence_statistics:
            self.size_input_vector += 9

        self.input_dense_layer = tf.keras.layers.InputLayer(
            shape=(size_input_vector,)
        )

        self.custom_dense_layers = []
        for layer in hidden_layers:
            self.custom_dense_layers.append(
                tf.keras.layers.Dense(
                    units=layer["units"],
                    activation=layer["activation"]
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
        input = inputs['bilstm_input']

        # Add statistics, the first 9 are prevalence, last 12 are quantification
        if self.use_quantification_statistics:
            # may not be a list after the bilstm, fix typing if needed
            input.extend(inputs['statistics'][-12:])

        if self.use_prevalence_statistics:
            input.extend(inputs['statistics'][:9])

        hidden_state = self.input_dense_layer(input)
        for layer in self.custom_dense_layers:
            # might not be correct syntax
            hidden_state = layer(hidden_state)

        probabilites = self.prediction_layer(hidden_state)




