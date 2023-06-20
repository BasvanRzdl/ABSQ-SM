seed_value = 0

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['PYTHONHASHSEED'] = str(seed_value)

import random
random.seed(seed_value)

import numpy as np
np.random.seed(seed_value)

import tensorflow as tf
tf.random.set_seed(seed_value)
tf.config.experimental.enable_op_determinism()

import keras_tuner as kt
from AspEntQuaNet.asp_ent_qua_net import AspEntQuaNet
from tensorflow_addons.metrics import F1Score
from utils.data_loader import load_results_LCRRothop
from subdatasets import create_trainings_data
from measures import kullback_leibler_divergence, relative_absolute_error, absolute_error
import numpy as np
from AspEntQuaNet.baselines import get_baseline_metrics

def build_model_BiLSTM(hp):
    tf.random.set_seed(0)

    # Tune whether to use quantification and/or prevalence statistics
    #hp_quantification_statistics = hp.Boolean('quantification_stats', default='true', parent_name='self.use_quantification_statistics')
    #hp_prevalence_statistics = hp.Boolean('prevalence_stats', default='true', parent_name='self.use_prevalence_statistics')

    # Tune layers with values for the number of neurons and the drop rate
    hp_neurons_lay1 = 512
    hp_drop_out_lay1 = hp.Float('dropout_lay1', 0.2, 0.6, step=0.2)
    layer1 = {"neurons": hp_neurons_lay1, "activation_function": "relu", "drop_out": hp_drop_out_lay1}

    hp_neurons_lay2 = 256
    hp_drop_out_lay2 = hp.Float('dropout_lay2', 0.2, 0.6, step=0.2)
    layer2 = {"neurons": hp_neurons_lay2, "activation_function": "relu", "drop_out": hp_drop_out_lay2}

    hp_neurons_lay3 = 128
    hp_drop_out_lay3 = hp.Float('dropout_lay3', 0.2, 0.6, step=0.2)
    layer3 = {"neurons": hp_neurons_lay3, "activation_function": "relu", "drop_out": hp_drop_out_lay3}

    hp_neurons_lay4 = 64
    hp_drop_out_lay4 = hp.Float('dropout_lay4', 0.2, 0.6, step=0.2)
    layer4 = {"neurons": hp_neurons_lay4, "activation_function": "relu", "drop_out": hp_drop_out_lay4}

    layer_choice = hp.Choice('layer_choice', values=[0, 1, 2])
    if layer_choice == 0:
        hp_hidden_layers = [layer1, layer2, layer3, layer4]
    elif layer_choice == 1:
        hp_hidden_layers = [layer1, layer2, layer3]
    elif layer_choice == 2:
        hp_hidden_layers = [layer2, layer3, layer4]

    # Tune learning rate for Adam optimizer with values from 0.01, 0.001 & 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[10**-i for i in range(2, 5)])
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    hp_bilstm_output_neurons = hp.Choice('bilstm_output_neurons', values=[512, 256, 128])

    #hp_subdataset_size = hp.Choice('subdataset_size', values=[1700, 2200, 3000, 4000, 5000, 7000, 10000])

    f1 = F1Score(num_classes=3, average='macro', name='f1')

    loss_KLD = tf.keras.losses.KLDivergence()

    model = AspEntQuaNet(hidden_layers=hp_hidden_layers,
                         use_quantification_statistics=True,
                         use_prevalence_statistics=True,
                         bilstm_output_neurons=hp_bilstm_output_neurons,
                         vector_size=14,
                         subdataset_size=250,
                         number_of_classes=3)
    model.compile(optimizer=optimizer, loss=loss_KLD, loss_weights=1, metrics=['acc', f1], run_eagerly=True)

    return model


sort_key = 'entropy'

train_results_lcr = load_results_LCRRothop(
    y_true_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/y_train.csv',
    y_pred_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/y_train_pred.csv',
    probabilities_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_train_probabilities.csv',
    embeddings_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_train_embeddings.csv'
)

val_results_lcr = load_results_LCRRothop(
    y_true_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/y_val.csv',
    y_pred_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/y_val_pred.csv',
    probabilities_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_val_probabilities.csv',
    embeddings_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_val_embeddings.csv'
)

x_train = create_trainings_data(
    all_data=train_results_lcr,
    sub_dataset_percentage=0.4,
    number_of_subdatasets=1000,
    sort_key=sort_key,
    subdataset_size=250
)
y_train = x_train.pop('true_prevalence')

x_val = create_trainings_data(
    all_data=val_results_lcr,
    sub_dataset_percentage=0.4,
    number_of_subdatasets=1000,
    sort_key=sort_key,
    subdataset_size=250
)
y_val = x_val.pop('true_prevalence')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# Instantiate the tuner
tuner = kt.Hyperband(build_model_BiLSTM,
                        objective=kt.Objective("val_f1", direction="max"),
                        max_epochs=10,
                        factor=3,
                        hyperband_iterations=1,
                        overwrite=False,
                        directory="logs/hp",
                        project_name="AspEntQuaNet",)

try:
    tuner.search(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, callbacks=[stop_early], verbose=1)
except Exception as e:
    print(f"Error reached: {e}")

models = tuner.get_best_models(num_models=1)
best_model = models[0]
best_model.evaluate(x_val, y_val)
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)
print(best_hyperparameters)

best_model.save_weights('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/AspEntQuaNet.h5')

test_results_lcr = load_results_LCRRothop(
    y_true_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Evaluation/Eval Food Quality/y_test_true_food_quality.csv',
    y_pred_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Evaluation/Eval Food Quality/y_test_pred_food_quality.csv',
    probabilities_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Evaluation/Eval Food Quality/x_test_probabilities_food_quality.csv',
    embeddings_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Evaluation/Eval Food Quality/x_test_embeddings_food_quality.csv'
)

x_test_ent = create_trainings_data(all_data=test_results_lcr, sub_dataset_percentage=0.4, number_of_subdatasets=1000, sort_key=sort_key, subdataset_size=250)
y_test_true_ent = x_test_ent.pop('true_prevalence')

y_test_pred_ent = best_model.predict(x_test_ent)

kld_ent = kullback_leibler_divergence(y_test_true_ent, y_test_pred_ent)
rae_ent = relative_absolute_error(y_test_true_ent, y_test_pred_ent)
ae_ent = absolute_error(y_test_true_ent, y_test_pred_ent)
print(f'KLD: {kld_ent}, RAE: {rae_ent}, AE: {ae_ent}')

y_test_baselines = get_baseline_metrics(y_test_true_ent, x_test_ent)
print(y_test_baselines)