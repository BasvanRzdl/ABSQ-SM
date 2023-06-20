import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from models.layers.embedding import BERTEmbedding
import tensorflow as tf
import keras_tuner as kt
from models.LCRRothopPP import LCRRothopPP
from tensorflow_addons.metrics import F1Score
from utils.data_loader import load_data, train_val_split, save_data_to_csv
import pandas as pd


def build_model(hp):
    tf.random.set_seed(0)
    emb = BERTEmbedding()

    # Tune regularizers rate for L1 regularizer with values from 0.001, 0.0001, 1e-05, 1e-06, 1e-07, 1e-08 or 1e-09
    hp_l1_rates = hp.Choice("l1_regularizer", values=[10**-i for i in range(3, 10, 2)])

    # Tune regularizers rate for L2 regularizer with values from 0.001, 1e-05, 1e-07, or 1e-09
    hp_l2_rates = hp.Choice("l2_regularizer", values=[10**-i for i in range(3, 10, 2)])

    regularizer = tf.keras.regularizers.L1L2(l1=hp_l1_rates, l2=hp_l2_rates)


    # Tune learning rate for Adam optimizer with values from 0.01, 0.001 & 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[10**-i for i in range(2, 5)])

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)

    # Tune dropout layers with values from 0.2 - 0.7 with stepsize of 0.1.
    drop_rate_1 = hp.Float("dropout_1", 0.2, 0.6, step=0.2)
    drop_rate_2 = hp.Float("dropout_2", 0.2, 0.6, step=0.2)

    # Tune number of hidden layers for the BiLSTMs
    hidden_units = hp.Int("hidden_units", min_value=200, max_value=400, step=100)

    f1 = F1Score(num_classes=3, average='macro', name='f1')

    model = LCRRothopPP(embedding_layer=emb, hop=3, hierarchy=(False, True), drop_1=drop_rate_1, drop_2=drop_rate_2, hidden_units=hidden_units, regularizer=regularizer)
    model.compile(optimizer=optimizer, loss={'asp': 'categorical_crossentropy'}, loss_weights={'asp': 1}, metrics={'asp':['acc', f1]}, run_eagerly=True)

    return model

restaurant2016_train_path = r'/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/CSV/Restaurant/Restaurant2016train.csv'

# LOAD DATA
x_train_bf_split, y_train_bf_split = load_data(asp_path=restaurant2016_train_path)
x_train, x_val, y_train, y_val = train_val_split(x_data=x_train_bf_split, y_data=y_train_bf_split, val_percentage=0.2)


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# Instantiate the tuner
tuner = kt.Hyperband(build_model,
                    objective=kt.Objective("val_f1", direction="max"),
                    max_epochs=10,
                    factor=3,
                    hyperband_iterations=1,
                    directory="logs/hp",
                    overwrite=False,
                    project_name="lcrrothop",)

try:
    tuner.search(x_train, y_train, validation_data=(x_val, y_val), batch_size=16, callbacks=[stop_early], verbose=1)
except Exception as e:
    print(f"Error reached: {e}")

models = tuner.get_best_models(num_models=1)
best_model = models[0]

best_model.evaluate(x_val, y_val)
y_train_pred = best_model.predict(x_train)
y_val_pred = best_model.predict(x_val)

best_model.save_weights('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/LCRRotHopPP.h5')

save_data_to_csv(
    data=x_train,
    model=best_model,
    text_path="/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_train_text.csv",
    embeddings_path="/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_train_embeddings.csv",
    probabilities_path="/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_train_probabilities.csv"
)

save_data_to_csv(
    data=x_val,
    model=best_model,
    text_path="/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_val_text.csv",
    embeddings_path="/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_val_embeddings.csv",
    probabilities_path="/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_val_probabilities.csv"
)

pd.DataFrame(y_train['asp']).to_csv("/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/y_train.csv")
pd.DataFrame(y_val['asp']).to_csv("/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/y_val.csv")

pd.DataFrame(y_train_pred['asp']).to_csv("/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/y_train_pred.csv")
pd.DataFrame(y_val_pred['asp']).to_csv("/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/y_val_pred.csv")

# Evaluation
restaurant2016_test_path = r'/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Categories/test_service_general.csv'
x_test, y_test_true = load_data(asp_path=restaurant2016_test_path)

y_test_pred = best_model.predict(x_test)

save_data_to_csv(
    data=x_test,
    model=best_model,
    text_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Evaluation/Eval Service General/x_test_text_service_general.csv',
    embeddings_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Evaluation/Eval Service General/x_test_embeddings_service_general.csv',
    probabilities_path='/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Evaluation/Eval Service General/x_test_probabilities_service_general.csv',
)

pd.DataFrame(y_test_true['asp']).to_csv('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Evaluation/Eval Service General/y_test_true_service_general.csv')
pd.DataFrame(y_test_pred['asp']).to_csv('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Evaluation/Eval Service General/y_test_pred_service_general.csv')
