import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from models.layers.embedding import BERTEmbedding
import tensorflow as tf
from models.LCRRothopPP import LCRRothopPP
from tensorflow_addons.metrics import F1Score
from utils.data_loader import load_data, train_val_split
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import random

np.random.seed(0)

restaurant2016_train_path = r'/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/CSV//Restaurant/Restaurant2016train.csv'
restaurant2016_test_path = r'/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/CSV//Restaurant/Restaurant2016test.csv'

# LOAD DATA
x_train_bf_split, y_train_bf_split = load_data(asp_path=restaurant2016_train_path)
x_train, x_val, y_train, y_val = train_val_split(x_data=x_train_bf_split, y_data=y_train_bf_split, val_precentage=0.2)
x_test, y_test = load_data(asp_path=restaurant2016_test_path)

seed_value = 0
tf.random.set_seed(seed_value)

emb = BERTEmbedding()

drop_rate_1 = 0.5
drop_rate_2 = 0.2
hidden_units = 250
regularizer = tf.keras.regularizers.L1L2(l1=1e-09, l2=1e-09)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model = LCRRothopPP(embedding_layer=emb,
                    hop=3,
                    hierarchy=(False, True),
                    drop_1=drop_rate_1,
                    drop_2=drop_rate_2,
                    hidden_units=hidden_units,
                    regularizer=regularizer)

f1 = F1Score(num_classes=3, average='macro', name='f1')
batch_size = 16

model.compile(optimizer=optimizer, loss={'asp': 'categorical_crossentropy'}, loss_weights={'asp': 1}, metrics={'asp': ['acc', f1]}, run_eagerly=True)

history = model.fit(x_train, y_train, epochs=2, validation_data=(x_val, y_val), batch_size = batch_size)
model.save('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/model_before_compile')

results = model.predict(x_test)
probabilities = results['asp']

embeddings = model.get_embeddings(x_test)

eval_results = model.evaluate(x_test, y_test)

# convert probabilities and embeddings to dataframe and save as excel
df = pd.DataFrame(data=probabilities, columns=['probabilities'])
df['embeddings'] = list(embeddings)
df['eval_results'] = list(eval_results)

# save to excel
df.to_excel('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/results.xlsx', index=False)

with open('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_train.csv', 'wb') as f:
    w = csv.writer(f)
    w.writerows(x_train.items())

with open('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/y_train.csv', 'wb') as f:
    w = csv.writer(f)
    w.writerows(y_train.items())

with open('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_val.csv', 'wb') as f:
    w = csv.writer(f)
    w.writerows(x_val.items())

with open('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/y_val.csv', 'wb') as f:
    w = csv.writer(f)
    w.writerows(y_val.items())

with open('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/x_test.csv', 'wb') as f:
    w = csv.writer(f)
    w.writerows(x_test.items())

with open('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/LCRRothopPP outputs restaurant 2016/y_test.csv', 'wb') as f:
    w = csv.writer(f)
    w.writerows(y_test.items())