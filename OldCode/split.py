from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
data_before_split = pd.read_csv('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/CSV/Restaurant/Restaurant2016train.csv')

x_train_bf_split = data_before_split[['context_left', 'target', 'context_right']]
y_train_bf_split = data_before_split[['polarity']]

x_train, x_val, y_train, y_val = train_test_split(x_train_bf_split, y_train_bf_split, test_size=0.1, random_state = 0)

val_set = x_val.merge(y_val, how = 'inner')

train_set = x_train.merge(y_train, how = 'inner')

filepath = Path('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/CSV/Restaurant/Restaurant2016val')
filepath.parent.mkdir(parents=True, exist_ok=True)
val_set.to_csv(filepath)

filepath = Path('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/CSV/Restaurant/Restaurant2016trainSplit.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
train_set.to_csv(filepath)