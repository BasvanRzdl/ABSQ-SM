import csv
import pandas as pd
test_path = '/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/CSV/Restaurant/Restaurant2016test.csv'

#out_path_food_quality = '/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Categories/test_food_quality.csv',

#out_path_service_general = '/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Categories/test_service_general.csv',

#out_path_ambience_general = '/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Categories/test_ambience_general.csv',

#out_path_restaurant_general = '/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Categories/test_restaurant_general.csv',

df = pd.read_csv(test_path)

food_quality = ['FOOD#QUALITY']
service_general = ['SERVICE#GENERAL']
ambience_general = ['AMBIENCE#GENERAL']
restaurant_general = ['RESTAURANT#GENERAL']

filtered_df = df[df.iloc[:, -1].isin(food_quality)]
filtered_df.to_csv('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Categories/test_food_quality.csv', index=False)

filtered_df = df[df.iloc[:, -1].isin(service_general)]
filtered_df.to_csv('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Categories/test_service_general.csv', index=False)

filtered_df = df[df.iloc[:, -1].isin(ambience_general)]
filtered_df.to_csv('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Categories/test_ambience_general.csv', index=False)

filtered_df = df[df.iloc[:, -1].isin(restaurant_general)]
filtered_df.to_csv('/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/Categories/test_restaurant_general.csv', index=False)


