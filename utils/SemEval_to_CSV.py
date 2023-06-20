from typing import Sequence
import dask.dataframe as dd
import numpy as np
import pandas as pd
import tensorflow as tf
import xml.etree.ElementTree as ET
import csv
import os


def semeval_to_csv_restaurant(f_in: str, f_out: str):
    root = ET.parse(f_in).getroot()

    with open(f_out, 'w', encoding='utf-8', newline='') as file:
        columns = ("context_left", "target", "context_right", "polarity", "category")
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for sentence in root.iter('sentence'):
            sent = sentence.find('text').text
            for opinion in sentence.iter('Opinion'):
                sentiment = opinion.get('polarity')
                if sentiment == "positive":
                    polarity = 1
                elif sentiment == "neutral":
                    polarity = 0
                elif sentiment == "negative":
                    polarity = -1
                else:
                    polarity = None

                start_attr = opinion.get('from')
                end_attr = opinion.get('to')

                # skip the current iteration if either 'from' or 'to' attribute is missing
                if start_attr is None or end_attr is None:
                    continue

                start = int(start_attr)
                end = int(end_attr)

                # skip implicit targets
                if start == end == 0:
                    continue

                context_left = sent[:start]
                context_right = sent[end:]
                writer.writerow(
                    {"context_left": context_left, "target": sent[start:end], "context_right": context_right,
                     "polarity": polarity, "category": opinion.get('category')})


def semeval_to_csv_laptop(f_in: str, f_out: str):
    root = ET.parse(f_in).getroot()

    with open(f_out, 'w', encoding='utf-8', newline='') as file:
        columns = ("context_left", "target", "context_right", "polarity", "category")
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for review in root.iter('Review'):
            for sentence in review.iter('sentence'):
                sent = sentence.find('text').text
                for opinion in sentence.iter('Opinion'):
                    sentiment = opinion.get('polarity')
                    category = opinion.get('category')

                    if sentiment == "positive":
                        polarity = 1
                    elif sentiment == "neutral":
                        polarity = 0
                    elif sentiment == "negative":
                        polarity = -1
                    else:
                        polarity = None

                    # Here, we assume that every word in the sentence is an aspect.
                    # You could modify this part if you have a way to identify the specific aspect words in the sentence.
                    context_left = ""
                    target = sent
                    context_right = ""

                    writer.writerow(
                        {"context_left": context_left, "target": target, "context_right": context_right,
                         "polarity": polarity, "category": category})


input_folder_restaurant = "/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/XML/Restaurant"
output_folder_restaurant = "/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/CSV/Restaurant"

for filename in os.listdir(input_folder_restaurant):
    if filename.endswith(".xml"):
        input_file = os.path.join(input_folder_restaurant, filename)
        output_file = os.path.join(output_folder_restaurant, filename.replace(".xml", ".csv"))
        semeval_to_csv_restaurant(input_file, output_file)

input_folder_laptop = "/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/XML/Laptop"
output_folder_laptop = "/Users/basvanroozendaal/Downloads/DATA THESIS/Raw data/CSV/Laptop"

for filename in os.listdir(input_folder_laptop):
    if filename.endswith(".xml"):
        input_file = os.path.join(input_folder_laptop, filename)
        output_file = os.path.join(output_folder_laptop, filename.replace(".xml", ".csv"))
        semeval_to_csv_laptop(input_file, output_file)
