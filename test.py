# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:13:07 2020

@author: kach
"""

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from parameters import *
from utils import create_model, load_data

import pickle
import os

# dataset name
dataset_name = "convnet"
# get the unique model name based on hyper parameters on parameters.py
model_name = get_model_name(dataset_name)

data = load_data(N_WORDS, SEQUENCE_LENGTH, TEST_SIZE, oov_token=OOV_TOKEN)

model = create_model(data["tokenizer"].word_index, units=UNITS, n_layers=N_LAYERS, 
                    cell=RNN_CELL, bidirectional=IS_BIDIRECTIONAL, embedding_size=EMBEDDING_SIZE, 
                    sequence_length=SEQUENCE_LENGTH, dropout=DROPOUT, 
                    loss=LOSS, optimizer=OPTIMIZER, output_length=data["y_train"][0].shape[0])

model.load_weights(os.path.join("results", f"{model_name}.h5"))


def get_predictions(text):
    sequence = data["tokenizer"].texts_to_sequences([text])
    # pad the sequences
    sequence = pad_sequences(sequence, maxlen=SEQUENCE_LENGTH)
    # get the prediction
    prediction = model.predict(sequence)[0]
    print("output vector:", prediction)
    return data["int2label"][np.argmax(prediction)]


text = input("Enter your text: ")
prediction = get_predictions(text)
print("="*50)
print("The class is:", prediction)