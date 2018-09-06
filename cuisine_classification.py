#!/usr/bin/python

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import h5py
import json
import sys
import getopt

word_index_path = 'data/word_index.json'

cuisines_path = 'data/cuisines.json'

model_path = 'data/model.h5'

MAX_INGREDIENTS = 65

MAX_WORDS_IN_INGREDIENT = 16

MAX_CUISINES = 64

MAX_VOCAB_SIZE = 10000


def load_cuisines():
    try:
        f = open(cuisines_path, 'r')
        cuisines = json.loads(f.read())
        f.close()
    except FileNotFoundError:
        print('creating cuisines list')
        cuisines = []
    return cuisines


def save_cuisines(cuisines):
    print('saving cuisines list')
    cuisines_file = open(cuisines_path, 'w+')
    cuisines_file.write(json.dumps(cuisines))
    cuisines_file.close()


def load_word_index():
    try:
        f = open(word_index_path, 'r')
        word_index = json.loads(f.read())
        f.close()
    except FileNotFoundError:
        print('creating word_index')
        word_index = {'<PAD>': 0, '<UNK>': 1}
    return word_index


def save_word_index(word_index):
    print('saving word_index')
    word_index_file = open(word_index_path, 'w+')
    word_index_file.write(json.dumps(word_index))
    word_index_file.close()


def train(train_path, epochs_num):
    print('reading train data')
    try:
        f = open(train_path, 'r')
    except FileNotFoundError:
        print('specified training file not found')
        sys.exit(1)
    raw_data = json.loads(f.read())
    f.close()
    # creating vocabulary
    word_index = load_word_index()
    for recipe in raw_data:
        for ingredient in recipe['ingredients']:
            for word in ingredient.split(' '):
                word_lower = word.lower()
                if word_lower not in word_index:
                    word_index[word_lower] = len(word_index)
    save_word_index(word_index)
    # creating list of cuisine types
    cuisines = load_cuisines()
    for recipe in raw_data:
        cuisine_lower = recipe['cuisine'].lower()
        if cuisine_lower not in cuisines:
            cuisines.append(cuisine_lower)
    save_cuisines(cuisines)
    # getting train_data ready to pass to the NN
    print('creating data tensors for NN')
    train_data = []
    for recipe in raw_data:
        train_data.append([])
        for ingredient in recipe['ingredients']:
            train_data[len(train_data) - 1].append([])
            for word in ingredient.split(' '):
                word_lower = word.lower()
                train_data[len(train_data) - 1][len(train_data[len(train_data) - 1]) - 1].append(word_index[word_lower])
    # creating fixed size for all recipes
    for dish in train_data:
        while len(dish) < MAX_INGREDIENTS:
            dish.append([])
    for dish in train_data:
        for ingredient in dish:
            while len(ingredient) < MAX_WORDS_IN_INGREDIENT:
                ingredient.append(word_index['<PAD>'])
    for dish in train_data:
        new_dish = []
        for ingredient in dish:
            for word in ingredient:
                new_dish.append(word)
        train_data[train_data.index(dish)] = new_dish
    train_labels = []
    for dish in raw_data:
        train_labels.append(cuisines.index(dish['cuisine']))
    train_labels = keras.utils.to_categorical(train_labels, num_classes=MAX_CUISINES)
    train_data = keras.preprocessing.sequence.pad_sequences(
        train_data, padding='post', maxlen=MAX_WORDS_IN_INGREDIENT * MAX_INGREDIENTS, value=word_index['<PAD>'])
    print('loading NN model')
    try:
        model = keras.models.load_model(model_path)
    except OSError:
        vocab_size = MAX_VOCAB_SIZE
        model = keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size, 256))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(256, activation=tf.nn.relu))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(MAX_CUISINES, activation=tf.nn.softmax))
        model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    model.fit(train_data, train_labels, epochs=epochs_num, verbose=2)
    keras.models.save_model(model, model_path)
    print('trained model saved')


def predict(test_path, out_path='out.csv'):
    word_index = load_word_index()
    cuisines = load_cuisines()
    print('reading test data')
    try:
        f = open(test_path, 'r')
    except FileNotFoundError:
        print('specified test data file not found')
        sys.exit(1)
    raw_test_data = json.loads(f.read())
    f.close()
    print('preprocessing test data')
    test_data = []
    for dish in raw_test_data:
        test_data.append([])
        for ingredient in dish['ingredients']:
            test_data[len(test_data) - 1].append([])
            for word in ingredient.split(' '):
                word_lower = word.lower()
                if word_lower in word_index:
                    test_data[len(test_data) - 1][len(test_data[len(test_data) - 1]) - 1].append(word_index[word_lower])
                else:
                    test_data[len(test_data) - 1][len(test_data[len(test_data) - 1]) - 1].append(word_index['<UNK>'])

    for dish in test_data:
        while len(dish) < 65:
            dish.append([])

    for dish in test_data:
        for ingredient in dish:
            while len(ingredient) < 16:
                ingredient.append(word_index['<PAD>'])

    for dish in test_data:
        new_dish = []
        for ingredient in dish:
            for word in ingredient:
                new_dish.append(word)
        test_data[test_data.index(dish)] = new_dish

    test_data = keras.preprocessing.sequence.pad_sequences(
        test_data, padding='post', maxlen=16 * 65, value=word_index['<PAD>'])

    try:
        model = keras.models.load_model(model_path)
    except OSError:
        print('model file not found')
        print('train the NN before requesting a prediction')
        print('see -t --train option in help')
        sys.exit(1)
    predictions = model.predict(test_data)
    data_frame_dict = {'id': [], 'cuisine': []}
    for dish in raw_test_data:
        data_frame_dict['id'].append(str(dish['id']))
        data_frame_dict['cuisine'].append(cuisines[np.argmax(predictions[raw_test_data.index(dish)])])
    data_frame = pd.DataFrame(data=data_frame_dict)
    data_frame.to_csv(out_path, sep=',', index=False)
    print('predictions saved to ' + out_path)


def print_cuisines():
    try:
        f = open("data/cuisines.json", "r")
    except FileNotFoundError:
        print('the list of cuisines is empty')
        sys.exit()
    cuisines = json.loads(f.read())
    f.close()
    print('list of cuisines:')
    for cuisine in cuisines:
        print(cuisine)


def print_help():
    print('cuisine classification program')
    print('options:')
    print('   -p --predict [FILE]  predict a cuisine given the json file consisting of')
    print('                        list of dictionaries containing key: ')
    print('                        "ingredients" - list of ingredients ')
    print('                        "id" - recipe id')
    print('                        the result is written into the file ')
    print('                        specified by -out-file and is in csv format')
    print('   -h --help            show this message')
    print('   -t --train [FILE]    train the neural network with the new data')
    print('                        [FILE] format should be json list')
    print('                        of dictionaries containing keys:')
    print('                        "cuisine" - string')
    print('                        "ingredients" - list of ingredients')
    print('                        default file path is "./input/train.json"')
    print('   -e --epochs [NUM]    change number of epochs executed during training')
    print('                        default is 10')
    print('   -c --cuisines        shows list of cuisines ')
    print('                        that the program is able to identify')
    print('   -o --out-file [FILE] specifies the out file name')


def main(argv):
    try:
        opts, args = getopt.getopt(argv,
                                   "p:ht:ce:o:", ["help", "train=", "predict=", "cuisines", "epochs=", "out-file="])
    except getopt.GetoptError:
        print_help()
        sys.exit(1)
    train_path = 'input/train.json'
    test_path = 'input/test.json'
    mode = "predict"
    out_file = "out.csv"
    epochs_num = 10
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print_help()
            sys.exit()
        elif opt in ("-t", "--train"):
            train_path = arg
            mode = "train"
        elif opt in ("-p", "--predict"):
            test_path = arg
            mode = "predict"
        elif opt in ("-c", "--cuisines"):
            print_cuisines()
            sys.exit()
        elif opt in ("-e", "--epochs"):
            epochs_num = arg
        elif opt in ("-o", "--out-file"):
            out_file = arg
    if mode == "predict":
        predict(test_path, out_file)
    elif mode == "train":
        train(train_path, int(epochs_num))
    else:
        print('undefined mode :' + mode)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
