#Used links: https://www.tensorflow.org/tutorials/keras/text_classification
#https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/


import tensorflow as tf
import numpy as np
import pandas as pd
import json
import re
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import sklearn.model_selection as sk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
tf.get_logger().setLevel('INFO')


#Specifying file name for training set
filename = "/home/marina/PycharmProjects/pythonProject1/ClassificationCompetition_private/data/train.jsonl"
#Specifying file name for test set
test_file = '/home/marina/PycharmProjects/pythonProject1/ClassificationCompetition_private/data/test.jsonl'


#Function to create a train tuple of lists of Labels/Context/Response from json
def getdatajson(path):
    with open(path, "r") as file_data:
        file_r = file_data.read()
        response = []
        context = []
        labels = []
        for line in file_r.splitlines():
            document = json.loads(line)
            labels.append(document["label"])
            intermediate = str(" ".join(document["context"]))
            context.append(intermediate)
            response.append(document["response"])

    return labels, context, response

#Function to create a test tuple of lists of Index/Context/Response from json
def getdatatest(path):
    with open(path, "r") as file_data:
        file_r = file_data.read()
        test_cont = []
        test_resp = []
        test_idx = []
        for line in file_r.splitlines():
            document = json.loads(line)
            intermediate = str(" ".join(document["context"]))
            test_cont.append(intermediate)
            test_resp.append(document["response"])
            test_idx.append(document["id"])
    return test_idx, test_cont, test_resp


#Applying function to get the training data file:

data = getdatajson(filename)

#Applying function to get the test data file:

test_data = getdatatest(test_file)


#Function, where we take 2 arrays of strings, each of shape (1, N), stack them to have the shape (2, N), and joining the strings for each index of axis 1.
def combine_vectors(vector1, vector2):
    array_data2 = np.array(vector1, dtype=str)
    array_data3 = np.array(vector2, dtype=str)
    Features = np.vstack((array_data3, array_data2))
    combined_features = []
    for item in Features.T:
        summed = str(" ".join(map(str, [item[0], item[1]])))
        combined_features.append(summed)
    return combined_features
    #return array_data3

#Applying the function to get training and test numpy arrays of data, having both shape (N, 1), where N - number of rows in initial file.
train_features = combine_vectors(data[1], data[2])
test_features = combine_vectors(test_data[1], test_data[2])


#Creating Label vector and transferring labels into integer
Labels_num = []
for item in data[0]:
    if item.endswith("NOT_SARCASM"):
        Labels_num.append(0)
    else:
        Labels_num.append(1)
Labels_num = np.array(Labels_num)

tf.random.set_seed(1)
np.random.seed(1)

#Split of training array on train and validation, converting to tensors from numpy array, transposing where needed

X_train, X_val, y_train, y_val = sk.train_test_split(train_features,Labels_num.T,test_size=0.2, random_state = 42)
tensor_train = tf.constant(X_train)
tensor_val = tf.constant(X_val)
tensor_label_train = tf.constant(y_train)
tensor_label_val = tf.constant(y_val)

#Converting test data to tensor:
tensor_test = tf.constant(test_features)


#Cutting html tags and symbols, putting all the text in lower case, removing punctuation signs:
#used from https://www.tensorflow.org/tutorials/text/word_embeddings
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')

#Vectorizing tensors of features for training, validation, test set.
#used from https://www.tensorflow.org/tutorials/text/word_embeddings
max_features = 12000
sequence_length =300

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

vectorize_layer.adapt(tensor_train)


train_dataset = vectorize_layer(tf.expand_dims(tensor_train, -1))
val_dataset = vectorize_layer(tf.expand_dims(tensor_val, -1))
test_dataset = vectorize_layer(tf.expand_dims(tensor_test, -1))

print(train_dataset)


#Building the models:
#Model used from: https://machinelearningmastery.com/develop-n-gram-multichannel-convolutional-neural-network-sentiment-analysis/
length = sequence_length
vocab_size = max_features
# channel 1
inputs1 = layers.Input(shape=(length,))
embedding1 = layers.Embedding(vocab_size, 100)(inputs1)
conv1 = layers.Conv1D(filters=64, kernel_size=4, activation='tanh', kernel_initializer = "lecun_normal")(embedding1)
drop1 = layers.Dropout(0.5)(conv1)
pool1 = layers.MaxPooling1D(pool_size=2)(drop1)
print(pool1.shape)
flat1 = layers.Flatten()(pool1)
# channel 2
inputs2 = layers.Input(shape=(length,))
embedding2 = layers.Embedding(vocab_size, 100)(inputs2)
conv2 = layers.Conv1D(filters=64, kernel_size=6, activation='tanh', kernel_initializer = "lecun_normal")(embedding2)
drop2 = layers.Dropout(0.5)(conv2)
pool2 = layers.MaxPooling1D(pool_size=2)(drop2)
flat2 = layers.Flatten()(pool2)
# channel 3
inputs3 = layers.Input(shape=(length,))
embedding3 = layers.Embedding(vocab_size, 100)(inputs3)
conv3 = layers.Conv1D(filters=64, kernel_size=8, activation='tanh', kernel_initializer = "lecun_normal")(embedding3)
drop3 = layers.Dropout(0.5)(conv3)
pool3 = layers.MaxPooling1D(pool_size=2)(drop3)
flat3 = layers.Flatten()(pool3)
# merge
merged = layers.concatenate([flat1, flat2, flat3])
# interpretation
dense1 = layers.Dense(10, activation='relu', kernel_initializer = "he_normal")(merged)
outputs = layers.Dense(1, activation='sigmoid')(dense1)
model1 = keras.Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)



# compile model
model1.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='nadam', metrics=['accuracy'])


print("defining model")
# fit model
model1.fit([train_dataset,train_dataset,train_dataset], tensor_label_train, epochs=10, batch_size=16)
# evaluate model on training dataset
loss, acc = model1.evaluate([val_dataset,val_dataset,val_dataset], tensor_label_val, verbose=0)
print('Train Accuracy: %f' % (acc * 100))


#Predicting the labels for test dataset
predictions = model1.predict([test_dataset, test_dataset, test_dataset])
print(predictions)

test_idx = test_data[0]
#writing the results in the file
fin = open("/home/marina/PycharmProjects/pythonProject1/ClassificationCompetition_private/answer.txt", "w")
idx = 0
print("Process predictions")
for item in predictions:
    fin.write("{},{}{}".format(test_idx[idx], "SARCASM" if item >0.5 else "NOT_SARCASM",  "\n" if idx < len(test_idx) - 1 else ""))
    idx += 1
fin.close()