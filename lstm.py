import json
import tensorflow as tf
import numpy as np
import sklearn.model_selection as sk
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re
import string
from tensorflow.keras import layers
from tensorflow.keras import losses
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download nltk data for tokenization
nltk.download('punkt')
# Download nltk data for stopwords
nltk.download('stopwords')
# Create stemmer object
stemming = PorterStemmer()
# As input data is in english, use english stopwords for removing high frequency words
stops = set(stopwords.words("english"))


# Method called during text vectorization to perform data cleanup
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html, '[%s]' % re.escape(string.punctuation), '')


# Read training or test files
def read_input_file(file_path, training=True):
    # Open and read file contents
    fin = open(file_path)
    data = fin.read()
    fin.close()
    # Every line in file is a json object.
    # Irrespective of file to be read read every json object and extract response, context
    tweets = [json.loads(jline) for jline in data.splitlines()]
    # I could not figure out how to feed context and response separately in this model, thus concatenating the two
    tweet_responses = [clean_input(" ".join([item.get("response"), " ".join(item.get("context"))])) for item in tweets]
    # If training file is being read, need to read the labels for each tweet
    if training:
        # Convert label into numeric values, SARCASM as 0 and NOT_SARCASM as 1
        tweet_labels = [0 if item.get("label") == "SARCASM" else 1 for item in tweets]
        return tweet_responses, tweet_labels
    # if the file type is not training, we will encounter this code path and will parse tweet ids
    ids = [item.get("id") for item in tweets]
    return ids, tweet_responses


# Method to clean data while parsing from file
def clean_input(seq):
    # remove common regex like emojis, symbols, etc
    # reference https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    regrex_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    seq = regrex_pattern.sub(r'', seq)
    # remove all html tags
    words = nltk.word_tokenize(re.sub("<.*?>", " ", seq.lower()))
    # remove punctuations
    token_words = [w for w in words if w.isalpha()]
    # stemming
    stemmed_words = [stemming.stem(word) for word in token_words]
    # remove stop words
    clean_words = [w for w in stemmed_words if not w in stops]
    return " ".join(clean_words)


if __name__ == '__main__':
    training_file = 'data/train.jsonl'
    test_file = 'data/test.jsonl'
    # Read training file
    training_responses, training_labels = read_input_file(training_file)
    # Red test file
    tweet_ids, test_tweets = read_input_file(test_file, training=False)

    # Split the dataset into training and evaluation datasets
    train_responses, eval_responses, train_labels, eval_labels = \
        sk.train_test_split(np.array(training_responses), np.array(training_labels), train_size=0.8)

    # Start pre-processing
    tensor_train_labels = tf.constant(train_labels)
    tensor_eval_labels = tf.constant(eval_labels)
    max_features = 125000
    sequence_length = 500
    # Create TextVectorization object to vectorize the text dataset
    text_vector = TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Adapt the training dataset for model to learn vocabulary
    text_vector.adapt(train_responses)
    # Vectorize training dataset
    train_dataset = text_vector(train_responses)
    # Vectorize evaluation dataset
    eval_dataset = text_vector(eval_responses)
    # Vectorize test dataset
    test_data_set = text_vector(test_tweets)

    # Build LSTM model
    model = tf.keras.Sequential([
        # Define input layer taking vocabulary from adapt step above
        layers.Embedding(input_dim=len(text_vector.get_vocabulary()),
                         output_dim=64,
                         mask_zero=True),
        # Define bidirectional LSTM model
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        # Add hidden layer with activation tanh
        tf.keras.layers.Dense(32, activation='tanh'),
        # Add dropout
        layers.Dropout(0.2),
        # Add output layer which holds the prediction probability
        layers.Dense(1)])

    # Print the model summary before training
    model.summary()

    # Compile model with adam optimizer
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    # In local testing, local_accuracy stops increasing after 4 epochs
    epochs = 4
    # Train and evaluate the model by asking model to fit to the training datasets
    history = model.fit(train_dataset, tensor_train_labels, epochs=epochs,
                        validation_data=(eval_dataset, tensor_eval_labels),
                        verbose=2)

    # Predict test dataset
    results = tf.sigmoid(model.predict(test_data_set))

    # Create answer.txt for submission from predictions
    fin = open("answer.txt", "w")
    idx = 0
    for x in np.nditer(results):
        fin.write("{},{}{}".format(tweet_ids[idx], "SARCASM" if x < 0.5 else "NOT_SARCASM", "\n" if idx < len(tweet_ids) - 1 else ""))
        idx += 1
    fin.close()
