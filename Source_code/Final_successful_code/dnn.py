import json
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn.model_selection as sk
import tensorflow_hub as hub
import os

tf.get_logger().setLevel('INFO')


def read_input_file(file_path, training=True):
    # Open and read file contents
    fin = open(file_path)
    data = fin.read()
    fin.close()
    # Every line in file is a json object.
    # Irrespective of file to be read read every json object and extract response, context
    tweets = [json.loads(jline) for jline in data.splitlines()]
    tweet_responses = [[item.get("response"), " ".join(item.get("context"))] for item in tweets]
    # If training file is being read, need to read the labels for each tweet
    if training:
        # Convert label into numeric values, SARCASM as 0 and NOT_SARCASM as 1
        tweet_labels = [0 if item.get("label") == "SARCASM" else 1 for item in tweets]
        return tweet_responses, tweet_labels
    # if the file type is not training, we will encounter this code path and will parse tweet ids
    tweet_ids = [item.get("id") for item in tweets]
    return tweet_ids, tweet_responses


def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to DataSet
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(10000).repeat()
    return dataset.batch(batch_size)


def test_input_fn(features, batch_size=256):
    # Convert the inputs to DataSet
    return tf.data.Dataset.from_tensor_slices((dict(features))).batch(batch_size)


if __name__ == '__main__':
    training_file = 'data/train.jsonl'
    test_file = 'data/test.jsonl'
    # Read training file
    training_responses, training_labels = read_input_file(training_file)
    # Read test file
    test_tweet_ids, test_responses = read_input_file(test_file, training=False)

    # Split the dataset into training and evaluation datasets
    train_conversation, eval_conversations, train_labels, eval_labels = \
        sk.train_test_split(np.array(training_responses), np.array(training_labels), train_size=0.8)

    # Start pre-processing
    df = pd.DataFrame(data=train_conversation, columns=["response", "context"])
    training_label_series = pd.Series(train_labels)

    eval_df = pd.DataFrame(data=eval_conversations, columns=["response", "context"])
    eval_label_series = pd.Series(eval_labels)

    test_ds = pd.DataFrame(data=test_responses, columns=["response", "context"])

    print("Creating text columns")
    # Feeding text as columns - https://medium.com/engineering-zemoso/text-classification-bert-vs-dnn-b226497c9de7
    os.environ['TFHUB_CACHE_DIR'] = 'tf_cache/'
    TFHUB_URL = "https://tfhub.dev/google/universal-sentence-encoder/2"
    embedded_text_response_column = hub.text_embedding_column(key="response", module_spec=TFHUB_URL)
    embedded_text_context_column = hub.text_embedding_column(key="context", module_spec=TFHUB_URL)
    print("Created text columns")

    # DNN Classifier reference - https://www.tensorflow.org/tutorials/estimator/premade
    # crelu activation whitepaper reference - https://arxiv.org/pdf/1603.05201.pdf
    classifier = tf.estimator.DNNClassifier(
        feature_columns=[embedded_text_response_column, embedded_text_context_column],
        hidden_units=[64, 32, 16],
        dropout=0.2,
        n_classes=2,
        optimizer="SGD",
        activation_fn=tf.nn.crelu)
    print("Created classifier, starting training")
    classifier.train(input_fn=lambda: input_fn(df, training_label_series, training=True), steps=5000)
    print("Trained classifier, initiating evaluation")
    result = classifier.evaluate(input_fn=lambda: input_fn(eval_df, eval_label_series, training=False))
    print("Evaluated classifier, printing results")
    print("Accuracy: {}".format(result["accuracy"]))
    print("Precision: {}".format(result["precision"]))
    print("Recall: {}".format(result["recall"]))
    print("Loss: {}".format(result["loss"]))

    # Use trained classifier to predict for the test dataset
    predictions = classifier.predict(input_fn=lambda: test_input_fn(test_ds))

    # Create answer.txt for submission from predictions
    fin = open("answer.txt", "w")
    idx = 0
    print("Process predictions")
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        fin.write("{},{}{}".format(test_tweet_ids[idx], "SARCASM" if class_id == 0 else "NOT_SARCASM",
                                   "\n" if idx < len(test_tweet_ids) - 1 else ""))
        idx += 1
    fin.close()
