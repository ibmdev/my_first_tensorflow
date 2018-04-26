import os
import pandas as pd
import tensorflow as tf


TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

LOCAL_TRAIN_URL = "/datasets/iris_training.csv"
LOCAL_TEST_URL = "/datasets/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth','PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# ignore warning : CPU does not support AVX
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# logging verbosity
tf.logging.set_verbosity(tf.logging.ERROR)


def remote_download():
    # Télecharge les fichiers csv et les pose dans users/.keras/datasets
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    return train_path, test_path

def local_download():
    # Retourne la localisation des fichiers csv contenus dans le répertoire datasets du filesystem
    parent = os.path.dirname(os.getcwd())
    return parent+LOCAL_TRAIN_URL, parent+LOCAL_TEST_URL

def load_data(y_name='Species'):
    train_path, test_path = local_download()
    # Chargement des fichiers à partir de Pandas
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)
    return (train_x, train_y), (test_x, test_y)

def train_input_fn(features, labels, batch_size):
    # Convertit les entrées en DataSets
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Mélanger(Shuffle), répéter (repeat) et batcher (batch) les exemples
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    
    return dataset




