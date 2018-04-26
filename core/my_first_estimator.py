import argparse
import tensorflow as tf
import my_first_dataset


# Parser pour les arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,help='number of training steps')

def main(argv):
    
    # Hello World TensorFlow
    hello = tf.constant('Hello TensorFlow')
    sess = tf.Session()
    print(sess.run(hello))
    print('TensorFlow version : ' +str(tf.__version__))

    # Récupération des arguments (optionnels) saisis en entrée
    args = parser.parse_args(argv[1:])
    print('paramètre batch_size : ' + str(args.batch_size))
    print('paramètre train_steps : ' + str(args.train_steps))

    # Chargement des données
    (train_x, train_y), (test_x, test_y) = my_first_dataset.load_data()
    
    # FeatureColumns: tools for ingesting and representing features
    # Numeric_Column represents real valued or numerical features
    features_columns = []
    for key in train_x.keys():
        features_columns.append(tf.feature_column.numeric_column(key=key))

    # Build a 2 hidden layer DNN of 10 nodes each with 3 classes to choose for the model
    # Construction d'un Deep Neural Network à 2 couches cachées de 10 noeuds chacun et 3 classes pour le modèle
    # tf.estimator.DNNClassifier(feature_columns,hidden_units=[],n_classes) : A classifier for TensorFlow DNN models
    classifier = tf.estimator.DNNClassifier(feature_columns=features_columns, hidden_units=[10,10], n_classes=3)
    
    # Train the model
    classifier.train(input_fn=lambda:my_first_dataset.train_input_fn(train_x, train_y,args.batch_size),steps=args.train_steps)

    

    # Display results
    
    print('Train X Size =  ' + str(train_x.keys().size))
    print('Train Y Size =  ' + str(train_y.size))
    print('Test X Size =  ' + str(test_x.keys().size))
    print('Test Y Size = ' + str(test_y.size))
    

tf.app.run(main)