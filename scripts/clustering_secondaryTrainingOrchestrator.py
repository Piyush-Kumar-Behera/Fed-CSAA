import nest_asyncio
import tensorflow_federated as tff
import collections
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
from sys import exit
import copy

def client_update(model, dataset, client_optimizer, client_loss):
    """Performs training (using the server model weights) on the client's dataset."""
    model.compile(optimizer=client_optimizer, loss=client_loss)
    model.fit(dataset)
    client_weights = model.trainable_variables
    return client_weights

class ClusteringSecondaryTraining:
    '''
        The Fed-CSAA algorithm consists of clustering clients based on label inclination.
        The clustering has majorly two steps:
            1. Training all the layers of the model
            2. Training the last layer of the model for clustering

        This class consists of the second step orchestrator.
    '''
    
    def __init__(self):
        self.dataset = None
        self.UPDATED_WTS = None
        self.client_count = None
        self.client_count_seed = None
        self.model_fn_id = "leNet-CNN"
        self.client_set = None

    def scenarioSetter(self, epochs, batch_size, rounds, 
                        dataset="emnist", shuffle_buffer=100, 
                        prefetch_buffer=10):
        self.E = epochs
        self.B = batch_size
        self.R = rounds
        self.dataset = dataset
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer

    def modelDetailsSetter(self, model_fn_id = "leNet-CNN"):
        if model_fn_id not in ["leNet-CNN"]:
            raise Exception("Invalid model_fn_id")
        self.model_fn_id = model_fn_id
    
    def clientCountSetter(self, client_count, seed = 42):
        self.client_count = client_count
        self.client_count_seed = seed
    
    def preprocess(self, dataset):

        def batch_format_fn(element):
            """Returns dataset in shape (28, 28, 1)"""
            return (tf.reshape(element['pixels'], [-1, 28, 28, 1]), 
                    tf.reshape(element['label'], [-1, 1]))

        return dataset.repeat(self.E).shuffle(self.shuffle_buffer, seed=1).batch(
            self.B).map(batch_format_fn).prefetch(self.prefetch_buffer)

    def make_federated_data(self, client_set, data_split = "train"):
        if data_split == "train":
            data = self.data_train
        else:
            data = self.data_test

        self.federated_train_data = [
            self.preprocess(data.create_tf_dataset_for_client(x))
            for x in client_set]
    
    def create_keras_model(self):
        initializer = tf.keras.initializers.GlorotNormal(seed=0)
        return tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), activation='relu', kernel_initializer=initializer),
            tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), activation='relu', kernel_initializer=initializer),
            tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)),
            tf.keras.layers.Conv2D(filters=60, kernel_size=(3,3), activation='relu', kernel_initializer=initializer),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(84, kernel_initializer=initializer, activation  = 'relu'),
            tf.keras.layers.Dense(10, kernel_initializer=initializer, activation = 'softmax')
        ])

    def create_half_frozen_keras_model(self, updated_weights):
        hf_model = self.create_keras_model()

        new_updated_weights = copy.deepcopy(updated_weights)
        tf.nest.map_structure(lambda x, y: x.assign(y),
                            hf_model.trainable_variables, new_updated_weights)

        layer_idx = [0, 2, 4, 6]
        for idx in layer_idx:
            hf_model.layers[idx].trainable = False

        return hf_model

    def orchestrate(self, primary_wts):
        print("Step 1: Downloading dataset...")
        if self.dataset == "emnist":
            data_train, data_test = tff.simulation.datasets.emnist.load_data()
            self.data_train = data_train
            self.data_test = data_test
        else:
            print("Dataset is {}, which is not supported"
                    .format(self.dataset))
            raise Exception('Set a valid dataset using scenarioSetter')

        print("Step 2: Clients are getting chosen randomly...")
        if self.client_count is None:
            raise Exception("Client Count must be initialized using ClientCountSetter")
        self.client_ids_list = self.data_train.client_ids
        random.seed(self.client_count_seed)
        self.client_set = random.sample(self.client_ids_list, self.client_count)

        print("Step 3: Creating Federated Learning Ready Data...")    
        self.make_federated_data(self.client_set, "train")

        print("Step 4: Fetching last layer learnings for clients")
        self.client_label_incl = []
        for client_data in tqdm(self.federated_train_data):
            temp_model = self.create_half_frozen_keras_model(primary_wts)
            client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
            client_loss = tf.keras.losses.SparseCategoricalCrossentropy()
            new_updates = client_update(temp_model, client_data, client_optimizer, client_loss)
            self.client_label_incl.append(new_updates)
        print("Successfully fetched the last layer parameters...")
        
        self.client_label_incl_numpy = []
        for cli in self.client_label_incl:
            cli1 = cli[0].numpy()
            cli2 = cli[1].numpy().reshape(-1, cli1.shape[1])
            new_cli = np.append(cli1, cli2, axis=0)
            final_cli = new_cli.flatten()    
            self.client_label_incl_numpy.append(final_cli)
            
        return self.client_label_incl_numpy
