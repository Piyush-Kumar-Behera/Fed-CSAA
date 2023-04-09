import nest_asyncio
import tensorflow_federated as tff
import collections
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
from sys import exit
import copy

def create_iterative_process(federated_train_data):
    def create_keras_model():
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
        
    def model_fn():
        keras_model = create_keras_model()
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=federated_train_data[0].element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    @tff.tf_computation
    def server_init():
        model = model_fn()
        return model.trainable_variables
    
    @tff.federated_computation
    def initialize_fn():
        return tff.federated_value(server_init(), tff.SERVER)
    
    model_weights_type = server_init.type_signature.result
    whimsy_model = model_fn()
    tf_dataset_type = tff.SequenceType(whimsy_model.input_spec)

    federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
    federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

    @tf.function
    def client_update(model, dataset, server_weights, client_optimizer):
        """Performs training (using the server model weights) on the client's dataset."""
        # Initialize the client model with the current server weights.
        client_weights = model.trainable_variables
        # Assign the server weights to the client model.
        tf.nest.map_structure(lambda x, y: x.assign(y),
                            client_weights, server_weights)

        # Use the client_optimizer to update the local model.
        for batch in dataset:
            with tf.GradientTape() as tape:
                # Compute a forward pass on the batch of data
                outputs = model.forward_pass(batch)

        # Compute the corresponding gradient
        grads = tape.gradient(outputs.loss, client_weights)
        grads_and_vars = zip(grads, client_weights)

        # Apply the gradient using a client optimizer.
        client_optimizer.apply_gradients(grads_and_vars)

        return client_weights
    
    @tf.function
    def server_update(model, mean_client_weights):
        """Updates the server model weights as the average of the client model weights."""
        model_weights = model.trainable_variables
        # Assign the mean client weights to the server model.
        tf.nest.map_structure(lambda x, y: x.assign(y),
                                model_weights, mean_client_weights)
        return model_weights
    
    @tff.tf_computation(tf_dataset_type, model_weights_type)
    def client_update_fn(tf_dataset, server_weights):
        model = model_fn()
        client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        return client_update(model, tf_dataset, server_weights, client_optimizer)

    @tff.tf_computation(model_weights_type)
    def server_update_fn(mean_client_weights):
        model = model_fn()
        return server_update(model, mean_client_weights)

    @tff.federated_computation(federated_server_type, federated_dataset_type)
    def next_fn(server_weights, federated_dataset):
        # Broadcast the server weights to the clients.
        server_weights_at_client = tff.federated_broadcast(server_weights)

        # Each client computes their updated weights.
        client_weights = tff.federated_map(
            client_update_fn, (federated_dataset, server_weights_at_client))

        # The server averages these updates.
        mean_client_weights = tff.federated_mean(client_weights)

        # The server updates its model.
        server_weights = tff.federated_map(server_update_fn, mean_client_weights)

        return server_weights

    federated_algorithm = tff.templates.IterativeProcess(
        initialize_fn=initialize_fn,
        next_fn=next_fn
    )

    return federated_algorithm        

class ClusteringPrimaryTraining:
    '''
        The Fed-CSAA algorithm consists of clustering clients based on label inclination.
        The clustering has majorly two steps:
            1. Training all the layers of the model
            2. Training the last layer of the model for clustering

        This class consists of the first step orchestrator.
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
    
    def orchestrate(self):
        print("Step 1: Initalizing the environment...")
        nest_asyncio.apply()
        np.random.seed(0)
        tff.federated_computation(lambda: "Hello from tff!")

        print("Step 2: Downloading dataset...")
        if self.dataset == "emnist":
            data_train, data_test = tff.simulation.datasets.emnist.load_data()
            self.data_train = data_train
            self.data_test = data_test
        else:
            print("Dataset is {}, which is not supported"
                    .format(self.dataset))
            raise Exception('Set a valid dataset using scenarioSetter')

        print("Step 3: Clients are getting chosen randomly...")
        if self.client_count is None:
            raise Exception("Client Count must be initialized using ClientCountSetter")
        self.client_ids_list = self.data_train.client_ids
        random.seed(self.client_count_seed)
        self.client_set = random.sample(self.client_ids_list, self.client_count)

        print("Step 4: Creating Federated Learning Ready Data...")    
        self.make_federated_data(self.client_set, "train")

        print("Step 5: Creating Iterative Process...")
        self.federated_algorithm = create_iterative_process(self.federated_train_data)

        print("Step 6: Initializing Iterative Process...")
        server_state = self.federated_algorithm.initialize()

        print("Step 7: Starting federated training rounds...")
        for round in self.R:
            server_state = self.federated_algorithm.next(server_state, 
                                                         self.federated_train_data)
            
        updated_wts = copy.deepcopy(server_state)
        return updated_wts
