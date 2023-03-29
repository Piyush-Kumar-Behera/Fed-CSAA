import nest_asyncio
import tensorflow_federated as tff
import collections
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random

class FederatedLearningOrchestrator:
    '''
        This class contains methods to initiate a workflow for the Federated
        Learning Process.
        Constants: 
            G -> No. of grouped clients participating for a shared model
                (An abstract constant for large number of clients and 
                clustering of clients scenarios)
            C -> Fraction of clients that are participating in the current 
                round of Federated Learning (refer communication-Efficient 
                Learning of Deep Networks from Decentralized Data 
                - McMahan et. al.)
            E -> Number of Epochs of training in each round of FL
            B -> Batch Size of the local training in client device    
            R -> R is the number of rounds of training
    '''
    dataset_element_spec = None
    def __init__(self):
        self.G = None
        self.C = None
        self.E = None
        self.B = None        
        self.grouped_clients = []
        self.dataset = None

    def scenarioSetter(self, C, E, B, R, G=None, grouped_clients=None,
                        dataset="emnist", shuffle_buffer=100, prefetch_buffer=10,
                        preprocessing_operation_type = "flatten", agg_algo = "fedAvg"):
        '''
            Initializes the Federated Learning Setting
        '''
        self.C = C
        self.E = E
        self.B = B
        self.R = R
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.dataset_name = dataset
        self.op_type = preprocessing_operation_type
        self.agg_algo = agg_algo

        if dataset is not None:
            self.dataset = dataset

        self.is_G_intialized = False
        if G is not None:
            if grouped_clients is not None:
                self.G = G
                self.grouped_clients = grouped_clients
                self.is_G_intialized = True
            else:
                print("grouped_clients argument missing, " +
                "hence skipping G initialization. G should be " +
                "initialized only with grouped_clients")

        

    def modelDetailsSetter(self, model_fn_id = "keras_dnn_mnist_simple"):
        is_valid_input = True
        if model_fn_id == "keras_dnn_mnist_simple":
            self.model_id = model_fn_id
        else:
            is_valid_input = False
       
        if not is_valid_input:
            raise Exception("Inputs to modelDetailsSetter is not supported.")

    @staticmethod
    def create_keras_dnn_mnist_simple():
        return tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(784,)),
            tf.keras.layers.Dense(10, kernel_initializer='zeros'),
            tf.keras.layers.Softmax()])
    
    @staticmethod
    def model_keras_dnn_mnist_simple():
        keras_model = FederatedLearningOrchestrator.create_keras_dnn_mnist_simple()
        return tff.learning.from_keras_model(
            keras_model,
            input_spec=FederatedLearningOrchestrator.dataset_element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalCrossentropy()])

    def make_federated_data(self, data, client_ids):
        return [
            self.preprocess_federated_datasets(data
                        .create_tf_dataset_for_client(client), self.op_type)
            for client in client_ids
        ]

    def preprocess_federated_datasets(self, dataset, type="flatten"):
        def batch_format_flatten_fn_for_emnist(element):
            '''
                Flatten a batch of pixels and return the features as an OrderedDict
            '''
            return collections.OrderedDict(
                x=tf.reshape(element['pixels'], [-1, 784]),
                y=tf.reshape(element['label'], [-1,1]))

        if self.dataset_name == "emnist":
            if type == "flatten":
                return dataset.repeat(self.E).shuffle(self.shuffle_buffer, seed=1).batch(
                    self.B).map(batch_format_flatten_fn_for_emnist).prefetch(self.prefetch_buffer)
            else:
                is_valid_input = False
        else:
            is_valid_input = False
        if not is_valid_input:
            raise Exception("The support for the dataset={} or operation={} currently does not exist"
                            .format(self.dataset_name, type))

            
    def orchestrate(self):
        print("Step 1: Initalizing the environment...")
        nest_asyncio.apply()
        np.random.seed(0)
        tff.federated_computation(lambda: "Hello from tff!")

        print("Step 2: Downloading dataset...")
        if self.dataset is "emnist":
            data_train, data_test = tff.simulation.datasets.emnist.load_data()
            self.data_train = data_train
            self.data_test = data_test
        else:
            print("Dataset is {}, which is not supported"
                    .format(self.dataset))
            raise Exception('Set a valid dataset using scenarioSetter')

        print("Step 3: Validating client group for Federated Training...")
        sample_clients = []
        if self.is_G_intialized:
            for client_id in self.grouped_clients:
                if client_id not in self.data_train.client_ids:
                    print("Client {} not in training data hence skipping...")
                else:
                    sample_clients.append(client_id)
        else:
            sample_clients = self.data_train.client_ids
        
        print("Updating Grouped Clients Attributes...")
        self.grouped_clients = sample_clients
        self.G = len(self.grouped_clients)

        print("Step 4: Creating Federated train data...")
        self.federated_train_data = self.make_federated_data(self.data_train, 
                                                        self.grouped_clients)
        FederatedLearningOrchestrator.dataset_element_spec = \
                                    self.federated_train_data[0].element_spec

        print("Step 5: Setting up training model and parameters...")
        if self.model_id == "keras_dnn_mnist_simple" and self.agg_algo == "fedAvg":
            self.iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
                model_fn=FederatedLearningOrchestrator.model_keras_dnn_mnist_simple,
                client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
                server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))
            self.train_state = self.iterative_process.initialize()
            print("Initialized iterative process variable: {}".format(
                self.iterative_process.initialize.type_signature.formatted_representation()))
            
        print("Step 6: Federated Training Rounds Begins...")
        
        if self.C < 0 or self.C > 1:
            raise Exception("Value of C must be in range [0,1]")
        
        self.metric_capture = []
        # acc => actual client count
        self.acc = int(self.C * self.G)
        RANDOM_MUL, RANDOM_ADD = 17, 11
        for round_no in range(1, self.R):
            random.seed(RANDOM_MUL * round_no + RANDOM_ADD)
            client_final_list = random.sample(self.federated_train_data, self.acc)
            result = self.iterative_process.next(self.train_state, client_final_list)
            self.train_state = result.state
            train_metrics = result.metrics
            self.metric_capture.append(train_metrics)
            print('round {:2d}, metrics={}'.format(round_no, train_metrics))