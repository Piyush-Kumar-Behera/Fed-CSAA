import nest_asyncio
import tensorflow_federated as tff
import collections
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import random
from sys import exit

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
        self.client_count = None

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

        
    def clientCountSetter(self, client_count, seed = 42):
        self.client_count = client_count
        self.client_init_seed = seed

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
            tf.keras.layers.InputLayer(input_shape=(3072,)),
            tf.keras.layers.Dense(1000, activation='relu'),
            tf.keras.layers.Dense(20, kernel_initializer='zeros'),
            tf.keras.layers.Softmax()])
    
    @staticmethod
    def model_keras_dnn_mnist_simple():
        keras_model = FederatedLearningOrchestrator.create_keras_dnn_mnist_simple()
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=FederatedLearningOrchestrator.dataset_element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

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
                x=tf.reshape(element['image'], [-1, 3072]),
                y=tf.reshape(element['coarse_label'], [-1,1]))

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
        elif self.dataset == "cifar":
            data_train, data_test = tff.simulation.datasets.cifar100.load_data()
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
        if self.client_count != None:
            random.seed(self.client_init_seed)
            self.grouped_clients = random.sample(sample_clients, self.client_count)
            self.G = self.client_count
        else:
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

        if self.client_count == None and not self.is_G_intialized:
            print("Training on complete client list... Do you want to continue (Y/N)")
            input_from_user = input()
            if input_from_user.upper() not in ['Y', 'YES']:
                return None
        print("Step 6: Federated Training Rounds Begins...")
        
        if self.C < 0 or self.C > 1:
            raise Exception("Value of C must be in range [0,1]")
        
        self.metric_capture = []
        # acc => actual client count
        self.acc = int(self.C * self.G)
        RANDOM_MUL, RANDOM_ADD = 17, 11
        self.test_metrics_list_during_training = []
        for round_no in range(1, self.R):
            random.seed(RANDOM_MUL * round_no + RANDOM_ADD)
            client_final_list = random.sample(self.federated_train_data, self.acc)
            result = self.iterative_process.next(self.train_state, client_final_list)
            self.train_state = result.state
            train_metrics = result.metrics
            self.metric_capture.append(train_metrics)
            print('round {:2d}, metrics={}'.format(round_no, train_metrics))
            print("Evaluating on test-data...")
            self.model_accuracy_on_test_data = self.evaluate_trained_model()
            self.sca_test_data = self.model_accuracy_on_test_data['client_work']['eval']['current_round_metrics']['sparse_categorical_accuracy']
            self.test_metrics_list_during_training.append(self.sca_test_data)

    def evaluate_trained_model(self):
        print("In evaluation...Evaluation must be executed only after orchestrate is" + \
               "successfully executed...")

        self.federated_test_data = self.make_federated_data(self.data_test, 
                                                            self.grouped_clients)
        self.evaluation_process = tff.learning.algorithms.build_fed_eval(
            model_fn = FederatedLearningOrchestrator.model_keras_dnn_mnist_simple)
        self.evaluation_state = self.evaluation_process.initialize()
        self.model_wts_eval = self.iterative_process.get_model_weights(self.train_state)
        self.evaluation_state = self.evaluation_process.set_model_weights(self.evaluation_state, 
                                                                        self.model_wts_eval)
        
        self.evaluation_output = self.evaluation_process.next(self.evaluation_state,
                                                              self.federated_test_data)
        
        print("Metrics: {}".format(str(self.evaluation_output.metrics)))

        return self.evaluation_output.metrics