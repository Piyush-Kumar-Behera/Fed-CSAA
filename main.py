import json
from scripts.clustering_primaryTrainingOrchestrator import *
from scripts.clustering_secondaryTrainingOrchestrator import *
from scripts.clusteringClients import ClusterClients
from scripts.federatedLearningOrchestrator import FederatedLearningOrchestrator
import pickle
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param')
    parser_args = parser.parse_args()
    param_file_path = parser_args.param

    if param_file_path is None:
        param_file_path = "test_env.json"
    
    JSON_FILE_PATH = "config_files/{}".format(param_file_path)
    print("looking for params in {}".format(JSON_FILE_PATH))

    OUTPUT_FOLDER_PATH = "output_files/"
    args = None
    with open(JSON_FILE_PATH, 'r') as jsonfile:
        args = json.load(jsonfile)

    print("Arguments received from test_env.json: {}".format(args))
    
    clientPoolCount = args["clientPoolCount"]
    testingName = args["name"]
    if args.get("useFedCSAA"):
        pt_epochs = args["primary_training"]["epochs"]
        pt_batch_size = args["primary_training"]["batch_size"]
        pt_rounds = args["primary_training"]["rounds"]

        print("Starting Primary Training for Clustering...")
        pt_clust_object = ClusteringPrimaryTraining()
        pt_clust_object.scenarioSetter(pt_epochs, pt_batch_size, pt_rounds)
        pt_clust_object.modelDetailsSetter()
        pt_clust_object.clientCountSetter(clientPoolCount)
        wts_after_pt = pt_clust_object.orchestrate()
        print("Finished Primary Training for Clustering...")

        st_epochs = args["secondary_training"]["epochs"]
        st_batch_size = args["secondary_training"]["batch_size"]
        st_rounds = args["secondary_training"]["rounds"]

        print("Starting Secondary Training for Clustering...")
        st_clust_object = ClusteringSecondaryTraining()
        st_clust_object.scenarioSetter(st_epochs, st_batch_size, st_rounds)
        st_clust_object.modelDetailsSetter()
        st_clust_object.clientCountSetter(clientPoolCount)
        lbl_inclination, client_set_st = st_clust_object.orchestrate(wts_after_pt)
        print("Finished Secondary Training for CLustering...")

        print("Starting Clustering...")
        no_of_clusters = args["clustering"]["clusterCount"]
        clustering_object = ClusterClients()
        clustered_clients = clustering_object.clusterClientsMethod(lbl_inclination, client_set_st, no_of_clusters)
        print("Clustering Finished...")
        print("Clustered Clients: {}".format(clustered_clients))

        for cluster_idx in range(len(clustered_clients)):
            OUTPUT_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, "{}-cluster-{}".format(testingName, cluster_idx))
            fed_client_proportion = args["federatedLearning"]["client_proportion"]
            fed_epochs =  args["federatedLearning"]["epochs"]
            fed_batch_size =  args["federatedLearning"]["batch_size"]
            fed_rounds =  args["federatedLearning"]["rounds"]
            client_group_idx = clustered_clients[cluster_idx]
            client_group_idx_count = len(client_group_idx)
            federated_object = FederatedLearningOrchestrator()
            federated_object.scenarioSetter(fed_client_proportion, fed_epochs, fed_batch_size, fed_rounds, 
                                            client_group_idx_count, client_group_idx)
            federated_object.modelDetailsSetter()
            federated_object.orchestrate(is_clustered_case=True, total_clients=clientPoolCount)
            with open(OUTPUT_FILE_PATH, "wb") as fp:
                pickle.dump(federated_object.test_metrics_list_during_training, fp)
    else:
        OUTPUT_FILE_PATH = os.path.join(OUTPUT_FOLDER_PATH, "{}-without-clustering".format(testingName))
        fed_client_proportion = args["federatedLearning"]["client_proportion"]
        fed_epochs =  args["federatedLearning"]["epochs"]
        fed_batch_size =  args["federatedLearning"]["batch_size"]
        fed_rounds =  args["federatedLearning"]["rounds"]
        federated_object = FederatedLearningOrchestrator()
        federated_object.scenarioSetter(fed_client_proportion, fed_epochs, fed_batch_size, fed_rounds)
        federated_object.modelDetailsSetter()
        federated_object.clientCountSetter(client_count=clientPoolCount)
        federated_object.orchestrate()
        with open(OUTPUT_FILE_PATH, "wb") as fp:
            pickle.dump(federated_object.test_metrics_list_during_training, fp)

    # Testing
    print(federated_object.test_metrics_list_during_training)