import json
from scripts.clustering_primaryTrainingOrchestrator import *
from scripts.clustering_secondaryTrainingOrchestrator import *
from scripts.clusteringClients import ClusterClients
from scripts.federatedLearningOrchestrator import FederatedLearningOrchestrator

if __name__ == "__main__":
    JSON_FILE_PATH = "config_files/test_env.json"
    args = None
    with open(JSON_FILE_PATH, 'r') as jsonfile:
        args = json.load(jsonfile)

    print("Arguments received from test_env.json: {}".format(args))
    
    clientPoolCount = args["clientPoolCount"]
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
            federated_object.orchestrate()
    else:
        fed_client_proportion = args["federatedLearning"]["client_proportion"]
        fed_epochs =  args["federatedLearning"]["epochs"]
        fed_batch_size =  args["federatedLearning"]["batch_size"]
        fed_rounds =  args["federatedLearning"]["rounds"]
        federated_object = FederatedLearningOrchestrator()
        federated_object.scenarioSetter(fed_client_proportion, fed_epochs, fed_batch_size, fed_rounds)
        federated_object.modelDetailsSetter()
        federated_object.clientCountSetter(client_count=clientPoolCount)
        federated_object.orchestrate()

    # Testing
    print(federated_object.test_metrics_list_during_training)