import numpy as np
from collections import defaultdict
import random
from datetime import datetime
from sklearn_extra.cluster import KMedoids
import pandas as pd

def find_top_k_candidates(windowing_sets, interest_parameters, N, no_of_clusters, no_of_candidates):
    """
    Function to compute the top-K windowing-set based on user-preference
        windowing_sets (list): A list of windowing_sets
        no_of_candidates (int): no of candidates to be returned
        interest_parameters (dict):
        model_info (dict):
        mongo_client (pymongo.MongoClient):
    """
    global DSIR_DB, DSAR_DB

    X = np.zeros((N, len(interest_parameters)))
    for index, record in enumerate(windowing_sets):
        for i, parameter in enumerate(interest_parameters):
            X[index][i] = record[parameter]
        # End of loop
    # End of loop

    kmediod = KMedoids(no_of_clusters).fit(X)
    # pruning candidate_ws_list
    candidate_ws_list = [windowing_sets[index] for index in kmediod.medoid_indices_]
    return block_nested_loops_skyline(candidate_ws_list, interest_parameters, no_of_candidates)


def block_nested_loops_skyline(windowing_sets, interest_parameters, no_of_candidates):
    """
    Function to compute topK skylines
        Parameters:
            windowing_sets (list):
            no_of_candidates (int): Number of top K candidates
            interest_parameters (dict):
            config (dict):
        Returns:
            candidate_list (list):
    """
    no_of_windowing_sets = len(windowing_sets)
    candidate_list = list()

    # Base Case
    if no_of_windowing_sets <= no_of_candidates:
        return windowing_sets

    while len(candidate_list) < no_of_candidates:
        i = 0
        skyline_candidate_index_list = []
        no_of_windowing_sets = len(windowing_sets)
        while i < no_of_windowing_sets:
            j = 0
            del_list = []
            while j < len(skyline_candidate_index_list):
                index = skyline_candidate_index_list[j]
                parameter_value_dict = windowing_sets[index]
                i_dominate = 0
                j_dominate = 0
                for parameter in parameter_value_dict:
                    if interest_parameters[parameter] == "max":
                        if parameter_value_dict[parameter] > windowing_sets[i][parameter]:
                            # skyline_candidate dominates in dimension parameter
                            j_dominate += 1
                        elif parameter_value_dict[parameter] < windowing_sets[i][parameter]:
                            i_dominate += 1

                    elif interest_parameters[parameter] == "min":
                        if parameter_value_dict[parameter] < windowing_sets[i][parameter]:
                            # skyline_candidate dominates in dimension parameter
                            j_dominate += 1
                        elif parameter_value_dict[parameter] > windowing_sets[i][parameter]:
                            i_dominate += 1
                # End of loop
                if j_dominate > 0 and i_dominate == 0:
                    # skyline_candidate dominates windowing_sets[i]
                    i += 1
                    if i == no_of_windowing_sets:
                        break
                elif i_dominate > 0 and j_dominate == 0:
                    # windowing_sets[i] dominates the skyline_candidate
                    del_list.append(j)
                j += 1
            # End of while
            # Deleting the dominated candidates in the skyline
            for index in reversed(del_list):
                del skyline_candidate_index_list[index]

            # windowing_sets[i] is not dominated by all skyline_candidate
            if i < no_of_windowing_sets:
                skyline_candidate_index_list.append(i)

            i = i + 1
        # End of loop

        for index in skyline_candidate_index_list:
            candidate_list.append(windowing_sets[index])

        for index in reversed(skyline_candidate_index_list):
            del windowing_sets[index]
    # End of while

    return candidate_list[:no_of_candidates]


def perform_scalability_N(iter, no_of_sample_parameters):
    N_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,
              2400, 2500, 2600, 2700, 2800, 2900, 3000]
    interest_parameters = {"param_" + str(i): "min" for i in range(0, no_of_sample_parameters)}
    interest_parameters["score"] = "max"
    compute_time = defaultdict(list)
    for N in N_list:
        for i in range(0, iter):
            windowing_sets = generate_windowing_set(no_of_sample_parameters, N)
            start = datetime.now()
            # windowing_sets, interest_parameters, N, no_of_clusters, no_of_candidates
            find_top_k_candidates(windowing_sets, interest_parameters, N, 50, 20)
            end = datetime.now()
            compute_time[N].append((end - start).total_seconds())
            # print("completed" + str(i) + "iteration for " + str(N))
    # End of loop

    b = np.zeros((len(N_list), 1))
    for i, N in enumerate(N_list):
        b[i] = np.mean(compute_time[N])

    p = pd.DataFrame(b)
    p.index = N_list
    writer = pd.ExcelWriter("scalability-N.xlsx", engine='xlsxwriter')
    p.to_excel(writer, sheet_name="compute time")
    writer.save()
    writer.close()


def perform_scalability_cluster_size(iter, no_of_sample_parameters):
    N = 2000
    interest_parameters = {"param_" + str(i): "min" for i in range(0, no_of_sample_parameters)}
    interest_parameters["score"] = "max"
    compute_time = defaultdict(list)
    cluster_size_list = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150,
                        1200, 1250, 1300, 1350, 1400, 1450, 1500]
    for cluster_size in cluster_size_list:
        for i in range(0, iter):
            windowing_sets = generate_windowing_set(no_of_sample_parameters, N)
            start = datetime.now()
            # windowing_sets, interest_parameters, N, no_of_clusters, no_of_candidates
            find_top_k_candidates(windowing_sets, interest_parameters, N, cluster_size, 20)
            end = datetime.now()
            compute_time[cluster_size].append((end - start).total_seconds())
        print("completed " + str(cluster_size))

    b = np.zeros((len(cluster_size_list), 1))
    for i, cluster_size in enumerate(cluster_size_list):
        b[i] = np.mean(compute_time[cluster_size])

    p = pd.DataFrame(b)
    p.index = cluster_size_list
    writer = pd.ExcelWriter("scalability-no-of-clusters.xlsx", engine='xlsxwriter')
    p.to_excel(writer, sheet_name="compute time")
    writer.save()
    writer.close()


def perform_scalability_no_of_points(iter, no_of_sample_parameters):
    no_of_points_list = [i for i in range(0, 501, 20)]
    no_of_points_list[0] = 1
    cluster_size = 1000
    N = 2000
    interest_parameters = {"param_" + str(i): "min" for i in range(0, no_of_sample_parameters)}
    interest_parameters["score"] = "max"
    compute_time = defaultdict(list)
    for no_of_points in no_of_points_list:
        for i in range(0, iter):
            windowing_sets = generate_windowing_set(no_of_sample_parameters, N)
            start = datetime.now()
            # windowing_sets, interest_parameters, N, no_of_clusters, no_of_candidates
            find_top_k_candidates(windowing_sets, interest_parameters, N, cluster_size, no_of_points)
            end = datetime.now()
            compute_time[no_of_points].append((end - start).total_seconds())
        print("completed " + str(no_of_points))

    b = np.zeros((len(no_of_points_list), 1))
    for i, no_of_points in enumerate(no_of_points_list):
        b[i] = np.mean(compute_time[no_of_points])

    p = pd.DataFrame(b)
    p.index = no_of_points_list
    writer = pd.ExcelWriter("scalability-no-of-points.xlsx", engine='xlsxwriter')
    p.to_excel(writer, sheet_name="compute time")
    writer.save()
    writer.close()


def perform_scalability_no_of_interest_parameters(iter):
    no_of_sample_parameters_list = [i for i in range(1, 31, 2)]
    compute_time = defaultdict(list)
    N = 2000
    cluster_size = 1000
    no_of_points = 300
    for no_of_sample_parameters in no_of_sample_parameters_list:
        interest_parameters = {"param_" + str(i): "min" for i in range(0, no_of_sample_parameters)}
        interest_parameters["score"] = "max"
        for i in range(0, iter):
            windowing_sets = generate_windowing_set(no_of_sample_parameters, N)
            start = datetime.now()
            # windowing_sets, interest_parameters, N, no_of_clusters, no_of_candidates
            find_top_k_candidates(windowing_sets, interest_parameters, N, cluster_size, no_of_points)
            end = datetime.now()
            compute_time[no_of_sample_parameters].append((end - start).total_seconds())
        print("completed " + str(no_of_sample_parameters))
    # End of loop

    b = np.zeros((len(no_of_sample_parameters_list), 1))
    for i, no_of_sample_parameters in enumerate(no_of_sample_parameters_list):
        b[i] = np.mean(compute_time[no_of_sample_parameters])

    p = pd.DataFrame(b)
    p.index = no_of_sample_parameters_list
    writer = pd.ExcelWriter("scalability-no-of-sample-parameters.xlsx", engine='xlsxwriter')
    p.to_excel(writer, sheet_name="compute time")
    writer.save()
    writer.close()


def generate_windowing_set(no_of_sample_parameters, N):
    windowing_sets = list()
    parameter_name = ["param_" + str(i) for i in range(0, no_of_sample_parameters)]
    for i in range(0, N):
        windowing_set = dict()
        for param_name in parameter_name:
            windowing_set[param_name] = random.randint(0, 100)
        # End of loop
        windowing_set["score"] = round(random.random(), 2)
        windowing_sets.append(windowing_set)
    return windowing_sets


if __name__ == '__main__':
    # perform_scalability_N(50, 15)
    # perform_scalability_cluster_size(50, 15)
    perform_scalability_no_of_points(50, 15)
    # perform_scalability_no_of_interest_parameters(50)
