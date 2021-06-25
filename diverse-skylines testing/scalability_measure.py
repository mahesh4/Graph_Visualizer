import numpy as np
from collections import defaultdict
import random
from sklearn_extra.cluster import KMedoids

def find_top_k_candidates(windowing_sets, no_of_candidates, interest_parameters, N):
    """
    Function to compute the top-K windowing-set based on user-preference
        windowing_sets (list): A list of windowing_sets
        no_of_candidates (int): no of candidates to be returned
        interest_parameters (dict):
        model_info (dict):
        mongo_client (pymongo.MongoClient):
    """
    global DSIR_DB, DSAR_DB


    candidate_ws_list = list()
    interest_parameters = set()
    X = np.zeros((len(windowing_sets), len(interest_parameters)))
    for index, record in enumerate(windowing_sets):
        for i, parameter in enumerate(interest_parameters):
            X[index][i] = record[parameter]
        # End of loop
    # End of loop

    kmediod = KMedoids(N).fit(X)
    # pruning candidate_ws_list
    candidate_ws_list = [candidate_ws_list[index] for index in kmediod.medoid_indices_]
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
        return [(windowing_set[0], windowing_set[2]) for windowing_set in windowing_sets]

    while len(candidate_list) < no_of_candidates:
        i = 0
        skyline_candidate_index_list = []
        no_of_windowing_sets = len(windowing_sets)
        while i < no_of_windowing_sets:
            j = 0
            del_list = []
            while j < len(skyline_candidate_index_list):
                index = skyline_candidate_index_list[j]
                parameter_value_dict = windowing_sets[index][1]
                i_dominate = 0
                j_dominate = 0
                for parameter in parameter_value_dict:
                    if interest_parameters[parameter] == "max":
                        if parameter_value_dict[parameter] > windowing_sets[i][1][parameter]:
                            # skyline_candidate dominates in dimension parameter
                            j_dominate += 1
                        elif parameter_value_dict[parameter] < windowing_sets[i][1][parameter]:
                            i_dominate += 1

                    elif interest_parameters[parameter] == "min":
                        if parameter_value_dict[parameter] < windowing_sets[i][1][parameter]:
                            # skyline_candidate dominates in dimension parameter
                            j_dominate += 1
                        elif parameter_value_dict[parameter] > windowing_sets[i][1][parameter]:
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
            candidate_list.append((windowing_sets[index][0], windowing_sets[index][2]))

        for index in reversed(skyline_candidate_index_list):
            del windowing_sets[index]
    # End of while

    return candidate_list[:no_of_candidates]

def generate_windowing_set(no_sample_parameters, )