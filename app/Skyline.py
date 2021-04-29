import pymongo
import numpy as np
from app import ds_utils
from collections import defaultdict
import random
from bson.objectid import ObjectId
from sklearn_extra.cluster import KMedoids

CONFIG: dict = None
JOB_DB: pymongo.collection.Collection = None
DSAR_DB: pymongo.collection.Collection = None
DSIR_DB: pymongo.collection.Collection = None


def initialize_parameter_dict(dsir_id_list, interest_parameters):
    global JOB_DB, DSIR_DB

    parameter_values_dict = defaultdict(list)
    for dsir_id in dsir_id_list:
        dsir = DSIR_DB.find_one({"_id": ObjectId(dsir_id)})
        if dsir["created_by"] == "PostSynchronizationManager":
            parent_dsir_list = dsir["parents"]
            jobs_list = list(JOB_DB.find({"output_dsir": {"$in": parent_dsir_list}}))
        else:
            jobs_list = list([JOB_DB.find_one({"output_dsir": dsir["_id"]})])

        for job in jobs_list:
            for variable, value in job["variables"].items():
                if variable in interest_parameters:
                    parameter_values_dict[variable].append(value)

    parameter_avg_value_dict = dict()
    for parameter, value_list in parameter_values_dict.items():
        parameter_avg_value_dict[parameter] = np.sum(value_list) / len(value_list)

    return parameter_avg_value_dict


def connect_to_mongo(MONGO_CLIENT):
    """
    Function to connect to Mongo, and setup the databases
    Parameters:
        mongo_client(pymongo.MongoClient)
    """
    global CONFIG, HISTORY_DB, DSIR_DB, JOB_DB, DSAR_DB

    CONFIG = MONGO_CLIENT["ds_config"]["collection"].find_one({})
    HISTORY_DB = MONGO_CLIENT["ds_provenance"]["history"]
    DSIR_DB = MONGO_CLIENT["ds_results"]["dsir"]
    DSAR_DB = MONGO_CLIENT["ds_results"]["dsar"]
    JOB_DB = MONGO_CLIENT["ds_results"]["jobs"]


# Entry Point
def find_top_k_candidates(windowing_sets, no_of_candidates, interest_parameters, model_info, mongo_client):
    """
    Function to compute the top-K windowing-set based on user-preference
        windowing_sets (list): A list of windowing_sets
        no_of_candidates (int): no of candidates to be returned
        interest_parameters (dict):
        model_info (dict):
        mongo_client (pymongo.MongoClient):
    """
    global DSIR_DB, DSAR_DB

    connect_to_mongo(mongo_client)

    candidate_ws_list = list()
    valid_parameters = set()
    dependent_models = list()
    if "upstream_models" in model_info:
        dependent_models.extend([model for model in model_info["upstream_models"].values()])

    for parameter_config in model_info["sampled_variables"].values():
        valid_parameters.add(parameter_config["name"])
    # End of loop

    for model in dependent_models:
        info = ds_utils.access_model_by_name(CONFIG, model)
        for parameter_config in info["sampled_variables"].values():
            valid_parameters.add(parameter_config["name"])

    interest_parameters = {parameter_name: strategy for parameter_name, strategy in interest_parameters.items() if parameter_name in valid_parameters}
    X = np.zeros((len(windowing_sets), len(interest_parameters)))
    for index, record in enumerate(windowing_sets):
        dsir_list = record[0]
        parameter_value_dict = initialize_parameter_dict(dsir_list, interest_parameters)
        for i, parameter in enumerate(interest_parameters):
            if parameter in parameter_value_dict:
                X[index][i] = parameter_value_dict[parameter]
            else:
                X[index][i] = 0

        parameter_value_dict["score"] = record[3]
        candidate_ws_list.append([record[0], parameter_value_dict, record[2]])
    # End of loop

    # N = random.randint(no_of_candidates, len(windowing_sets))
    diff = abs(no_of_candidates - len(windowing_sets))
    N = no_of_candidates + int(model_info["wm_settings"]["candidates_threshold"] * diff)
    kmediod = KMedoids(N).fit(X)
    # pruning candidate_ws_list
    candidate_ws_list = [candidate_ws_list[index] for index in kmediod.medoid_indices_]
    interest_parameters["score"] = "max"

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
    if no_of_windowing_sets == no_of_candidates:
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

