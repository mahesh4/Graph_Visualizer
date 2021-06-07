import collections
import random
import time
import openpyxl
import xlsxwriter
import bson
import pymongo
from bson.objectid import ObjectId
from collections import defaultdict
from sklearn.cluster import SpectralClustering
import itertools
import copy
import numpy
from scipy import stats as scipy_stats
import networkx as nx
import pandas as pd

from app import ds_utils
from app import ProvenanceCriteria
from app import wm_utils
from app import Skyline
from app.timeline_visualizer.model_graph import ModelGraph
from app.timeline_visualizer.timelines import Timelines

DS_CONFIG: dict = dict()
MONGO_CLIENT: pymongo.MongoClient = None
KEPLER_DB: pymongo.collection.Collection = None
DSIR_DB: pymongo.collection.Collection = None
JOBS_DB: pymongo.collection.Collection = None
HISTORY_DB: pymongo.collection.Collection = None
DB_PROVENANCE: pymongo.collection.Collection = None
PURGE = True
global_list = list()
completion_list = list()


def _connect_to_mongo():
    global DS_CONFIG, DSIR_DB, MONGO_CLIENT, KEPLER_DB, JOBS_DB, HISTORY_DB, DB_PROVENANCE

    DS_CONFIG = MONGO_CLIENT["ds_config"]["collection"].find_one({})
    KEPLER_DB = MONGO_CLIENT["ds_state"]["kepler"]
    DSIR_DB = MONGO_CLIENT["ds_results"]["dsir"]
    JOBS_DB = MONGO_CLIENT["ds_results"]["jobs"]
    HISTORY_DB = MONGO_CLIENT["ds_provenance"]["history"]
    DB_PROVENANCE = MONGO_CLIENT["ds_provenance"]["provenance"]


def increase_temporal_context(model, model_info):
    global EXIT_CODE_OVERRIDE, KEPLER_DB

    current_model_state = KEPLER_DB.find_one({"model_type": model})
    current_end = float(current_model_state["temporal_context"]['end'])
    shift_size = float(model_info["temporal"]["shift_size"])
    # Update begin time
    updated_begin = current_end + shift_size
    KEPLER_DB.update_one({"model_type": model_info}, {"$set": {"temporal_context.begin": updated_begin}})

    # Update end time
    updated_end = updated_begin + float(model_info["temporal"]['output_window'])
    KEPLER_DB.update_one({"model_type": model}, {"$set": {"temporal_context.end": updated_end}})
    # if this model has produced results which reach to the end of the simulation window, then it should stop now
    if updated_end >= float(DS_CONFIG["simulation_context"]["temporal"]["end"]):
        EXIT_CODE_OVERRIDE = ds_utils.ExitCode.SimulationComplete


def _perform_output(model, model_info):
    global KEPLER_DB

    state = KEPLER_DB.find_one({"model_type": model})
    dsir_list = list(state["result_pool"]["to_output"])
    # Emptying to_output
    state["result_pool"]["to_output"] = []
    KEPLER_DB.save(state)
    downstream_model_list = list()
    if "downstream_models" in model_info:
        downstream_model_list = list(model_info["downstream_models"].values())

    for dsir_id in dsir_list:
        KEPLER_DB.update_one({"model_type": model}, {"$addToSet": {"result_pool.to_window": dsir_id}})

        for downstream_model in downstream_model_list:
            KEPLER_DB.update_one({"model_type": downstream_model},
                                 {"$addToSet": {"result_pool.to_window": dsir_id}})
        # End of loop
    # End of loop
    increase_temporal_context(model, model_info)
    return


def _create_provenance(target_dsir, model):
    """
    Function to create provenance of the target_dsir, and store it in mongoDB
        Parameters:
            target_dsir (dict): Aggregate DSIR
        Returns:
            None
    """
    global DS_CONFIG, DS_RESULTS, DB_PROVENANCE, JOBS_DB

    workflow_id = DS_CONFIG['workflow_id']
    state = KEPLER_DB.find_one({"model_type": model})
    job_list = list()

    if target_dsir["created_by"] == "JobGateway":
        job_list.append(JOBS_DB.find_one({"output_dsir": target_dsir["_id"]}))
    else:
        for output_dsir in target_dsir["parents"]:
            job_list.append(JOBS_DB.find_one({"output_dsir": output_dsir}))

    # Create Provenance to the target_dsir
    history_id_set = set()
    for job in job_list:
        input_dsirs = job["input_dsir"]  # These are DSIRs created by AM_resampling
        upstream_dsirs = [upstream_dsir_id for input_dsir_id in input_dsirs for upstream_dsir_id in
                          DSIR_DB.find_one({"_id": input_dsir_id})["parents"]]
        for history in state["history"]:
            if set(upstream_dsirs) == set(history["upstream_dsir"]):
                history_id_set.add(history["history_id"])
        # End of loop
    # End of loop

    provenance = ProvenanceCriteria.create_provenance_from_histories(list(history_id_set))
    ProvenanceCriteria.add_target_dsir_to_provenance(provenance, target_dsir)
    DB_PROVENANCE.insert_one({"provenance": provenance, "workflow_id": workflow_id, "dsir_id": target_dsir["_id"]})
    return


def get_parameter_value_dict(dsir_id_list):
    global JOBS_DB

    parameter_value_dict = defaultdict(list)
    for dsir_id in dsir_id_list:
        target_job = JOBS_DB.find_one({"output_dsir": dsir_id})
        for variable, value in target_job["variables"].items():
            parameter_value_dict[variable].append(value)
    return parameter_value_dict


def _perform_clustering(affinity_matrix, no_of_clusters, topK=5):
    # Determining the optimal no of clusters
    affinity_matrix = numpy.asarray(affinity_matrix)
    eigenvalues, eigenvectors = numpy.linalg.eig(affinity_matrix)

    diff = numpy.round(numpy.absolute(numpy.diff(eigenvalues)), 4)
    # print(numpy.where(diff > 0)[0])
    if len(numpy.where(diff > 0)[0]) == 0:
        nb_clusters = affinity_matrix.shape[0]
    else:
        index_largest_gap = numpy.argsort(diff)[::-1][:topK]
        index_largest_gap = index_largest_gap + 1
        if no_of_clusters in index_largest_gap:
            nb_clusters = no_of_clusters
        else:
            nb_clusters = max(index_largest_gap)

    # if index_largest_gap + 1 >= no_of_candidates:
    #     nb_clusters = index_largest_gap + 1
    # else:
    #     nb_clusters = no_of_candidates

    print("Optimal number of clusters is: " + str(nb_clusters))

    clustering = SpectralClustering(n_clusters=nb_clusters, assign_labels="discretize", random_state=0) \
        .fit(affinity_matrix)

    print("Cluster labels: " + str(clustering.labels_))

    return nb_clusters, clustering.labels_


def _perform_postsynchronization(model, begin, output_end, model_info):
    global DSIR_DB, KEPLER_DB

    aggregation_strategy = model_info["psm_settings"]["psm_strategy"]
    state = KEPLER_DB.find_one({"model_type": model})
    dsir_list = state["result_pool"]["to_sync"]
    to_output = list()
    temporal_context = {"begin": begin, "end": output_end}
    shift_size = model_info["temporal"]["shift_size"]
    output_window = model_info["temporal"]["output_window"]
    if aggregation_strategy == 'none':
        for dsir_id in dsir_list:
            target_dsir = DSIR_DB.find_one({"_id": dsir_id})
            _create_provenance(target_dsir, model)
            to_output.append(dsir_id)
        # End of loop
    elif aggregation_strategy == "cluster":
        parametric_similarity = model_info["psm_settings"]["psm_parametric_sim"]
        provenance_similarity = model_info["psm_settings"]["psm_provenance_sim"]

        no_of_candidates = model_info["psm_settings"]["psm_candidates"]
        no_of_dsirs = len(dsir_list)

        # Creating the provenance for each DSIR
        for dsir_id in dsir_list:
            dsir = DSIR_DB.find_one({"_id": dsir_id})
            _create_provenance(dsir, model)
        # End of loop

        dsir_pair_list = [[dsir_list[i], dsir_list[j]] for i in range(no_of_dsirs) for j in range(no_of_dsirs) if i != j]
        similarity_graph = [[1] * no_of_dsirs for _ in range(no_of_dsirs)]
        parametric_compatibility_function = None
        if parametric_similarity:
            if DS_CONFIG["compatibility_settings"]["parametric_mode"] == "conjunction":
                parametric_compatibility_function = wm_utils.conjunction
            elif DS_CONFIG["compatibility_settings"]["parametric_mode"] == "union":
                parametric_compatibility_function = wm_utils.union

        for dsir_pair in dsir_pair_list:
            if dsir_pair[0] == dsir_pair[1]:
                continue

            compatibility_score_list = list()
            if parametric_similarity:
                # Computing Parametric Similarity
                parameter_value_dict = get_parameter_value_dict(dsir_pair)
                compatibility_satisfaction, parametric_metric_score = parametric_compatibility_function(parameter_value_dict, DS_CONFIG)
                compatibility_score_list.append(parametric_metric_score)

            if provenance_similarity:
                metric_score = ProvenanceCriteria.find_provenance_similarity(dsir_pair[0], dsir_pair[1], temporal_context)
                compatibility_score_list.append(metric_score)

            score = scipy_stats.hmean(compatibility_score_list)
            i_index = dsir_list.index(dsir_pair[0])
            j_index = dsir_list.index(dsir_pair[1])
            similarity_graph[i_index][j_index] = score
            similarity_graph[j_index][i_index] = score
        # End of loop

        nb_clusters, cluster_labels = _perform_clustering(similarity_graph, no_of_candidates)

        # Creating the scores for the cluster
        scores = dict()
        for i in range(0, nb_clusters):
            cluster_score = 0
            nb_nodes = 0
            for index in range(len(cluster_labels)):
                if cluster_labels[index] == i:
                    dsir = DSIR_DB.find_one({"_id": dsir_list[index]})
                    job = JOBS_DB.find_one({"_id": dsir["metadata"]["job_id"]})
                    cluster_score += job["relevance"]
                    nb_nodes += 1
            # End of loop
            scores[i] = cluster_score / nb_clusters

        # Sorting the scores in descending order
        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        top_cluster_labels = [k for k, v in scores.items()]

        # Performing aggregation
        for i in range(0, min(no_of_candidates, nb_clusters)):
            dsir_aggr_list = []
            label = top_cluster_labels[i]
            for index in range(len(cluster_labels)):
                if cluster_labels[index] == label:
                    dsir_aggr_list.append(dsir_list[index])
            # End of loop

            new_dsir = _create_dsir(model, begin, output_end, shift_size, output_window, "PostSynchronizationManager")
            new_dsir["parents"] = dsir_aggr_list
            DSIR_DB.save(new_dsir)
            to_output.append(new_dsir["_id"])
            # Adding the children to each dsir in dsir_aggr_list
            for dsir_id in dsir_aggr_list:
                dsir = DSIR_DB.find_one({"_id": dsir_id})
                dsir["children"] = [new_dsir["_id"]]
                DSIR_DB.save(dsir)
            # End of loop

            # creating provenance
            _create_provenance(new_dsir, model)
        # End of loop
    else:
        raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")
    # Updating the state
    state["result_pool"]["to_output"] = to_output
    state["result_pool"]["to_sync"] = []
    KEPLER_DB.save(state)
    return


def _sample_values(model, begin, output_end, model_info):
    global KEPLER_DB, JOBS_DB, DS_CONFIG, DSIR_DB

    state = KEPLER_DB.find_one({"model_type": model})
    shift_size = model_info["temporal"]["shift_size"]
    output_window = model_info["temporal"]["output_window"]
    candidate_list = state["result_pool"]["to_sample"]
    sampling_strategy = DS_CONFIG["compatibility_settings"]["compatibility_strategy"]
    model_vars = list(model_info["sampled_variables"].values())
    to_sync = list()
    job_count = model_info["sm_settings"]["sm_fanout"]
    for candidate_index, dsir_list in enumerate(candidate_list):
        for i in range(job_count):
            new_job = _create_job(model)
            new_job["input_dsir"] = dsir_list

            if sampling_strategy == "independent":
                for each_var in model_vars:
                    var_name = each_var["name"]
                    low = each_var["lower_bound"]
                    high = each_var["upper_bound"]
                    new_job["variables"][var_name] = round(random.uniform(low, high), 2)
                # End of loop
            elif sampling_strategy == "provenance":
                prev_parameter_value_dict = DS_CONFIG["sampled_parameters"]
                if len(prev_parameter_value_dict) == 0:
                    for each_var in model_vars:
                        var_name = each_var["name"]
                        low = each_var["lower_bound"]
                        high = each_var["upper_bound"]
                        new_job["variables"][var_name] = round(random.uniform(low, high), 2)
                else:
                    for each_var in model_vars:
                        var_name = each_var["name"]
                        if var_name in prev_parameter_value_dict[candidate_index]:
                            prev_value = prev_parameter_value_dict[candidate_index][var_name]
                            new_job["variables"][var_name] = min(round(prev_value + each_var["bin_size"], 2), each_var["upper_bound"])
                        else:
                            low = each_var["lower_bound"]
                            high = each_var["upper_bound"]
                            new_job["variables"][var_name] = round(random.uniform(low, high), 2)

            # creating new_dsirs for job <---Execution--->
            new_dsir = _create_dsir(model, begin, output_end, shift_size, output_window, "JobGateway")
            new_dsir["metadata"]["job_id"] = new_job["_id"]
            new_job["output_dsir"] = new_dsir["_id"]
            new_job["relevance"] = round(random.uniform(0, 1), 2)
            new_dsir["parents"] = dsir_list
            DSIR_DB.save(new_dsir)
            # Adding the new dsir to to_sync
            to_sync.append(new_dsir["_id"])
            # Adding children for each dsir in dsir_list
            for dsir_id in dsir_list:
                dsir = DSIR_DB.find_one({"_id": dsir_id})
                dsir["children"].append(new_dsir["_id"])
                DSIR_DB.save(dsir)
            # End of loop
            JOBS_DB.save(new_job)
        # End of loop
    # End of loop
    state["result_pool"]["to_sync"] = to_sync
    state["result_pool"]["to_sample"] = []
    state["subactor_state"] = "PostSynchronizationManager"
    KEPLER_DB.save(state)
    return


def _perform_alignment(model, begin, output_end, model_info):
    global KEPLER_DB, DS_CONFIG, DSIR_DB

    state = KEPLER_DB.find_one({"model_type": model})
    output_window = model_info["temporal"]["output_window"]
    shift_size = model_info["temporal"]["shift_size"]
    for candidate_set in state["result_pool"]["to_align"]:
        model_instances_dict = _group_instances_by_model(candidate_set)
        aligned_dsirs = list()
        for model, instance_list in model_instances_dict.items():
            new_dsir = _create_dsir(model, begin, output_end, shift_size, output_window, "AM_resampling")
            new_dsir["parents"] = instance_list
            aligned_dsirs.append(new_dsir["_id"])
            DSIR_DB.save(new_dsir)
            # Updating the children relationship
            for instance_id in instance_list:
                instance_dsir = DSIR_DB.find_one({"_id": instance_id})
                instance_dsir["children"].append(new_dsir["_id"])
                DSIR_DB.save(instance_dsir)
        # End of loop
        state["result_pool"]["to_sample"].append(aligned_dsirs)
    # End of loop
    state["subactor_state"] = "SamplingManager"
    state["result_pool"]["to_align"] = []
    KEPLER_DB.save(state)


def _power_set(input_iterable: collections.Iterable):
    global DSIR_DB

    record_list = list(input_iterable)
    model_record_dict = defaultdict(dict)
    iter_list = []
    # splitting records by model
    for record_id in record_list:
        record = DSIR_DB.find_one({"_id": record_id})
        begin = record["metadata"]["temporal"]["begin"]
        if begin in model_record_dict[record["metadata"]["model_type"]]:
            model_record_dict[record["metadata"]["model_type"]][begin].append(str(record["_id"]))
        else:
            model_record_dict[record["metadata"]["model_type"]][begin] = [str(record["_id"])]
    # End of loop
    for model_type, record_dict in model_record_dict.items():
        # computing factor
        if model_type == "undefined":
            iter_list.extend(list(record_dict.values()))
        else:
            upstream_model_info = ds_utils.access_model_by_name(DS_CONFIG, model_type)
            l = len(record_list)
            time_list = list(record_dict.keys())
            instance_list = list()
            time_index_power_set = list(itertools.chain.from_iterable(itertools.combinations(time_list, r) for r in range(1, l + 1)))
            for time_index_sublist in time_index_power_set:
                record_sub_list = [record_dict[time] for time in time_index_sublist]
                instance_list.extend(list(itertools.product(*record_sub_list)))
            iter_list.append(instance_list)
    # End of loop
    return itertools.product(*iter_list)


def _check_compatibility_temporal(candidate_sets, begin, model_info):
    # TODO: Need to check
    # records are temporally-ordered
    # if there are enough to satisfy the current window, repackage the constituent DSIRs into a new DSAR
    new_candidates = []
    for candidate in candidate_sets:
        # check to see if this is a valid set of results
        record_set = candidate[0]
        compat_dict = candidate[1]
        compat_dict["temporal"] = _compute_compatibility_temporal(record_set, begin, model_info)
        if compat_dict["temporal"] > 0:
            new_candidates.append([record_set, compat_dict])
    # End of loop
    print(f"\tOf these, {len(new_candidates)} candidate(s) were temporally compatible.")
    return new_candidates


def generate_provenance_score(candidate_sets, temporal_context):
    """
    Function to generate the provenance_score for each candidate_set
        Parameters:
            candidate_sets (list): A list of candidate windowing_set
            temporal_context (dict): Current temporal context of the execution
        Returns:
            new_candidate (list): A list of candidate windowing_set with their corresponding provenance scores computed
    """
    global MONGO_CLIENT

    new_candidates = []
    for candidate_record_set in candidate_sets:
        # check to see if this is a valid set of results
        # TODO: Need to evaluate
        record_set = candidate_record_set[0]
        compat_dict = candidate_record_set[1]
        dsir_set = record_set
        provenance_score = ProvenanceCriteria.find_provenance_score(dsir_set, temporal_context)
        compat_dict["provenance"] = provenance_score
        new_candidates.append([record_set, compat_dict, candidate_record_set[2]])
    # End of loop
    return new_candidates


def _compute_compatibility_temporal(list_of_records, begin, model_info):
    global DSIR_DB, DS_CONFIG

    model_window = _temporal_generation(model_info)
    simulation_end = DS_CONFIG["simulation_context"]["temporal"]["end"]
    input_window = model_info["temporal"]["input_window"]
    # don't allow the required temporal window to extend beyond the simulation...
    window_end = min(begin + input_window, simulation_end)
    old_dsir_list = []
    ending_list = []
    record_cache = dict()
    for record_id in list_of_records:
        record = DSIR_DB.find_one({"_id": record_id})
        record_cache[record_id] = record
        record_model = record["metadata"]["model_type"]

        if model_window[record_model]["satisfied"]:
            continue  # don't process records from models that have already satisfied the current window

        this_begin = model_window[record_model]["begin"]

        if this_begin is None:
            model_window[record_model]["begin"] = record["metadata"]["temporal"]["begin"]
            model_window[record_model]["end"] = record["metadata"]["temporal"]["end"]
            this_begin = model_window[record_model]["begin"]
        else:
            model_window[record_model]["end"] = record["metadata"]["temporal"]["end"]

        if this_begin >= window_end:
            # we see data from the next window;
            #       send the current window (whatever it is, even empty) and stop
            model_window[record_model]["satisfied"] = True
        else:
            # we're looking in the right window, so include this record
            old_dsir_list.append(record_id)

            # and include its end time, so that we can decide to trim it or not
            ending_list.append(model_window[record_model]["end"])

        if model_window[record_model]["end"] >= window_end:
            # our data reaches to the end of the window
            model_window[record_model]["satisfied"] = True

        # our data doesn't span the window; keep looking

    if len(old_dsir_list) != len(list_of_records):
        return 0  # don't allow cases where not all the records are used
        #           (this set of inputs is a superset, so the subset will appear as an alternative candidate)

    # if all models are satisfied, we need to generate DSARs from those records
    overall_satisfaction = True
    for model_type in model_window.keys():
        overall_satisfaction = overall_satisfaction and model_window[model_type]["satisfied"]

    if not overall_satisfaction:
        return 0

    # compute compatibility metric
    meta_dict = {model_type: {"overlaps": 0, "gaps": 0, "end": 0} for model_type in model_window.keys()}
    record_list = list(record_cache.values())
    # sort the records based on start-time
    record_list.sort(key=lambda r: r["metadata"]["temporal"]["begin"])
    for record in record_list:
        model_type = record["metadata"]["model_type"]
        if meta_dict[model_type]["end"] == 0:
            meta_dict[model_type]["end"] = record["metadata"]["temporal"]["end"]
        else:
            overlaps = meta_dict[model_type]["end"] - record["metadata"]["temporal"]["begin"]
            if overlaps > 0:
                meta_dict[model_type]["overlaps"] += overlaps
            elif overlaps < 0:
                # its a gap
                meta_dict[model_type]["gaps"] += overlaps

    # Calculating cumulative overlaps and gaps
    sec_overlap, sec_gap = 0, 0
    for model, meta in meta_dict.items():
        sec_gap += meta["gaps"] + max(0, window_end - meta["end"])
        sec_overlap += meta["overlaps"]

    window_size = input_window
    if window_size == 0:
        return 1  # just in case it's 0, don't want a div by zero

    WINDOWING_POLICY = copy.deepcopy(model_info["wm_settings"])
    if WINDOWING_POLICY["wm_strategy"] == "least_gap":
        gap_weight = 2
        overlap_weight = 0
    elif WINDOWING_POLICY["wm_strategy"] == "least_overlap":
        gap_weight = 0
        overlap_weight = 2
    elif WINDOWING_POLICY["wm_strategy"] == "diverse":
        gap_weight = 2
        overlap_weight = 1
    elif WINDOWING_POLICY["wm_strategy"] == "compatible":
        gap_weight = 2
        overlap_weight = 1
    else:
        raise ValueError("Unknown windowing mode: {0}".format(WINDOWING_POLICY["wm_strategy"]))

    metric = 1 - (1 * ((gap_weight * sec_gap + overlap_weight * sec_overlap) / window_size))
    return metric


def _temporal_generation(model_info):
    # determine the upstream models, so we can synchronize their windows independently
    # that is, ALL upstream models must have provided sufficient records to satisfy a window
    if "upstream_models" not in model_info:
        model_list = []
    else:
        model_list = list(model_info["upstream_models"].values())

    if len(model_list) == 0:
        model_list.append("undefined")
    model_window = dict()
    for model_type in model_list:
        model_window[model_type] = dict.fromkeys(["begin", "end"])
        model_window[model_type]["satisfied"] = False

    return model_window


def _check_compatibility_startstate(candidate_states, model_info, start):
    global DSIR_DB
    # do we have records from the current model, but in a previous window?
    # make sure to include the record which covers or abuts against the current start time

    if DS_CONFIG["simulation_context"]["temporal"]["begin"] == start:
        return None  # this is the first time this model has run - no previous state possible!

    if "stateful" in model_info and model_info["stateful"] is False:
        return None  # this is a state-free model, no need to pass previous state

    new_candidates = []

    for record_id in candidate_states:
        candidate_dsir = DSIR_DB.find_one({"_id": record_id})
        if candidate_dsir["metadata"]["temporal"]["begin"] <= start <= candidate_dsir["metadata"]["temporal"]["end"]:
            new_candidates.append(record_id)

    return new_candidates


def _group_instances_by_model(candidate_set):
    global DSIR_DB

    model_instances_dict = defaultdict(list)
    for dsir_id in candidate_set:
        dsir = DSIR_DB.find_one({"_id": dsir_id})
        model_type = dsir["metadata"]["model_type"]
        model_instances_dict[model_type].append(dsir_id)
    # End of loop
    return model_instances_dict


def _create_dsir(model, start, end, shift_size, output_window, created_by):
    global DS_CONFIG

    # Create a new DSIR template
    new_dsir = dict()
    new_dsir["_id"] = bson.objectid.ObjectId()
    new_dsir["workflow_id"] = DS_CONFIG['workflow_id']
    new_dsir["metadata"] = {
        "temporal": {"begin": start, "end": end, "window_size": output_window, "shift_size": shift_size, "timestamp_list": list()},
        "model_type": model}
    new_dsir["is_aggregated"] = False
    new_dsir["do_visualize"] = False
    # TODO: Need to change here
    new_dsir["created_by"] = created_by
    new_dsir["parents"] = list()
    new_dsir["children"] = list()
    return new_dsir


def _create_job(model):
    global DS_CONFIG

    new_job = dict()
    new_job["_id"] = bson.objectid.ObjectId()
    new_job["workflow_id"] = DS_CONFIG['workflow_id']
    new_job["model_type"] = model
    new_job["created_by"] = "SamplingManager"
    new_job["input_dsirs"] = list()
    new_job["variables"] = dict()
    return new_job


def _split_results(all_results, model, begin, model_info):
    global DSIR_DB

    self, other = [], []
    shift_size = model_info["temporal"]["shift_size"]
    input_window = model_info["temporal"]["input_window"]
    input_end = begin - shift_size + input_window
    for record_id in all_results:
        record = DSIR_DB.find_one({"_id": record_id})
        # removing data which are not compatible for current window
        # TODO: Need to evaluate here
        if record["metadata"]["model_type"] == model and record["metadata"]["temporal"]["end"] == begin - shift_size:
            self.append(record_id)
        elif (begin <= record["metadata"]["temporal"]["begin"] < input_end) or (begin < record["metadata"]["temporal"]["end"] <= input_end) or \
                (record["metadata"]["temporal"]["begin"] <= begin and record["metadata"]["temporal"]["end"] >= input_end):
            other.append(record_id)
        # else:
        # print("not compatible record for current execution: {0}".format(record_id))
    return self, other


def _generate_windowing_sets(model, begin, output_end, model_info):
    global DS_CONFIG, KEPLER_DB, DSIR_DB

    state = KEPLER_DB.find_one({"model_type": model})
    simulation_end = DS_CONFIG["simulation_context"]["temporal"]["end"]
    is_seed = DS_CONFIG["simulation_context"]["temporal"]["begin"] == begin and len(model_info["upstream_models"]) == 0
    input_window = model_info["temporal"]["input_window"]
    if is_seed:
        # TODO:Need to write logic for is_seed
        # creating seed dsir
        seed_dsir = dict()
        seed_dsir["_id"] = bson.objectid.ObjectId()
        seed_dsir["workflow_id"] = DS_CONFIG["workflow_id"]
        seed_dsir["IS_SEED"] = True
        seed_dsir["parents"] = []
        seed_dsir["children"] = []
        seed_dsir["created_by"] = "WindowManager"
        seed_dsir["metadata"] = {"temporal": {"begin": begin, "end": output_end}}
        seed_dsir["metadata"]["model_type"] = "is_seed"
        temporal_context = {"begin": begin, "end": min(begin + input_window, DS_CONFIG["simulation_context"]["temporal"]["end"])}
        history = ProvenanceCriteria.create_history_from_windowing_set([seed_dsir], temporal_context)
        # Writing history to DB
        history_id = HISTORY_DB.insert_one({"history": history, "workflow_id": DS_CONFIG["workflow_id"]})
        history_list = list()
        history_list.append({"history_id": history_id.inserted_id, "upstream_dsir": [seed_dsir["_id"]]})
        state["subactor_state"] = "AlignmentManager"
        state["history"] = history_list
        state["result_pool"]["to_align"].append([seed_dsir["_id"]])
        KEPLER_DB.save(state)
        DSIR_DB.save(seed_dsir)
        return

    # splitting results
    self_results, other_results = _split_results(state["result_pool"]["to_window"], model, begin, model_info)

    state_candidates = _check_compatibility_startstate(self_results, model_info, begin)

    if state_candidates is not None and len(state_candidates) == 0:
        raise Exception("[ERROR] Stateful model expected previous window data, but didn't receive it!")

    raw_powerset = list(_power_set(other_results))
    input_powerset = []
    for entry in raw_powerset:
        metadata = dict()
        input_powerset.append([[ObjectId(_id) for _id_list in entry for _id in _id_list], metadata])
    # End of loop

    print('no of candidates generated were: ' + str(len(input_powerset)))
    # performing temporal compatibility
    candidate_sets = _check_compatibility_temporal(input_powerset, begin, model_info)

    if not is_seed:
        # inject the self-states back into the candidates now that temporal compatibility is done
        if state_candidates is not None:
            new_candidate_sets = list()
            for each_self_state in state_candidates:
                candidate_sets_cpy = copy.deepcopy(candidate_sets)
                for each_candidate in candidate_sets_cpy:
                    new_candidate_component = bson.objectid.ObjectId(each_self_state)
                    old_candidate_tuple = list(each_candidate[0])
                    old_candidate_tuple.insert(0, new_candidate_component)
                    each_candidate[0] = tuple(old_candidate_tuple)
                    each_candidate[1]["state"] = 1.0  # state compatibility is boolean, either 0 (false) or 1 (true)
                # End of loop
                new_candidate_sets.extend(candidate_sets_cpy)
            # End of loop
            candidate_sets = new_candidate_sets
            if len(candidate_sets) == 0:
                candidate_sets = [[[bson.objectid.ObjectId(dsir_id)], {"temporal": 1, "state": 1}] for dsir_id in state_candidates]

    # find parametric similarity score for the candidate set of records
    candidate_sets = wm_utils.compute_parametric_similarity(candidate_sets, model, MONGO_CLIENT)

    # find provenance score for the candidate sets of records
    temporal_context = {"begin": begin, "end": min(begin + input_window, simulation_end)}
    candidate_sets = generate_provenance_score(candidate_sets, temporal_context)

    # remove the DSARs of MY_MODEL if not Stateful
    if not model_info["stateful"] and state_candidates is not None:
        for each_candidate in candidate_sets:
            record_set = each_candidate[0]
            new_record_set = [record_id for record_id in record_set if DSIR_DB.find_one({"_id": record_id})["metadata"]["model_type"] != model]
            each_candidate[0] = new_record_set
        # End of loop

    # recompute weights as the harmonic mean of the compatibilities
    hard_metric = {"temporal"}
    soft_metric = {"parametric", "provenance"}
    for index in range(len(candidate_sets)):
        hard_score = list()
        soft_score = list()
        for key, score in candidate_sets[index][1].items():
            if key in hard_metric:
                hard_score.append(score)
            elif key in soft_metric:
                soft_score.append(score)
        # End of loop
        metrics = list()
        if len(soft_score) > 0:
            metrics.append(numpy.mean(soft_score))
        if len(hard_score) > 0:
            metrics.extend(hard_score)
        hmean = scipy_stats.hmean(metrics)
        candidate_sets[index].append(hmean)
    # End of loop

    # sort the candidates by their weighted compatibility
    candidate_sets.sort(key=lambda each: each[3], reverse=True)

    history_list = list()
    parameter_list = list()
    WINDOWING_POLICY = model_info["wm_settings"]
    if WINDOWING_POLICY["wm_strategy"] == "compatible":
        # repackage the most-compatible candidate for use by the model, give better records the chance to appear
        dsir_list = candidate_sets[0][0]
        state["result_pool"]["to_align"].append(dsir_list)
        parameter_list.append(candidate_sets[0][2])
        history = ProvenanceCriteria.create_history_from_windowing_set(dsir_list, temporal_context)
        # Writing history to DB
        history_id = HISTORY_DB.insert_one({"history": history, "workflow_id": DS_CONFIG["workflow_id"]})
        history_list.append({"history_id": history_id.inserted_id, "upstream_dsir": candidate_sets[0][0]})
    else:
        # send multiple candidates to the aligner
        K = int(WINDOWING_POLICY["wm_fanout"])
        interest_parameters = {param_dict["parameter"]: param_dict["strategy"] for param_dict in DS_CONFIG["exploration"].values()}
        candidate_sets = Skyline.find_top_k_candidates(candidate_sets, K, interest_parameters, model_info, MONGO_CLIENT)
        for each_candidate in candidate_sets:
            dsir_list = each_candidate[0]
            state["result_pool"]["to_align"].append(dsir_list)
            parameter_list.append(each_candidate[1])
            history = ProvenanceCriteria.create_history_from_windowing_set(dsir_list, temporal_context)
            # Writing history to DB
            history_id = HISTORY_DB.insert_one({"history": history, "workflow_id": DS_CONFIG["workflow_id"]})
            history_list.append({"history_id": history_id.inserted_id, "upstream_dsir": each_candidate[0]})
        # End of loop

    print("All candidates generated.")
    DS_CONFIG["sampled_parameters"] = parameter_list
    MONGO_CLIENT["ds_config"].collection.save(DS_CONFIG)
    # go to alignment manager
    # If WM receives more than one model, candidates from different models will be stored in the same candidate list.
    state["subactor_state"] = "AlignmentManager"
    state["history"] = history_list
    KEPLER_DB.save(state)
    return


def execute_model(model, begin, output_end, model_info):
    _generate_windowing_sets(model, begin, output_end, model_info)
    _perform_alignment(model, begin, output_end, model_info)
    _sample_values(model, begin, output_end, model_info)
    _perform_postsynchronization(model, begin, output_end, model_info)
    _perform_output(model, model_info)
    return


def _temporal_quantization(begin, output_window, model_info):
    # note that this reads the CURRENT temporal window, or generates one if none exists
    # it doesn't iterate to the next window step (that's done when the new records are saved)
    print("Fetching current temporal window...")
    state = KEPLER_DB.find_one({"model_type": model_info["name"]})
    if state["temporal_context"]["begin"] != 0:
        print("Valid temporal window found.")
    else:
        # There is no shift in the first window
        print("No valid window found, building...")
        state["temporal_context"]["begin"] = begin
        # the "end" of a model is the end of the current window, but this is confusing and should be fixed
        state["temporal_context"]["end"] = begin + output_window
        KEPLER_DB.save(state)
    return state["temporal_context"]["begin"]


def _increase_temporal_context(begin, output_window, shift_size, model_info):
    global DS_CONFIG, KEPLER_DB

    updated_begin = begin + output_window + shift_size
    KEPLER_DB.update_one({"model_type": model_info["name"]},
                         {"$set": {"temporal_context.begin": updated_begin}})
    updated_end = updated_begin + float(model_info["temporal"]['output_window'])
    KEPLER_DB.update_one({"model_type": model_info["name"]},
                         {"$set": {"temporal_context.end": updated_end}})
    if updated_end >= float(DS_CONFIG["simulation_context"]["temporal"]["end"]):
        return True
    else:
        return False


def simulate_model(model):
    global DS_CONFIG

    model_info = ds_utils.access_model_by_name(DS_CONFIG, model)
    simulation_begin = DS_CONFIG["simulation_context"]["temporal"]["begin"]
    simulation_end = DS_CONFIG["simulation_context"]["temporal"]["end"]
    # begin = simulation_begin
    # begin = 1602053874
    output_window = model_info["temporal"]["output_window"]
    print(model_info["temporal"])
    input_window = model_info["temporal"]["input_window"]
    shift_size = model_info["temporal"]["shift_size"]
    print("----------------> starting model " + model + " <---------------------------")
    terminate = False
    while not terminate:
        t_begin = _temporal_quantization(float(DS_CONFIG["simulation_context"]["temporal"]["begin"]),
                                         float(model_info["temporal"]["output_window"]), model_info)
        output_end = t_begin + output_window
        ds_utils.set_model(model)
        print(model + " starting with begin " + str(t_begin) + " and end " + str(output_end))
        execute_model(model, t_begin, output_end, model_info)
        terminate = _increase_temporal_context(t_begin, float(model_info["temporal"]["output_window"]), float(model_info["temporal"]["shift_size"]),
                                               model_info)
        print(model + " completed with begin " + str(t_begin) + " and end " + str(output_end))

    print("---------------> completed " + model + " <-----------------------")


def _absolute(causal_depth, causal_width, causal_edges):
    G = nx.DiGraph()

    for w in range(causal_width):
        nodes_list = [str(w) + '_' + str(d) for d in range(causal_depth)]
        G.add_nodes_from(nodes_list)
        if w - 1 >= 0:
            # Linking causal-edges
            parent = numpy.random.permutation(causal_depth)
            for idx, l in enumerate(parent):
                parent_id = str(w - 1) + '_' + str(l)
                node_id = nodes_list[idx]
                G.add_edge(parent_id, node_id)
            # Adding the remaining causal_edges for each node
            if causal_edges - 1 > 0:
                for idx, node_id in enumerate(nodes_list):
                    parent_nodes = random.sample(range(0, causal_depth - 1), causal_edges - 1)
                    for parent_node in parent_nodes:
                        if parent_node == parent[idx]:
                            parent_id = str(w - 1) + '_' + str(parent_node + 1)
                        else:
                            parent_id = str(w - 1) + '_' + str(parent_node)
                        G.add_edge(parent_id, node_id)
    return G


def generate_workflow(causal_depth, causal_width, causal_edges):
    G = _absolute(causal_depth, causal_width, causal_edges)
    model_dict_template = {
        "am_settings": {
            "am_gap": "ignore",
            "am_overlap": "ignore"
        },
        "name": "",
        "psm_settings": {
            "psm_provenance_sim": True,
            "psm_parametric_sim": True,
            "psm_strategy": "cluster",
            "psm_candidates": 4
        },
        "sampled_variables": {
            "variable_0": {
                "bin_size": 1,
                "lower_bound": 100,
                "upper_bound": 200,
                "name": "_sm_var_0"
            },
            "variable_1": {
                "bin_size": 0.5,
                "lower_bound": 1,
                "upper_bound": 10,
                "name": "_sm_var_1"
            },
        },
        "sm_settings": {
            "sm_fanout": 5,
        },
        "stateful": True,
        "temporal": {
            "input_window": 43200,
            "output_window": 43200,
            "shift_size": 0
        },
        "wm_settings": {
            "wm_fanout": 5,
            "candidates_threshold": 1,
            "wm_strategy": "least_gap"
        },
        "upstream_models": {},
        "downstream_models": {}
    }
    simulation_context = {"temporal": {"begin": 1602010674, "end": 1602182780}}
    compatibility_settings = {"compatibility_strategy": "provenance", "parametric_mode": "union", "provenance_size": 10800}
    DS_CONFIG = {
        "simulation_context": simulation_context,
        "compatibility_settings": compatibility_settings,
        "model_settings": {},
        "exploration": {},
        "sampled_parameters": {}
    }
    for model_no, model in enumerate(list(G.nodes())):
        model_dict = copy.deepcopy(model_dict_template)
        model_dict["name"] = "model_" + model
        for idx, upstream_model in enumerate(list(G.predecessors(model))):
            model_dict["upstream_models"]["upstream_model_" + str(idx)] = "model_" + upstream_model
        for idx, downstream_model in enumerate(list(G.successors(model))):
            model_dict["downstream_models"]["downstream_model_" + str(idx)] = "model_" + downstream_model
        for sm_var_config in model_dict["sampled_variables"].values():
            sm_var_config["name"] = "model_" + model + sm_var_config["name"]
        DS_CONFIG["model_settings"]["model_" + str(model_no)] = model_dict

    OLD_DS_CONFIG = MONGO_CLIENT["ds_config"]["collection"].find_one({})
    if OLD_DS_CONFIG is not None:
        DS_CONFIG["workflow_id"] = OLD_DS_CONFIG["workflow_id"]
        MONGO_CLIENT["ds_config"]["collection"].delete_one({"_id": OLD_DS_CONFIG["_id"]})
    MONGO_CLIENT["ds_config"]["collection"].insert_one(DS_CONFIG)
    return


def reset_mongo():
    global PURGE

    # load some information about the workflow
    ds_config = MONGO_CLIENT.ds_config.collection.find_one()
    if "workflow_id" in ds_config:
        old_workflow_id = ds_config["workflow_id"]
    else:
        old_workflow_id = None
    new_workflow_id = bson.objectid.ObjectId()
    # store activation / deactivation times
    if "activation_time" in ds_config.keys():
        old_activation_time = ds_config["activation_time"]
    else:
        old_activation_time = "undefined"
    time_now = time.time()

    # shift to the new workflow
    ds_config["activation_time"] = time_now
    ds_config["workflow_id"] = new_workflow_id
    MONGO_CLIENT.ds_config.collection.save(ds_config)

    # # Store the parameters in the workflow
    # ds_config["sampled_parameters"] = parameters

    # store a snapshot of the previous configuration
    ds_config["activation_time"] = old_activation_time
    ds_config["deactivation_time"] = time_now
    ds_config["_id"] = new_workflow_id  # rotate document ID only in the workflows collection
    ds_config.pop("workflow_id", None)
    MONGO_CLIENT.ds_config.workflows.save(ds_config)

    del ds_config["sampled_parameters"]

    if old_workflow_id is not None:
        print(f"Resetting everything from workflow ID {old_workflow_id}...")

    # fix the kepler state
    kepler_state = MONGO_CLIENT.ds_state.kepler.find()
    for model_ptr in kepler_state:
        # temporal context
        model_ptr["temporal_context"] = dict()
        model_ptr["temporal_context"]["begin"] = 0
        model_ptr["temporal_context"]["end"] = 0
        model_ptr["temporal_context"]["window_size"] = 0
        model_ptr["timeline"] = []
        model_ptr["history"] = []

        # subactor state
        model_ptr["subactor_state"] = "WindowManager"

        # result pools
        for queue in model_ptr["result_pool"]:
            model_ptr["result_pool"][queue] = []

        MONGO_CLIENT.ds_state.kepler.save(model_ptr)

    # fix the cluster state in mongo (WARNING: JobGateway states may still be messed up)
    cluster_state = list(MONGO_CLIENT.ds_state.cluster.find())

    # add missing instances to the cluster, if needed (and configured to do so)
    local_model_list = dict()
    with open("/etc/hosts", "r") as hostfile:
        hostfile_data = hostfile.readlines()
    for entry in hostfile_data:
        if "ds_model" not in entry:
            continue
        entry = entry.replace("\t", " ")
        entry = entry.replace("\n", "")
        split_data = entry.split(" ")
        ip, full_hostname = split_data[0], split_data[-1]
        hostname = full_hostname.replace("ds_model_", "")
        instance_num = int(hostname.split("_")[-1])
        model_name = hostname.replace(f"_{instance_num}", "")
        local_model_list[model_name + "_" + str(instance_num)] = {"status": "idle", "time_updated": 0.0,
                                                                  "hostname": full_hostname,
                                                                  "model_type": model_name,
                                                                  "_id": bson.objectid.ObjectId(),
                                                                  "instance": int(instance_num), "ip": ip,
                                                                  "pool": dict.fromkeys(
                                                                      ["running", "waiting", "fetching"], [])}

    # reset the states for the instances
    for instance_record in cluster_state:
        instance_record["pool"] = dict.fromkeys(["running", "waiting", "fetching"], [])
        instance_record["status"] = "idle"
        instance_record["time_updated"] = 0
        instance_key = instance_record["model_type"] + "_" + str(int(instance_record["instance"]))
        local_model_list.pop(instance_key, None)
        MONGO_CLIENT.ds_state.cluster.save(instance_record)
    local_model_list = list(local_model_list.values())
    if len(local_model_list) > 0:
        MONGO_CLIENT.ds_state.cluster.insert_many(local_model_list)

    if PURGE and old_workflow_id is not None:
        print("Purging data...")
        # purge DSFRs based on parent DSIRs under the old workflow ID
        relevant_dsir_list = list(MONGO_CLIENT.ds_results.dsir.find(filter={"workflow_id": old_workflow_id}))
        if len(relevant_dsir_list) == 0:
            print("No DSIRs to remove, skipping...")
        for each_dsir in relevant_dsir_list:
            print(f"Removing DSFRs for DSIR {each_dsir}...")
            MONGO_CLIENT.ds_results.dsfr.delete_many(filter={"parent": each_dsir["_id"]})
            print(f"Removing DSIR {each_dsir} itself...")
            MONGO_CLIENT.ds_results.dsir.delete_one(each_dsir)

        # purge other records that contain workflow ID
        print(f"Removing associated DSARs, if any...")
        MONGO_CLIENT.ds_results.dsar.delete_many(filter={"workflow_id": old_workflow_id})
        print(f"Removing associated Timelines, if any...")
        MONGO_CLIENT.ds_results.timeline.delete_many(filter={"workflow_id": old_workflow_id})
        print(f"Removing associated jobs, if any...")
        MONGO_CLIENT.ds_results.jobs.delete_many(filter={"workflow_id": old_workflow_id})
        print(f"Removing associated timelines, if any...")
        MONGO_CLIENT.ds_timelines.timeline.delete_many(filter={"workflow_id": old_workflow_id})
        print(f"Removing associated provenances and histories, if any...")
        MONGO_CLIENT.ds_provenance.history.delete_many(filter={"workflow_id": old_workflow_id})
        MONGO_CLIENT.ds_provenance.provenance.delete_many(filter={"workflow_id": old_workflow_id})
        print(f"All results associated with workflow ID {old_workflow_id} have been purged.")
        # Cleaning up model_graph database
        MONGO_CLIENT.model_graph.node.delete_many({})
        MONGO_CLIENT.model_graph.edge.delete_many({})
        MONGO_CLIENT.model_graph.model_paths.delete_many({})
        MONGO_CLIENT.model_graph.timelines.delete_many({})
    else:
        if old_workflow_id is not None:
            print(f"All results associated with workflow ID {old_workflow_id} were preserved.")

    print(f"Ready to run!")


def compute_metrics(workflow_id):
    global global_list, completion_list, MONGO_CLIENT

    workflow = MONGO_CLIENT["ds_config"]["workflows"].find_one({"_id": workflow_id})
    timelines = MONGO_CLIENT["model_graph"]["timelines"].find({"no_of_models": len(workflow["model_settings"].values())})
    start_time = workflow["start_time"]
    end_time = workflow["end_time"]
    timelines_order = list()
    for timeline in timelines:
        print(str(round((timeline["insert_time"] - start_time).total_seconds(), 2)))
        timelines_order.append(round((timeline["insert_time"] - start_time).total_seconds(), 2))
    # End of loop

    print("total_time: " + str(round((end_time - start_time).total_seconds(), 2)))
    completion_list.append(str(round((end_time - start_time).total_seconds(), 2)))
    global_list.append(timelines_order)


def main():
    global DS_CONFIG, MONGO_CLIENT

    MONGO_CLIENT = ds_utils.connect_to_mongo()
    iter = 30
    K = 20
    probability = 0
    penalty = 0.5
    max_model_path = 5
    diversity = 7
    causal_depth, causal_width, causal_edges = 3, 3, 1
    for i in range(0, iter):
        generate_workflow(causal_depth, causal_width, causal_edges)
        reset_mongo()
        _connect_to_mongo()
        model_list = [model_config["name"] for model_config in DS_CONFIG["model_settings"].values()]
        for model in model_list:
            simulate_model(model)

        workflow_id = DS_CONFIG["workflow_id"]
        model_graph = ModelGraph(MONGO_CLIENT, workflow_id)
        model_graph.generate_model_graph()

        timeline = Timelines(MONGO_CLIENT, workflow_id)
        timeline.generate_timelines_via_A_star(K, probability, penalty, max_model_path, diversity)
        compute_metrics(workflow_id)
        MONGO_CLIENT["model_graph"]["model_paths"].remove({})
        MONGO_CLIENT["model_graph"]["timelines"].remove({})
        MONGO_CLIENT["model_graph"]["causal_pairs"].remove({})
    # End of loop

    b = numpy.zeros([len(global_list), len(max(global_list, key=lambda x: len(x)))])
    for i, j in enumerate(global_list):
        b[i][0:len(j)] = j
    b = numpy.transpose(numpy.array(b))
    print(b)
    p = pd.DataFrame(numpy.array(b), columns=list(range(0, iter)))
    data = numpy.array(b)
    data[data == 0] = numpy.nan
    print(numpy.nanmean(data, axis=1))
    print(numpy.nanvar(data, axis=1))
    print()
    writer = pd.ExcelWriter("metric.xlsx", engine='xlsxwriter')
    p.to_excel(writer, sheet_name="exp-2")
    writer.save()
    writer.close()
    ds_utils.disconnect_from_mongo()


if __name__ == '__main__':
    main()
