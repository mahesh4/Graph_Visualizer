import collections
import random
import bson
import pymongo
from bson.objectid import ObjectId
from collections import defaultdict, OrderedDict
from itertools import chain, combinations
import itertools
import math
import copy
import numpy
from scipy import stats as scipy_stats
from app import utils
from app.db_connect import DBConnect

DS_CONFIG = None
MONGO_CLIENT = None
KEPLER_DB: pymongo.collection.Collection = None
DSIR_DB: dict = dict()
JOBS_DB: pymongo.collection.Collection = None


def _connect_to_mongo():
    global DS_CONFIG, DSIR_DB, MONGO_CLIENT, KEPLER_DB, JOBS_DB

    db_connect = DBConnect()
    db_connect.connect_db()
    MONGO_CLIENT = db_connect.get_connection()
    DS_CONFIG = MONGO_CLIENT["ds_config"]["collection"].find_one({})
    KEPLER_DB = MONGO_CLIENT["ds_state"]["kepler"]
    DSIR_DB = MONGO_CLIENT["ds_results"]["dsir"]
    JOBS_DB = MONGO_CLIENT["ds_results"]["jobs"]


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
    increase_temporal_context()
    return


def _perform_postsynchronization(model, model_info):
    global DSIR_DB, KEPLER_DB

    aggregation_strategy = model_info["psm_settings"]["psm_strategy"]
    state = KEPLER_DB.find_one({"model_type": model})
    dsir_list = state["result_pool"]["to_sync"]
    to_output = list()
    if aggregation_strategy == 'none':
        for dsir_id in dsir_list:
            target_dsir = DSIR_DB.find_one({"_id": dsir_id})
            create_provenance(target_dsir)
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
            create_provenance(dsir)
        # End of loop

        dsir_pair_list = [[dsir_list[i], dsir_list[j]] for i in range(no_of_dsirs) for j in range(no_of_dsirs) if i != j]
        similarity_graph = [[1] * no_of_dsirs for _ in range(no_of_dsirs)]
        parametric_compatibility_function = None
        if parametric_similarity:
            if DS_CONFIG["compatibility_settings"]["parametric_mode"] == "conjunction":
                parametric_compatibility_function = ds_compatibility.conjunction
            elif DS_CONFIG["compatibility_settings"]["parametric_mode"] == "union":
                parametric_compatibility_function = ds_compatibility.union

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

        nb_clusters, cluster_labels = perform_clustering(similarity_graph, no_of_candidates)

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

            new_dsir = _create_dsir(model, start, end, shift_size, output_window, "PostSynchronizationManager")
            new_dsir["parents"] = dsir_aggr_list
            DSIR_DB.save(new_dsir)
            to_output.append(new_dsir["_id"])
            # Adding the children to each dsir in dsir_aggr_list
            for dsir_id in dsir_aggr_list:
                dsir = DSIR_DB.find_one({"_id": dsir_id})
                dsir["children"] = new_dsir["_id"]
                DSIR_DB.save(dsir)
            # End of loop

        # End of loop
    else:
        raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")
    # Updating the state
    state["result_pool"]["to_output"] = to_output
    KEPLER_DB.save(state)
    return


def _sample_values(model, start, end, model_info):
    global KEPLER_DB, JOBS_DB, DS_CONFIG, DSIR_DB

    state = KEPLER_DB.find_one({"model_type": model})
    candidate_list = state["result_pool"]["to_sample"]
    sampling_strategy = DS_CONFIG["compatibility_settings"]["compatibility_strategy"]
    model_vars = list(model_info["sampled_variables"].values())
    to_sync = list()
    job_count = model_info["sm_settings"]["sm_fanout"]
    for dsir_list in candidate_list:
        new_job = _create_job(model)
        new_job["input_dsir"] = dsir_list
        for i in range(job_count):
            if sampling_strategy == "independent":
                for each_var in model_vars:
                    var_name = each_var["name"]
                    low = each_var["lower_bound"]
                    high = each_var["upper_bound"]
                    new_job["variables"][var_name] = round(random.uniform(low, high), 2)
                # End of loop
            # TODO: Need to develop provenance_sampling
            # creating new_dsirs for job
            new_dsir = _create_dsir(model, start, end, shift_size, output_window)
            new_job["output_dsir"] = new_dsir["_id"]
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
    KEPLER_DB.save(state)
    return


def _perform_alignment(model, start, end, model_info):
    global KEPLER_DB, DS_CONFIG, DSIR_DB

    state = KEPLER_DB.find_one({"model_type": model})
    output_window = model_info["temporal"]["output_window"]
    for candidate_set in state["result_pool"]["to_align"]:
        model_instances_dict = _group_instances_by_model(candidate_set)
        aligned_dsirs = list()
        for model, instance_list in model_instances_dict.items():
            new_dsir = _create_dsir(model, start, end, shift_size, output_window)
            new_dsir["parents"] = instance_list
            aligned_dsirs.append(new_dsir["_id"])
            DSIR_DB.save(new_dsir)
        # End of loop
        state["result_pool"]["to_sample"].append(aligned_dsirs)
    # End of loop
    KEPLER_DB.save(state)


def _power_set(input_iterable: collections.Iterable):
    s = list(input_iterable)
    return list(itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1)))


def _check_compatibility_temporal(candidate_sets, begin, model_info):
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
    new_dsir["metadata"] = {"temporal": {"begin": start, "end": end, "window_size": output_window, "shift_size": shift_size, "timestamp_list": list(),
                                         "model_type": model}}
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
    for record_id in all_results:
        record = DSIR_DB.find_one({"_id": record_id})
        # removing data which are not compatible for current window
        # TODO: Shift added here
        if record["metadata"]["model_type"] == model and record["metadata"]["temporal"]["end"] == begin + shift_size:
            self.append(record_id)
        elif begin <= record["metadata"]["temporal"]["end"] or record["metadata"]["temporal"]["end"] >= begin + input_window:
            other.append(record_id)
        else:
            print("not compatible record for current execution: {0}".format(record_id))
    return self, other


def _generate_windowing_sets(model, begin, output_end, model_info):
    global DS_CONFIG, KEPLER_DB

    state = KEPLER_DB.find_one({"model_type": model})
    simulation_end = DS_CONFIG["simulation_context"]["temporal"]["end"]
    is_seed = DS_CONFIG["simulation_context"]["temporal"]["begin"] == begin and "upstream_models" not in model_info
    # TODO:Need to write logic for is_seed

    # splitting results
    self_results, other_results = _split_results(state["result_pool"]["to_window"], model, begin, model_info)

    state_candidates = _check_compatibility_startstate(self_results, model_info, begin)

    if state_candidates is not None and len(state_candidates) == 0:
        raise Exception("[ERROR] Stateful model expected previous window data, but didn't receive it!")

    raw_powerset = _power_set(other_results)
    input_powerset = []
    for entry in raw_powerset:
        metadata = dict()
        input_powerset.append([entry, metadata])
    # End of loop

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

    # find parametric similarity score for the candidate set of records
    candidate_sets = _compute_parametric_similarity(candidate_sets)

    # find provenance score for the candidate sets of records
    temporal_context = {"begin": start, "end": min(start + input_window, simulation_end)}
    candidate_sets = generate_provenance_score(candidate_sets, temporal_context)

    # remove the DSARs of MY_MODEL if not Stateful
    if not model_info["stateful"] and state_candidates is not None:
        for each_candidate in candidate_sets:
            record_set = each_candidate[0]
            new_record_set = [record_id for record_id in record_set if DSAR_DB.find_one({"_id": record_id})["metadata"]["model_type"] != MY_MODEL]
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

    # sort the candidates by their weighted compatibility
    candidate_sets.sort(key=lambda each: each[3], reverse=True)
    new_state = None

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
        history_list.append({"history_id": history_id.inserted_id, "upstream_dsar": candidate_sets[0][0]})
    else:
        # send multiple candidates to the aligner
        K = int(WINDOWING_POLICY["wm_fanout"])
        interest_parameters = {param_dict["parameter"]: param_dict["strategy"] for param_dict in DS_CONFIG["exploration"].values()}
        candidate_sets = Skyline.find_top_k_candidates(candidate_sets, K, interest_parameters, MODEL_INFO, MONGO_CLIENT)
        for each_candidate in candidate_sets:
            dsir_list = each_candidate[0]
            state["result_pool"]["to_align"].append(dsir_list)
            parameter_list.append(each_candidate[1])
            history = ProvenanceCriteria.create_history_from_windowing_set(dsir_list, temporal_context)
            # Writing history to DB
            history_id = HISTORY_DB.insert_one({"history": history, "workflow_id": DS_CONFIG["workflow_id"]})
            history_list.append({"history_id": history_id.inserted_id, "upstream_dsar": each_candidate[0]})
        # End of loop

    print("All candidates generated.")
    DS_CONFIG["sampled_parameters"] = parameter_list
    MONGO_CLIENT["ds_config"].collection.save(DS_CONFIG)
    # go to alignment manager
    # If WM receives more than one model, candidates from different models will be stored in the same candidate list.
    new_state["subactor_state"] = "AlignmentManager"
    new_state["history"] = history_list
    KEPLER_DB.save(state)
    return


def execute_model(model, begin, output_end, model_info):
    _generate_windowing_sets(model, begin, output_end, model_info)
    _perform_alignment(model, begin, output_end, model_info)
    _sample_values(model, begin, output_end, model_info)
    _perform_postsynchronization(model, model_info)
    _perform_output(model, model_info)
    return


def simulate_model(model):
    global DS_CONFIG

    model_info = utils.access_model_by_name(DS_CONFIG, model)
    simulation_begin = DS_CONFIG["simulation_context"]["temporal"]["begin"]
    simulation_end = DS_CONFIG["simulation_context"]["temporal"]["end"]
    begin = simulation_begin
    output_window = model_info["temporal"]["output_window"]
    shift_size = model_info["temporal"]["shift"]
    while begin < simulation_end:
        output_end = begin + output_window
        execute_model(model, begin, output_end, model_info)
        # TODO: Shift subtracted here
        begin = begin + output_window - shift_size


def main():
    global DS_CONFIG

    _connect_to_mongo()
    # TODO: Hardcoded Here
    model_list = ["hurricane", "flood"]
    for model in model_list:
        simulate_model(model)


if __name__ == '__main__':
    main()
