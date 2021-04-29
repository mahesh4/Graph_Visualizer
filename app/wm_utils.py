import app.ds_utils
import collections
from scipy import stats as scipy_stats

RULE_CONSTRAINT_LIST: dict = dict()


def compute_rule_constraint_list(config):
    global RULE_CONSTRAINT_LIST

    for model_id, model_config in config["model_settings"].items():
        for variable_id, variable_config in model_config["sampled_variables"].items():
            RULE_CONSTRAINT_LIST[variable_config["name"]] = float(variable_config["bin_size"])


def is_match(param_name, param_val_list, config):
    global RULE_CONSTRAINT_LIST

    if not bool(RULE_CONSTRAINT_LIST):
        compute_rule_constraint_list(config)

    match_counter = 0
    no_of_val = len(param_val_list)
    compatibility = True
    if no_of_val == 1:
        return compatibility, 1

    for v1_idx in range(no_of_val):
        for v2_idx in range(v1_idx + 1, no_of_val):
            if round(abs(param_val_list[v1_idx] - param_val_list[v2_idx]), 1) <= RULE_CONSTRAINT_LIST[param_name]:
                match_counter = match_counter + 1
        # End of loop
    # End of loop
    total_counter = (no_of_val * (no_of_val - 1)) / 2
    if match_counter == 0:
        compatibility = False
    return compatibility, match_counter / total_counter


def union(parameter_dict, config):
    metric_list = []
    compatibility = False
    for param_name, param_val_list in parameter_dict.items():
        variable_compatibility, metric = is_match(param_name, param_val_list, config)
        metric_list.append(metric)
        if variable_compatibility:
            compatibility = True
    # End of loop
    if compatibility:
        return True, scipy_stats.hmean(metric_list)
    else:
        return False, 0


def conjunction(parameter_dict, config):
    metric_list = []
    compatibility = True
    for param_name, param_val_list in parameter_dict.items():
        variable_compatibility, metric = is_match(param_name, param_val_list, config)
        metric_list.append(metric)
        if not variable_compatibility:
            compatibility = False
            break
    # End of loop
    if compatibility:
        return True, scipy_stats.hmean(metric_list)
    else:
        return False, 0


def connnect_to_mongo(mongo_client):
    global DSIR_DB, JOB_DB, DS_CONFIG

    DSIR_DB = mongo_client["ds_results"]["dsir"]
    JOB_DB = mongo_client["ds_results"]["jobs"]
    DS_CONFIG = mongo_client["ds_config"]["collection"].find_one({})


def compute_parametric_similarity(candidate_sets, model, mongo_client):
    """
    Returns parametric similarity score of the windowing_set
        Parameters:
            candidate_sets (list):
            model
        Returns:
            candidate_sets (list): A list of candidate windowing_set with their corresponding parametric similarity scores computed
    """
    global DSIR_DB, JOB_DB, DS_CONFIG

    connnect_to_mongo(mongo_client)

    if DS_CONFIG["compatibility_settings"]["parametric_mode"] == "conjunction":
        parametric_compatibility_function = conjunction
    elif DS_CONFIG["compatibility_settings"]["parametric_mode"] == "union":
        parametric_compatibility_function = union
    else:
        raise Exception("Illegal parametric compatibility")

    for candidate_record_set in candidate_sets:
        record_set = candidate_record_set[0]
        compat_dict = candidate_record_set[1]
        parameter_dict = collections.defaultdict(list)
        dsir_id_list = record_set
        prev_sampled_values = dict()
        for dsir_id in dsir_id_list:
            dsir = DSIR_DB.find_one({"_id": dsir_id})
            if dsir["created_by"] == "PostSynchronizationManager":
                parent_dsir_list = dsir["parents"]
                jobs_list = list(JOB_DB.find({"output_dsir": {"$in": parent_dsir_list}}))
            else:
                jobs_list = list([JOB_DB.find_one({"output_dsir": dsir_id})])

            for job in jobs_list:
                for variable, value in job["variables"].items():
                    parameter_dict[variable].append(value)
                # End of loop

            if dsir["metadata"]["model_type"] == model:
                for job in jobs_list:
                    for variable, value in job["variables"].items():
                        prev_sampled_values[variable] = value

        # End of loop

        # computing parametric similarity score
        compatibility, score = parametric_compatibility_function(parameter_dict, DS_CONFIG)
        compat_dict["parametric"] = score
        candidate_record_set.append(prev_sampled_values)

    return candidate_sets