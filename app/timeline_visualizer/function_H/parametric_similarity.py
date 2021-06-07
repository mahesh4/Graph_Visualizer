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

