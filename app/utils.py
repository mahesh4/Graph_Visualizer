MODEL_MAP = None


def model_name_to_index(ds_config, model_name):
    global MODEL_MAP

    if MODEL_MAP is None:
        MODEL_MAP = dict()
        for model_id in ds_config["model_settings"].keys():
            this_name = ds_config["model_settings"][model_id]["name"]
            MODEL_MAP[this_name] = model_id

    return MODEL_MAP[model_name]


def access_model_by_name(ds_config, model_name):
    return ds_config["model_settings"][model_name_to_index(ds_config, model_name)]

