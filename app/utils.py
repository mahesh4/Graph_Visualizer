
def convert_id_to_model_type(config):
    id_to_model_type = dict()
    for model_id, model_config in config["model_settings"].items():
        id_to_model_type[model_config["name"]] = model_id
    return id_to_model_type


def convert_model_type_to_id(config):
    model_type_to_id = dict()
    for model_id, model_config in config["model_settings"].items():
        model_type_to_id[model_id] = model_config["name"]
    return model_type_to_id
