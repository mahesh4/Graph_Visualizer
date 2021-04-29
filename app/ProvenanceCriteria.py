import pathlib
import sys
import pymongo
from bson.objectid import ObjectId
from scipy import stats as scipy_stats

MODULES_PATH = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(MODULES_PATH))  # this allows for script-relative library loading
from app import ds_utils

MONGO_CLIENT: pymongo.MongoClient = None
PROVENANCE_DB: pymongo.collection.Collection = None
DSIR_DB: pymongo.collection.Collection = None
MODEL_INFO = None
CONFIG = dict()


def connect_to_mongo():
    """
    Function to connect to MongoDB
    """
    global CONFIG, MONGO_CLIENT, PROVENANCE_DB, HISTORY_DB, DSIR_DB, MODEL_INFO

    MONGO_CLIENT = ds_utils.connect_to_mongo()
    CONFIG = MONGO_CLIENT["ds_config"]["collection"].find_one()
    PROVENANCE_DB = MONGO_CLIENT["ds_provenance"]["provenance"]
    HISTORY_DB = MONGO_CLIENT["ds_provenance"]["history"]
    DSIR_DB = MONGO_CLIENT["ds_results"]["dsir"]
    # model = ds_utils.get_model()
    # TODO:Hardcoded here
    model = "flood"
    MODEL_INFO = ds_utils.access_model_by_name(CONFIG, model)


def disconnect_to_mongo():
    ds_utils.disconnect_from_mongo()


def find_provenance_score(windowing_set, temporal_context):
    """
    Function to generate the provenance and compute its provenance_score for each windowing_set
        Parameters:
            windowing_set (list): A list of DSIR ids
            temporal_context (dict): Temporal Context of the current execution
        Returns:
            provenance_score (float):
    """
    history = create_history_from_windowing_set(windowing_set, temporal_context)
    provenance_score = compute_provenance_criteria(history)
    return provenance_score


def find_provenance_similarity(dsir_1, dsir_2, temporal_context):
    """
    Function to compute provenance similarity. Its computed as follows:
        1) combine the provenance of dsir_1 and dsir_2 to generate a history
        2) compute the overlaps and coverage in the history

        Parameters:
            dsir_1 (ObjectId):
            dsir_2 (ObjectId):
            temporal_context (dict):
            model (str):
        Returns:
            provenance_similarity (float):
    """
    global PROVENANCE_DB, DSIR_DB, CONFIG
    connect_to_mongo()

    provenance_1 = PROVENANCE_DB.find_one({"dsir_id": dsir_1})
    provenance_2 = PROVENANCE_DB.find_one({"dsir_id": dsir_2})
    history = dict()
    for provenance in [provenance_1, provenance_2]:
        for model, channel in provenance["provenance"].items():
            if model not in history:
                history[model] = dict()
            for endtime, instance_list in channel.items():
                if int(endtime) != temporal_context["end"]:
                    if endtime in history[model]:
                        history[model][endtime].update(instance_list)
                    else:
                        history[model][endtime] = set(instance_list)

    provenance_similarity = compute_provenance_criteria(history)
    return provenance_similarity


def create_history_from_windowing_set(windowing_set, temporal_context):
    """
    Returns the History of the windowing_set. The History of the windowing_set is defined as the Union of the Provenance of the individual instances in the
    windowing_set
        Parameters:
            windowing_set (list): A list of DSIR
            temporal_context (dict): temporal context begin, and end time of the current model in execution
        Returns:
            history (list): History of the windowing_set
    """
    global CONFIG, PROVENANCE_DB, DSIR_DB

    connect_to_mongo()
    history = dict()
    history_window = CONFIG["compatibility_settings"]["provenance_size"]
    threshold = temporal_context["begin"] - history_window
    for dsir_id in windowing_set:
        provenance = PROVENANCE_DB.find_one({"dsir_id": dsir_id})
        if provenance is not None:
            for model, channel in provenance["provenance"].items():
                if model not in history:
                    history[model] = dict()
                for endtime, instance_list in channel.items():
                    if int(endtime) >= threshold:
                        if endtime in history[model]:
                            history[model][endtime].update(instance_list)
                        else:
                            history[model][endtime] = set(instance_list)

    for model, channel in history.items():
        for endtime, instance_set in channel.items():
            channel[endtime] = list(instance_set)

    return history


def create_provenance_from_histories(history_id_list):
    """
    Returns the new provenance which is the union of the histories in the history_id_list
        Parameters:
            history_id_list (list): A list of history_ids
        Returns:
            provenance (dict):
    """
    global PROVENANCE_DB, HISTORY_DB, DSIR_DB

    connect_to_mongo()
    provenance = dict()
    for history_id in history_id_list:
        history = HISTORY_DB.find_one({"_id": history_id})
        for model, channel in history["history"].items():
            if model not in provenance:
                provenance[model] = dict()
            for endtime, instance_list in channel.items():
                if endtime in provenance[model]:
                    provenance[model][endtime].update(instance_list)
                else:
                    provenance[model][endtime] = set(instance_list)

    for model, channel in provenance.items():
        for endtime in channel:
            channel[endtime] = list(channel[endtime])

    return provenance


def compute_provenance_criteria(history):
    """
    Function to compute the overlaps in the provenance
        Parameters:
            history (dict):
        Returns:

    """
    overlaps = 0
    gaps = 0
    # Aggregating overlaps per model
    for model, instance_dict in history.items():
        model_info = ds_utils.access_model_by_name(CONFIG, model)
        output_window_size = model_info["temporal"]["output_window"]
        endtime_list = sorted(list(instance_dict.keys()))
        size = len(instance_dict)
        for index, endtime in enumerate(endtime_list):
            overlaps += (len(instance_dict[endtime]) - 1)
            endtime = int(endtime)
            for i in range(index + 1, size):
                # Checking for overlaps
                next_window_start_time = int(endtime_list[i]) - output_window_size
                if next_window_start_time < endtime:
                    overlaps += (abs(endtime - next_window_start_time) / output_window_size) * len(instance_dict[endtime_list[i]])
                else:
                    gaps += (next_window_start_time - endtime) / output_window_size
                    break
        # End of loop
    # End of loop

    return scipy_stats.hmean([1 / (overlaps + 1), 1 / (gaps + 1)])


def add_target_dsir_to_provenance(provenance, target_dsir):
    """
    Function to add the target_dsir to the provenance
        Parameters:
            provenance (dict):
            target_dsir (dict):
        Returns:
            provenance (dict):
    """
    model_type = target_dsir["metadata"]["model_type"]
    endtime = str(int(target_dsir["metadata"]["temporal"]["end"]))
    if model_type not in provenance:
        provenance[model_type] = dict()

    provenance[model_type][endtime] = [target_dsir["_id"]]
    return provenance
