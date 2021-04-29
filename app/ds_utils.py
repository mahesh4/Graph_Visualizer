import enum
import os
import pathlib
import subprocess
import sys
import time

import pymongo
import sshtunnel
import yaml

USE_SSH: bool = True
MONGO_SERVER: sshtunnel.SSHTunnelForwarder = None
MONGO_CLIENT: pymongo.MongoClient = None
LOCAL_KEYPATH: str = None
LOCAL_USER: str = None

MODEL_MAP: dict = None

BASE_DIR = pathlib.Path(__file__).parent.absolute()


class ExitCode(enum.IntEnum):
    Undefined: int = -1
    Success: int = 0
    Failure: int = 1
    Waiting: int = 2
    SubactorComplete: int = 5
    SimulationComplete: int = 6


def do_shell(cmd: str):
    process = subprocess.Popen(cmd, shell=True)  # blocking
    _ = process.communicate()[0]
    return process.returncode


def fire_shell(cmd: str):
    _ = subprocess.Popen(cmd, shell=True)  # non-blocking
    return


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


def connect_to_mongo():
    global MONGO_SERVER, MONGO_CLIENT, LOCAL_KEYPATH, LOCAL_USER

    if MONGO_CLIENT is not None:
        return MONGO_CLIENT

    # load connection settings
    if USE_SSH:
        try:
            LOCAL_KEYPATH = None
            LOCAL_USER = None
            settings_path = "settings.yaml"
            with open(settings_path, "r") as infile:
                settings = yaml.safe_load(infile.read())
            if settings["cluster_type"] == "AWS":
                LOCAL_KEYPATH = str(pathlib.Path(settings["aws_settings"]["KeyPath"]).expanduser().resolve())
                LOCAL_USER = settings["aws_settings"]["ImageUser"]
            elif settings["cluster_type"] == "OpenStack":
                LOCAL_KEYPATH = str(
                    pathlib.Path(settings["openstack_settings"]["local_keypair"]).expanduser().resolve())
                LOCAL_USER = settings["openstack_settings"]["image_user"]
        except Exception as e:
            print("DataStorm tried to connect via SSH for debugging, but couldn't find a valid key path. "
                  "Check DataStorm/wizard/settings.yaml")
            sys.exit(ExitCode.Failure)

        MONGO_SERVER = sshtunnel.SSHTunnelForwarder(
            ("ds_core_mongo", 22),
            ssh_username=LOCAL_USER,
            ssh_pkey=LOCAL_KEYPATH,
            remote_bind_address=('localhost', 27017)
        )
        MONGO_SERVER.start()
        MONGO_CLIENT = pymongo.MongoClient(host='localhost', port=MONGO_SERVER.local_bind_port)
    else:
        MONGO_CLIENT = pymongo.MongoClient(host="ds_core_mongo")

    return MONGO_CLIENT


def disconnect_from_mongo():
    global MONGO_CLIENT, MONGO_SERVER

    if MONGO_CLIENT is None:
        return
    MONGO_CLIENT.close()
    if USE_SSH:
        MONGO_SERVER.stop()
        MONGO_SERVER = None
    MONGO_CLIENT = None


def log(logfile: str = "log.txt", logstr: str = ""):
    if "/" not in str(logfile):
        logfile = expand_path(logfile)
    logfile = pathlib.Path(logfile)
    try:
        with open(logfile, "a") as log_handle:
            log_handle.write(logstr + "\n")
        print(logstr)
    except FileNotFoundError:
        print(f"[ERROR] Could not write to log at path {str(logfile)}")


def get_model():
    with open(expand_path("model.txt"), "r") as infile:
        model_name = infile.readline().replace("\n", "")
    return model_name


def set_model(model_name):
    with open(expand_path("model.txt"), "w") as outfile:
        outfile.write(model_name)


def expand_path(filename):
    return BASE_DIR / filename


def set_time_log(timepath):
    with open(expand_path("ds_utils.time.config"), "w") as outfile:
        outfile.write(str(timepath))


def get_time_log():
    if os.path.isfile(expand_path("ds_utils.time.config")):
        with open(expand_path("ds_utils.time.config"), "r") as infile:
            time_path = infile.readline()
    else:
        time_path = expand_path("../results/timings.csv")
    return time_path


def log_time(payload, elapsed):
    timepath = get_time_log()

    try:
        outstring = f'{payload},{elapsed},{time.time()}\n'
        with open(timepath, "a") as time_log:
            time_log.write(outstring)
    except FileNotFoundError:
        print("No time log found.")
