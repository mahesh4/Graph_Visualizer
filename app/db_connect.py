import pymongo
import enum
import pathlib
import sys
import sshtunnel
import yaml

LOCAL_KEYPATH: str = None
LOCAL_USER: str = None
USE_SSH = False


BASE_DIR = pathlib.Path(__file__).parent.absolute()


class ExitCode(enum.IntEnum):
    Undefined: int = -1
    Success: int = 0
    Failure: int = 1
    Waiting: int = 2
    SubactorComplete: int = 5
    SimulationComplete: int = 6


class DBConnect:

    def __init__(self):
        global LOCAL_KEYPATH, LOCAL_USER
        if USE_SSH:
            try:
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

            # SSH / Mongo Configuration #
            self.MONGO_SERVER = sshtunnel.SSHTunnelForwarder(
                ("ds_core_mongo", 22),
                ssh_username=LOCAL_USER,
                ssh_pkey=LOCAL_KEYPATH,
                remote_bind_address=('localhost', 27017)
            )
            self.MONGO_CLIENT: pymongo.MongoClient = None

    def connect_db(self):
        try:
            if USE_SSH:
                # open the SSH tunnel to the mongo server
                self.MONGO_SERVER.start()
                self.MONGO_CLIENT = pymongo.MongoClient(host='localhost', port=self.MONGO_SERVER.local_bind_port)
            else:
                self.MONGO_CLIENT = pymongo.MongoClient("localhost", 27017)
            print('connected')
        except Exception as e:
            print('cannot connect')
            raise e

    def get_connection(self):
        return self.MONGO_CLIENT

    def disconnect_db(self):
        try:
            # close mongo connection
            self.MONGO_CLIENT.close()
            if USE_SSH:
                # close the SSH tunnel to the mongo server
                self.MONGO_SERVER.close()
            self.MONGO_CLIENT = None
            print("disconnected")
        except Exception as e:
            print('cannot disconnect')
            raise e
