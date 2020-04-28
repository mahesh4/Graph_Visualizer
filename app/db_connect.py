import pymongo
from sshtunnel import SSHTunnelForwarder

MONGO_IP = "129.114.27.85"
GRAPH_IP = "129.114.26.12"
MONGO_KEYFILE = "dsworker_rsa"


class DBConnect:
    def __init__(self):
        # SSH / Mongo Configuration #
        self.MONGO_SERVER = SSHTunnelForwarder(
            (MONGO_IP, 22),
            ssh_username="cc",
            ssh_pkey="~/.ssh/{0}".format(MONGO_KEYFILE),
            remote_bind_address=('localhost', 27017)
        )

        # SSH / Graph Configuration #
        self.GRAPH_SERVER = SSHTunnelForwarder(
            (GRAPH_IP, 22),
            ssh_username="cc",
            ssh_pkey="~/.ssh/{0}".format(MONGO_KEYFILE),
            remote_bind_address=('localhost', 27017)
        )

        # open the SSH tunnel to the mongo server
        self.MONGO_SERVER.start()

        # open the SSH tunnel to the graph server
        self.GRAPH_SERVER.start()

        self.MONGO_CLIENT = pymongo.MongoClient('localhost', self.MONGO_SERVER.local_bind_port)
        self.GRAPH_CLIENT = pymongo.MongoClient('localhost', self.GRAPH_SERVER.local_bind_port)
        print('opened all connections')

    def __del__(self):
        # close graph connection
        self.GRAPH_CLIENT.close()
        # close the SSH tunnel to the graph server
        self.GRAPH_SERVER.close()
        # close mongo connection
        self.MONGO_CLIENT.close()
        # close the SSH tunnel to the mongo server
        self.MONGO_SERVER.close()

        print('closed all the connections')

