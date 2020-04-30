import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as ntx
import math


class VisualizeGraph:
    def __init__(self):
        self.MONGO_CLIENT = None
        self.GRAPH_CLIENT = None
        self.MODEL_GRAPH = None
        self.MODEL = None
        self.OPTIMAL_PATH = list()

    def connect_db(self, db_connect):
        self.MONGO_CLIENT = db_connect.MONGO_CLIENT
        self.GRAPH_CLIENT = db_connect.GRAPH_CLIENT

    def disconnect_db(self):
        self.MONGO_CLIENT.close()
        self.GRAPH_CLIENT.close()

    def draw_graph_hnx(self):
        flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
        edge_list = flow_graph_constraint_database['edge']

        scenes = {}
        for edge in edge_list:
            for i in range(len(edge['edge_names'])):
                source = edge['source']
                destination = edge['destination']
                source_label = source[i]['node_type'] + '-' + source[i]['type'] + '-' + \
                               source[i][source[i]['type']][-3:]
                destination_label = destination['node_type'] + '-' + destination['type'] + '-' + \
                                    destination[destination['type']][-3:]
                scenes[edge['edge_names'][i]] = [source_label, destination_label]
        H = hnx.Hypergraph(scenes)
        hnx.draw(H)
        plt.waitforbuttonpress()

    def draw_graph_optimal_path_ntx(self):
        flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
        edge_list_collection = flow_graph_constraint_database['edge']
        edge_list = edge_list_collection.find({})
        N = ntx.Graph()
        for edge in edge_list:
            for i in range(len(edge['source'])):
                source = edge['source']
                destination = edge['destination']
                edge_name = edge['edge_names'][i]
                source_label = source[i]['node_type'] + '-' + source[i]['type'] + '-' + \
                               str(source[i]['node_id'])[-3:]
                destination_label = destination['node_type'] + '-' + destination['type'] + '-' + \
                                    str(destination['node_id'])[-3:]
                if self.MODEL.getVarByName(edge_name).x == 1:
                    N.add_edge(source_label, destination_label, name=edge_name, color='r')
                else:
                    N.add_edge(source_label, destination_label, name=edge_name, color='g')

        pos = ntx.spring_layout(N)
        colors = [N[u][v]['color'] for u, v in N.edges()]
        ntx.draw(N, pos, edges=N.edges(), edge_color=colors, with_labels=True)
        edge_labels = dict([((u, v,), d['name']) for u, v, d in N.edges(data=True)])
        ntx.draw_networkx_edge_labels(N, pos, edge_labels)
        plt.show()

    def draw_graph_ntx(self):
        flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
        edge_list_collection = flow_graph_constraint_database['edge']
        edge_list = edge_list_collection.find({})
        N = ntx.Graph()
        for edge in edge_list:
            for i in range(len(edge['source'])):
                source = edge['source']
                destination = edge['destination']
                edge_name = edge['edge_names'][i]
                source_label = source[i]['node_type'] + '-' + source[i]['type'] + '-' + \
                               str(source[i]['node_id'])[-3:]
                destination_label = destination['node_type'] + '-' + destination['type'] + '-' + \
                                    str(destination['node_id'])[-3:]
                N.add_edge(source_label, destination_label, name=edge_name, color='g')

        pos = ntx.spring_layout(N, k=3/math.sqrt(N.order()))
        # pos = ntx.random_layout(N)
        colors = [N[u][v]['color'] for u, v in N.edges()]
        ntx.draw(N, pos, edges=N.edges(), edge_color=colors, with_labels=True)
        edge_labels = dict([((u, v,), d['name']) for u, v, d in N.edges(data=True)])
        ntx.draw_networkx_edge_labels(N, pos, edge_labels)
        plt.show()

    def set_model(self, model):
        self.MODEL = model
