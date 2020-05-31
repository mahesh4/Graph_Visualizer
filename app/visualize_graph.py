import hypernetx as hnx
import matplotlib.pyplot as plt
import networkx as ntx
import math
from app.db_connect import DBConnect


class VisualizeGraph(DBConnect):
    def __init__(self):
        DBConnect.__init__(self)

    def draw_graph_hnx(self):
        try:
            self.connect_db()
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
        except Exception as e:
            raise e
        finally:
            self.disconnect_db()

    def draw_graph_optimal_path_ntx(self, model):
        try:
            self.connect_db()
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
                    if model.getVarByName(edge_name).x == 1:
                        N.add_edge(source_label, destination_label, name=edge_name, color='r')
                    else:
                        N.add_edge(source_label, destination_label, name=edge_name, color='g')

            pos = ntx.spring_layout(N)
            colors = [N[u][v]['color'] for u, v in N.edges()]
            ntx.draw(N, pos, edges=N.edges(), edge_color=colors, with_labels=True)
            edge_labels = dict([((u, v,), d['name']) for u, v, d in N.edges(data=True)])
            ntx.draw_networkx_edge_labels(N, pos, edge_labels)
            plt.show()
        except Exception as e:
            raise e
        finally:
            self.disconnect_db()

    def draw_graph_ntx(self):
        try:
            self.connect_db()
            flow_graph_constraint_database = self.GRAPH_CLIENT['model_graph']
            edge_list_collection = flow_graph_constraint_database['edge']
            edge_list = edge_list_collection.find({})
            N = ntx.Graph()
            for edge in edge_list:
                source = edge['source']
                destination = edge['destination']
                edge_name = edge['edge_name']
                source_label = str(source)[-3:]
                destination_label = str(destination)[-3:]
                N.add_edge(source_label, destination_label, name=edge_name, color='g')

            pos = ntx.spring_layout(N, k=3/math.sqrt(N.order()))
            # pos = ntx.random_layout(N)
            colors = [N[u][v]['color'] for u, v in N.edges()]
            ntx.draw(N, pos, edges=N.edges(), edge_color=colors, with_labels=True)
            edge_labels = dict([((u, v,), d['name']) for u, v, d in N.edges(data=True)])
            ntx.draw_networkx_edge_labels(N, pos, edge_labels)
            plt.show()
        except Exception as e:
            raise e
        finally:
            self.disconnect_db()