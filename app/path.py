from app.db_connect import DBConnect
from bson.objectid import ObjectId


class PathFinder(DBConnect):
    def __init__(self):
        DBConnect.__init__(self)
        self.forward_timelines = list()
        self.backward_timelines = list()

    def get_timeline_tmp(self, node_id, path, node_collection, edge_collection):
        """Function to return the forward timeline. For the given input 'node_id', and node_type 'Model',we find all the
         edge_names in which the 'node_id' is present as a source node. The edges are all part of valid forward
         timeline. The source_nodes and destination nodes in these edges are the candidates for the next recursive call
         of DFS"""

        node = node_collection.find_one({'node_id': node_id})
        forward_edges = node['source']
        backward_edges = node['destination']
        upstream_path = list()
        downstream_path = list()

        # Adding the upstream nodes to the upstream_path
        for edge_name in backward_edges:
            edge = edge_collection.find_one({'edge_name': edge_name})
            candidate_node = node_collection.find_one({'node_id': edge['source']})

            if candidate_node['node_type'] == 'model' and candidate_node['model_type'] != node['model_type']:
                # Any upstream node will always be node_type of 'model'
                upstream_path.append(candidate_node['node_id'])

        # Adding the downstream nodes to the downstream_path
        for forward_edge in forward_edges:
            edge = edge_collection.find_one({'edge_name': forward_edge})
            candidate_node = node_collection.find_one({'node_id': edge['destination']})

            if candidate_node['node_type'] == 'model' and candidate_node['model_type'] != node['model_type']:
                # Add the downstream node to the path
                downstream_path.append(candidate_node['node_id'])

            if candidate_node['node_type'] == 'intermediate' and candidate_node['model_type'] != node['model_type']:
                # Move forward to find a node_type of 'model'
                candidate_edge = edge_collection.find_one({'edge_name': candidate_node['source'][0]})
                model_node = node_collection.find_one({'node_id': candidate_edge['destination']})
                downstream_path.append(model_node['node_id'])

        if len(upstream_path) == 0:
            for downstream_node in downstream_path:
                path.append([str(downstream_node), str(node_id)])
        elif len(downstream_path) == 0:
            # This the case where node with node_type 'model' or 'intermediate'
            for upstream_node in upstream_path:
                path.append([str(upstream_node), str(node_id)])
        else:
            for upstream_node in upstream_path:
                for downstream_node in downstream_path:
                    path.append([str(upstream_node), str(node_id), str(downstream_node)])

        self.forward_timelines = path

        return self.forward_timelines

    def get_timeline_backward(self, node_id, path, node_collection, edge_collection):
        """Function to return the backward timeline. For the given input 'node_id', and node_type 'Model',we find all the
         edge_names in which the 'node_id' is present as a source node"""

        node = node_collection.find_one({'node_id': node_id})
        forward_edges = node['source']
        backward_edges = node['destination']

        # Adding the downstream nodes to the path
        for forward_edge in forward_edges:
            edge = edge_collection.find_one({'edge_name': forward_edge})
            candidate_node = node_collection.find_one({'node_id': edge['destination']})

            if candidate_node['node_type'] == 'model' and candidate_node['model_type'] != node['model_type']:
                # Add the downstream node to the path
                path = [str(candidate_node['node_id'])] + path

            if candidate_node['node_type'] == 'intermediate' and candidate_node['model_type'] != node['model_type']:
                # Move forward to find a node_type of 'model'
                candidate_edge = edge_collection.find_one({'edge_name': candidate_node['source'][0]})
                model_node = node_collection.find_one({'node_id': candidate_edge['destination']})
                path = [str(model_node['node_id'])] + path

        if node['node_type'] == 'model':
            # Only node_type of 'model' is added to the path
            path = [str(node_id)] + path

        # Adding the upstream nodes to the path
        for edge_name in backward_edges:
            edge = edge_collection.find_one({'edge_name': edge_name})
            candidate_node = node_collection.find_one({'node_id': edge['source']})

            if candidate_node['node_type'] == 'model' and candidate_node['model_type'] != node['model_type']:
                # Any upstream node will always be node_type of 'model'
                path = [str(candidate_node['node_id'])] + path

        # Performing dfs
        # Flag to check if node is source
        flag = True
        for edge_name in backward_edges:
            edge = edge_collection.find_one({'edge_name': edge_name})
            candidate_node = node_collection.find_one({'node_id': edge['source']})
            if candidate_node['model_type'] == node['model_type']:
                # Setting this flag to denote the node had a edge to previous-window
                flag = False
                self.get_timeline_backward(candidate_node['node_id'], path, node_collection, edge_collection)
                if candidate_node['node_type'] == 'intermediate':
                    # TODO: Need to work on case where all the intermediate aren't aggregated into a single model
                    break

        if flag:
            self.backward_timelines.append(path)

        return

    def get_timeline(self, node_id):
        """
        Function to generate the time line for a node from the model_graph.
        Assumption: the node with node_type model is always input as parameter
        """
        timelines = []
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            node = node_collection.find_one({'node_id': node_id})
            if node['node_type'] == 'model':
                self.get_timeline_tmp(node_id, list(), node_collection, edge_collection)
            else:
                edge = edge_collection.find_one({'source': node_id})
                candidate_node = node_collection.find_one({'node_id': edge['destination']})
                self.get_timeline_tmp(candidate_node['node_id'], list(), node_collection, edge_collection)

            for timeline in self.forward_timelines:
                data = dict({
                    'score': 1,
                    'timeline': []
                })
                for node_id in timeline:
                    model_type = node_collection.find_one({'node_id': ObjectId(node_id)}, {'model_type': 1, '_id': 0})['model_type']
                    node = dict({
                        'name': model_type,
                        '_id': node_id
                    })
                    data['timeline'].append(node)

                timelines.append(data)

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()
            return timelines

    def get_timeline_multi_window(self, node_id):
        """
        Function to generate the time line for a node from the model_graph.
        Assumption: the node with node_type model is always input as parameter
        """

        # TODO: Have to change the recursive dfs to an iterative one

        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']

            # Generating the forward timelines
            self.get_timeline_forward(node_id, list(), node_collection, edge_collection)

            # Generating the backward timelines
            node = node_collection.find_one({'node_id': node_id})
            backward_edges = node['destination']
            for edge_name in backward_edges:
                edge = edge_collection.find_one({'edge_name': edge_name})
                candidate_node = node_collection.find_one({'node_id': edge['source']})
                if candidate_node['model_type'] == node['model_type']:
                    if candidate_node['node_type'] == 'model':
                        self.get_timeline_backward(candidate_node['node_id'], list(), node_collection, edge_collection)
                    else:
                        # Move backward to find a candidate of node_type 'model'
                        back_edges_list = candidate_node['destination']
                        for back_edge_name in back_edges_list:
                            back_edge = edge_collection.find_one({'edge_name': back_edge_name})
                            back_node = node_collection.find_one({'node_id': back_edge['source']})
                            if back_node['model_type'] == node['model_type']:
                                self.get_timeline_backward(back_node['node_id'], list(), node_collection,
                                                           edge_collection)
                                break

                        break  # break out of outer-loop

            timelines = []
            # cartesian product of forward path and backward path to generate a timeline for the node
            for backward_timeline in self.backward_timelines:
                for forward_timeline in self.forward_timelines:
                    timelines.append(backward_timeline + forward_timeline)

            if len(self.backward_timelines) == 0:
                timelines = self.forward_timelines
            elif len(self.forward_timelines) == 0:
                timelines = self.backward_timelines

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return timelines
