
class PathFinder:
    def __init__(self):
        self.MONGO_CLIENT = None
        self.GRAPH_CLIENT = None

    def connect_db(self, db_connect):
        self.MONGO_CLIENT = db_connect.MONGO_CLIENT
        self.GRAPH_CLIENT = db_connect.GRAPH_CLIENT

    def disconnect_db(self):
        self.MONGO_CLIENT.close()
        self.GRAPH_CLIENT.close()

    def get_timeline_forward(self, node_id, node_type, path):
        """Function to return the forward timeline. For the given input 'node_id' we find all the edge_names in which the
        'node_id' is present as a source node. The edges are all part of valid forward timeline. The source_nodes and
        destination nodes in these edges are the candidates for the next recursive call of DFS"""

        flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
        node_collection = flow_graph_constraint_database['node']

        flow_graph_path_database = self.GRAPH_CLIENT['flow_graph_path']
        node_edge_collection = flow_graph_path_database['node_edge']
        edge_node_collection = flow_graph_path_database['edge_node']

        # fetching the node from the node_collection in flow_graph_constraint database
        node = node_collection.find_one({
            'node_id': node_id,
            'node_type': node_type
        })

        path.append(node)

        edges = node_edge_collection.find_one({
            'node_id': node_id, 'node_type': node_type
        }, {
            'source_edge': 1,
            '_id': 0
        })

        forward_edges = list()

        if 'source_edge' in edges:
            forward_edges = edges['source_edge']

        for fr_edge_name in forward_edges:
            # fetching the nodes in the fr_edge
            edge = edge_node_collection.find_one({'edge_name': fr_edge_name})
            for source in edge['source']:
                if source['node_id'] != node_id:
                    self.get_timeline_forward(source['node_id'], source['node_type'], path)

            destination = edge['destination']
            self.get_timeline_forward(destination['node_id'], destination['node_type'], path)

        return path

    def get_timeline_backward(self, node_id, node_type, path):
        """Function to return the backward timeline"""

        flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
        node_collection = flow_graph_constraint_database['node']

        flow_graph_path_database = self.GRAPH_CLIENT['flow_graph_path']
        node_edge_collection = flow_graph_path_database['node_edge']
        edge_node_collection = flow_graph_path_database['edge_node']

        # fetching the node from the node_collection in flow_graph_constraint database
        node = node_collection.find_one({
            'node_id': node_id,
            'node_type': node_type
        })

        path.append(node)

        edges = node_edge_collection.find_one({
            'node_id': node_id, 'node_type': node_type
        }, {
            'destination_edge': 1,
            '_id': 0
        })

        backward_edges = list()

        if 'destination_edge' in edges:
            backward_edges = edges['destination_edge']

        for bk_edge_name in backward_edges:
            edge = edge_node_collection.find_one({'edge_name': bk_edge_name})
            for source in edge['source']:
                self.get_timeline_backward(source['node_id'], source['node_type'], path)

        return path

    def get_timeline(self, node_id, node_type='model'):
        """Function to generate the time line for a node from the flow_graph"""
        # generating the forward path
        forward_timeline = self.get_timeline_forward(node_id, node_type, list())
        # generating the backward path
        backward_timeline = self.get_timeline_backward(node_id, node_type, list())
        # joining both forward path and backward path to generate a timeline for the node
        timeline = backward_timeline[1:] + forward_timeline
        return timeline

