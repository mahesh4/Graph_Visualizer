from app.path_finder import PathFinder
from app.db_connect import DBConnect


class ModelGraphGenerator(DBConnect):

    def __init__(self):
        DBConnect.__init__(self)
        self.MODEL_GRAPH = None
        self.OPTIMAL_PATH = list()

    def create_node(self, dsir):
        """Function to convert the DSIR to a  node for the flow_graph_path"""
        node = dict()
        node['name'] = dsir['metadata']['model_type']
        node['periods'] = [
            dict({
                'id': str(dsir['_id']),
                'start': dsir['metadata']['temporal']['begin'],
                'end': dsir['metadata']['temporal']['end'],
                'connector': list()
            })
        ]
        return node

    def get_timeline(self, node_id, node_type='model'):
        """Function to get the timeline for a node. Its done by transpiling the timeline for a particular node in flow graph to
        a timeline for that node in model graph
        """
        path_finder = PathFinder()
        flow_graph_node_timeline = path_finder.get_timeline(node_id, node_type)
        timeline = list()
        # transpiling the flow path to timeline for the node
        for node in flow_graph_node_timeline:
            if node['node_type'] == 'model':
                timeline.append(str(node_id))

        return timeline

    def insert_edge(self, dsir, node):
        """
        We only insert the forward edge for a node. Edge insertion takes place as follows.

        If the current DSIR is created by the ‘JobGateway’, we look at its nearest descendant which is created by
    ‘PostSynchronizationManager’ of the present window or ‘JobGateway’ of the next window, and join them using an edge.

    If the current DSIR is created by the ‘PostSynchronizationManager’, we look at its nearest descendant which is
    created by the ‘JobGateway’ of the next window and join them using an edge. """

        database = self.MONGO_CLIENT['ds_results']
        dsir_collection = database['dsir']

        if dsir['created_by'] == 'JobGateway':
            if 'children' in dsir:
                for child_id in dsir['children']:
                    dsir_child = dsir_collection.find_one({'_id': child_id})

                    if dsir_child['created_by'] == 'PostSynchronizationManager':
                        # The dsir_descendant is also part of the graph, and we generate edges between them
                        node['periods'][0]['connector'].append({
                            'connectTo': str(child_id),
                            'connectorType': "finish-start"
                        })

                    elif dsir_child['created_by'] == 'AlignmentManager':
                        # Move forward to find the next descendant DSIR which is created by the JobGateway
                        dsir_descendant_id_list = dsir_child['children']
                        for dsir_descendant_id in dsir_descendant_id_list:
                            node['periods'][0]['connector'].append({
                                'connectTo': str(dsir_descendant_id),
                                'connectorType': "finish-start"
                            })

        elif dsir['created_by'] == 'PostSynchronizationManager':

            if 'children' in dsir:
                for child_id in dsir['children']:
                    dsir_child = dsir_collection.find_one({'_id': child_id})  # This is a DSIR
                    # created by AlignmentManager

                    # Move forward to find the descendant DSIR which is created by the JobGateway
                    dsir_descendant_id_list = dsir_child['children']
                    for dsir_descendant_id in dsir_descendant_id_list:
                        node['periods'][0]['connector'].append({
                            'connectTo': str(dsir_descendant_id),
                            'connectorType': "finish-start"
                        })

        return node

    def generate_model_graph(self):
        """Function to generate the flow graph. Its done by parsing the DSIRS. A DSIR created by the PSM becomes a node
        by default, but the DSIR created by JobGateway undergoes the following sanity checks.
        We check whether children DSIRs are created by PSM or not, if not, then the DSIR becomes a node"""

        self.connect_db()
        try:
            graph = list()
            database = self.MONGO_CLIENT['ds_results']
            dsir_collection = database['dsir']
            dsir_list = dsir_collection.find({'created_by': {'$in': ['JobGateway', 'PostSynchronizationManager']}})

            for dsir in dsir_list:

                # TODO: check if more than a single DSIR exists for a single model on a particular window
                # creating the graph node
                node = self.create_node(dsir)
                if 'children' in dsir:
                    # Generating forward edge
                    node = self.insert_edge(dsir, node)

                graph.append(node)

            self.MODEL_GRAPH = graph

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return

    def generate_optimal_path(self, model):
        """Function to generate the optimal path for model graph by transpiling the optimal path in the flow graph.
        Its done by taking into consideration of the nodes of type 'model' which are part of the optimal path. These nodes
        are of data-type DSAR, in which the DSIRs wrap around, are the nodes in the model graph"""
        try:
            self.connect_db()
            database = self.MONGO_CLIENT['ds_results']
            dsar_collection = database['dsar']
            flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
            path = list()
            edge_collection = flow_graph_constraint_database['edge']
            edge_list = edge_collection.find({'destination.node_type': 'model'})
            for edge in edge_list:
                dsar_id = edge['destination']['node_id']
                # Check if the model is part of the optimal path
                if model.getVarByName(edge['edge_names'][0]).x == 1:
                    # Get the DSIR of the model, which serves as a node in model_graph
                    dsar = dsar_collection.find_one({'_id': dsar_id})
                    dsir_id_list = dsar['dsir_list']
                    path.extend(dsir_id_list)

            self.OPTIMAL_PATH = path

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return

    def get_optimal_path(self):
        return self.OPTIMAL_PATH

    def get_model_graph(self):
        return self.MODEL_GRAPH
