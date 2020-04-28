class ModelGraphGenerator:
    def __init__(self):
        self.MONGO_CLIENT = None
        self.GRAPH_CLIENT = None
        self.MODEL_GRAPH = None
        self.OPTIMAL_PATH = list()

    def connect_db(self, db_connect):
        self.MONGO_CLIENT = db_connect.MONGO_CLIENT
        self.GRAPH_CLIENT = db_connect.GRAPH_CLIENT

    def disconnect_db(self):
        self.MONGO_CLIENT.close()
        self.GRAPH_CLIENT.close()

    @staticmethod
    def create_node(dsir):
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

    @staticmethod
    def get_timeline(path_finder, node_id, node_type='model'):
        """"""
        flow_graph_node_timeline = path_finder.get_timeline(node_id, node_type)
        timeline = list()
        # transpiling the flow path to timeline for the node
        for node in flow_graph_node_timeline:
            if node['node_type'] == 'model':
                timeline.append(node_id)

        return timeline

    def insert_edge(self, dsir, node):
        """If the current DSIR is created by the ‘JobGateway’, we look at its nearest ancestor which are created by
    ‘PostSynchronizationManager’ or ‘JobGateway’ of the previous window and join them using an edge.

    If the current DSIR is created by the ‘PostSynchronizationManager’, we look at its nearest ancestor which are
    created by the ‘JobGateway’ or ‘PostSynchronizationManager’ of the previous window and join them using an edge. """

        database = self.MONGO_CLIENT['ds_results']
        dsir_collection = database['dsir']

        if dsir['created_by'] == 'JobGateway':
            for dsir_parent_id in dsir['parents']:
                dsir_parent = dsir_collection.find_one({'_id': dsir_parent_id})  # This is a DSIR
                # created by AM

                # Move backward to find the next ancestor DSIR which is created by the PSM or JobGateway
                dsir_ancestor_id_list = dsir_parent['parents']
                for dsir_ancestor_id in dsir_ancestor_id_list:
                    node['periods'][0]['connector'].append({
                        'connectTo': str(dsir_ancestor_id),
                        'connectorType': "start-finish"
                    })

        elif dsir['created_by'] == 'PostSynchronizationManager':
            for dsir_parent_id in dsir['parents']:
                dsir_parent = dsir_collection.find_one({'_id': dsir_parent_id})  # This is a DSIR
                # created by JobGateway

                # Move backward to find the 1st level ancestor DSIR which is created by the AM
                dsir_ancestor_id_list = dsir_parent['parents']
                for dsir_ancestor_id in dsir_ancestor_id_list:
                    dsir_ancestor = dsir_collection.find_one({'_id': dsir_ancestor_id})
                    # Move backward to find the 2nd level ancestor DSIR which is created by the JobGateway or PSM
                    dsir_ancestor_2_id_list = dsir_ancestor['parents']

                    for dsir_ancestor_2_id in dsir_ancestor_2_id_list:
                        node['periods'][0]['connector'].append({
                            'connectTo': str(dsir_ancestor_2_id),
                            'connectorType': "start-finish"
                        })

        return node

    def generate_model_graph(self):
        """Function to generate the flow graph. Its done by parsing the DSIRS. A DSIR created by the PSM becomes a node by
        default, but the DSIR created by JobGateway undergoes the following sanity checks. We check whether children DSIRs
        are created by PSM or not, if not, then the DSIR becomes a node"""

        graph = list()

        database = self.MONGO_CLIENT['ds_results']
        dsir_collection = database['dsir']
        dsir_list = dsir_collection.find()

        for dsir in dsir_list:

            # TODO: check if more than a single DSIR exists for a single model on a particular window

            if dsir['created_by'] == 'JobGateway':
                # Check if the DSIR is not getting post-processed in PSM by checking the 'created_by' of any of its
                # children. If the DSIR doesn't have any children
                if 'children' not in dsir:
                    node = self.create_node(dsir)

                    # Generating backward edge
                    if len(dsir['parents']) == 0:
                        # No edge to insert
                        graph.append(node)
                    else:
                        # Generating backward edge
                        node = self.insert_edge(dsir, node)
                        graph.append(node)
                else:
                    # fetching a child
                    dsir_child = dsir_collection.find_one({'_id': dsir['children'][0]})

                    if dsir_child['created_by'] == 'PostSynchronizationManager':
                        # if the children DSIR is created by PSM, then we don't insert this DSIR
                        continue
                    else:
                        node = self.create_node(dsir)
                        # Generating backward edge
                        node = self.insert_edge(dsir, node)
                        graph.append(node)

            elif dsir['created_by'] == 'PostSynchronizationManager':
                node = self.create_node(dsir)
                # Generating backward edge
                node = self.insert_edge(dsir, node)
                graph.append(node)
            # End of loop

        self.MODEL_GRAPH = graph

    def generate_optimal_path(self, model):
        """Function to generate the optimal path for model graph by transpiling the optimal path in the flow graph.
        Its done by taking into consideration of the nodes of type 'model' which are part of the optimal path. These nodes
        are of data-type DSAR, in which the DSIRs wrap around, are the nodes in the model graph"""

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

    def get_optimal_path(self):
        return self.OPTIMAL_PATH

    def get_model_graph(self):
        return self.MODEL_GRAPH
