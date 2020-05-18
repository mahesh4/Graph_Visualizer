from app.path import PathFinder
from app.db_connect import DBConnect


class ModelGraphGenerator(DBConnect):

    def __init__(self):
        DBConnect.__init__(self)
        self.MODEL_GRAPH = None
        self.OPTIMAL_PATHS = list()
        self.EDGE_COUNT = 0

    def create_node(self, dsir):
        """Function to convert the DSIR to a  node for the flow_graph_path"""
        node = dict()
        node['name'] = dsir['metadata']['model_type']
        node['periods'] = [
            dict({
                'id': str(dsir['_id']),
                'start': dsir['metadata']['temporal']['begin'] * 1000,
                'end': dsir['metadata']['temporal']['end'] * 1000,
                'connector': list()
            })
        ]
        return node

    def get_timeline(self, node_id):
        """Function to get the timeline for a node"""
        path_finder = PathFinder()
        timelines = path_finder.get_timeline(node_id)
        return timelines

    def store_edge(self, source, destination, edge_name, edge_collection):
        try:
            # inserting the edge into the database
            edge_collection.insert({
                'source': source,
                'destination': destination,
                'edge_name': edge_name
            })

        except Exception as e:
            raise e

    def update_node(self, node_id, node_type, model_type, source_edge, destination_edge, node_collection):
        if len(source_edge) > 0:
            node_collection.update({'node_id': node_id}, {
                '$push': {
                    'source': source_edge,
                },
                '$set': {
                    'model_type': model_type,
                    'node_type': node_type,
                    'destination': []
                }
            }, upsert=True)

        if len(destination_edge) > 0:
            node_collection.update({'node_id': node_id}, {
                '$push': {
                    'destination': destination_edge,
                },
                '$set': {
                    'model_type': model_type,
                    'node_type': node_type,
                    'source': []
                }
            }, upsert=True)

    def generate_edge(self, dsir, node, dsir_collection, node_collection, edge_collection):
        """
        We only insert the forward edge for a node. Edge insertion takes place as follows.

        If the current DSIR is created by the ‘JobGateway’, we look at its nearest descendant which is created by
        ‘PostSynchronizationManager’ of the present window or ‘JobGateway’ of the next window, and join them using an
        edge.

        If the current DSIR is created by the ‘PostSynchronizationManager’, we look at its nearest descendant which is
        created by the ‘JobGateway’ of the next window and join them using an edge.

        """
        try:
            if 'children' not in dsir:
                return node

            if dsir['created_by'] == 'JobGateway':
                for child_id in dsir['children']:
                    dsir_child = dsir_collection.find_one({'_id': child_id})

                    if dsir_child['created_by'] == 'PostSynchronizationManager':
                        # The dsir_child is part of the graph. Now, we generate edge between them
                        node['periods'][0]['connector'].append({
                            'connectTo': str(child_id),
                            'connectorType': "finish-start"
                        })

                        # generating a edge_name
                        edge_name = 'e' + str(self.EDGE_COUNT)
                        self.EDGE_COUNT = self.EDGE_COUNT + 1
                        # storing the edge to the database
                        self.store_edge(dsir['_id'], child_id, edge_name, edge_collection)
                        # storing the nodes
                        self.update_node(dsir['_id'], 'intermediate', dsir['metadata']['model_type'], edge_name, '',
                                         node_collection)
                        self.update_node(child_id, 'model', dsir['metadata']['model_type'], '', edge_name,
                                         node_collection)

                    elif dsir_child['created_by'] == 'AlignmentManager':
                        # Move forward to find the descendant DSIR which is created by the JobGateway
                        dsir_descendant_id_list = dsir_child['children']
                        for dsir_descendant_id in dsir_descendant_id_list:
                            node['periods'][0]['connector'].append({
                                'connectTo': str(dsir_descendant_id),
                                'connectorType': "finish-start"
                            })
                            # fetching the DSIR descendant
                            dsir_descendant = dsir_collection.find_one({'_id': dsir_descendant_id})
                            # generating a edge_name
                            edge_name = 'e' + str(self.EDGE_COUNT)
                            self.EDGE_COUNT = self.EDGE_COUNT + 1
                            # storing the edge to the database
                            self.store_edge(dsir['_id'], dsir_descendant_id, edge_name, edge_collection)
                            # storing the nodes
                            self.update_node(dsir['_id'], 'model', dsir['metadata']['model_type'], edge_name, '',
                                             node_collection)
                            self.update_node(dsir_descendant_id, 'model', dsir_descendant['metadata']['model_type'], '', edge_name,
                                             node_collection)

            elif dsir['created_by'] == 'PostSynchronizationManager':
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
                        # fetching the DSIR descendant
                        dsir_descendant = dsir_collection.find_one({'_id': dsir_descendant_id})
                        # generating a edge_name
                        edge_name = 'e' + str(self.EDGE_COUNT)
                        self.EDGE_COUNT = self.EDGE_COUNT + 1
                        # storing the edge to the database
                        self.store_edge(dsir['_id'], dsir_descendant_id, edge_name, edge_collection)
                        # storing the nodes
                        self.update_node(dsir['_id'], 'model', dsir['metadata']['model_type'], edge_name, '',
                                         node_collection)
                        self.update_node(dsir_descendant_id, 'model', dsir_descendant['metadata']['model_type'], '', edge_name,
                                         node_collection)

            return node

        except Exception as e:
            raise e

    def generate_model_graph(self):
        """Function to generate the flow graph. Its done by parsing the DSIRS. A DSIR created by the PSM becomes a node
        by default, but the DSIR created by JobGateway undergoes the following sanity checks.
        We check whether children DSIRs are created by PSM or not, if not, then the DSIR becomes a node"""

        try:
            self.connect_db()

            database = self.MONGO_CLIENT['ds_results']
            dsir_collection = database['dsir']
            dsir_list = list(dsir_collection.find({'created_by': {'$in': ['JobGateway', 'PostSynchronizationManager']}}))

            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']

            graph = list()

            for dsir in dsir_list:
                # TODO: check if more than a single DSIR exists for a single model on a particular window

                # creating the graph node
                node = self.create_node(dsir)
                # generating the forward edges
                node = self.generate_edge(dsir, node, dsir_collection, node_collection, edge_collection)

                # adding the node to the graph
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

            self.OPTIMAL_PATHS = path

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return

    def get_optimal_path(self):
        return self.OPTIMAL_PATHS

    def get_model_graph(self):
        return self.MODEL_GRAPH

    def delete_data(self):
        try:
            self.connect_db()
            self.EDGE_COUNT = 0
            # removing all existing edge-node documents in edge_node collection in flow_graph_path database
            self.GRAPH_CLIENT['model_graph']['edge'].remove({})
            # removing all existing node-edge vertex pairs in flow_graph_path database
            self.GRAPH_CLIENT['model_graph']['node'].remove({})

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()
