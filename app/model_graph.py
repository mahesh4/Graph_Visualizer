from app.path_2 import PathFinder
from app.db_connect import DBConnect
from bson.objectid import ObjectId


class ModelGraphGenerator(DBConnect):

    def __init__(self):
        DBConnect.__init__(self)
        self.MODEL_GRAPH = None
        self.OPTIMAL_PATHS = list()
        self.model_dependency_list = ['hurricane', 'flood', 'human_mobility']
        self.EDGE_COUNT = 0

    def create_node(self, dsir):
        """Function to convert the DSIR to a  node for the flow_graph_path"""
        node = dict()
        node['name'] = dsir['metadata']['model_type']
        node['periods'] = [
            dict({
                'id': str(dsir['_id']),
                'end': int(dsir['metadata']['temporal']['end']) * 1000,
                'connector': list()
            })
        ]
        if dsir['created_by'] != 'PostSynchronizationManager':
            node['periods'][0]['start'] = int(dsir['metadata']['temporal']['begin']) * 1000

        return node

    def get_timeline(self, node_id):
        """Function to get the timeline for a node"""
        path_finder = PathFinder()
        paths = path_finder.get_timeline()
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']

            config_collection = model_graph_database['config']
            config = config_collection.find_one({})
            no_of_simulations = config["no_of_simulations"]

            node_timelines = list()
            for model_type in reversed(self.model_dependency_list):
                timelines = paths[model_type]['timelines']
                for timeline in timelines:
                    for t_node in timeline:
                        if t_node['_id'] == node_id:
                            node_timelines.append(timeline)

                if len(node_timelines) > 0:
                    break

            # Post processing the timelines
            ordered_timelines = []
            for timeline in node_timelines:
                ord_timeline = []
                for window_num in range(1, no_of_simulations + 1):
                    for model in self.model_dependency_list:
                        index = 0
                        while index < len(timeline):
                            t_node = timeline[index]
                            index += 1
                            t_window_num = node_collection.find_one({"node_id": ObjectId(t_node["_id"])})["window_num"]
                            if window_num == t_window_num and model == t_node["name"]:
                                t_node['_id'] = str(t_node['_id'])
                                t_node['destination'] = [str(destination) for destination in t_node['destination']]
                                if str(node_id) == t_node['_id']:
                                    t_node['selected'] = True
                                ord_timeline.append(t_node)

                ordered_timelines.append(ord_timeline)
        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return ordered_timelines

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
            node_collection.update({'node_id': node_id, }, {
                '$push': {
                    'source': source_edge,
                },
                '$setOnInsert': {
                    'destination': []
                },
                '$set': {
                    'model_type': model_type,
                    'node_type': node_type
                }
            }, upsert=True)

        if len(destination_edge) > 0:
            node_collection.update({'node_id': node_id}, {
                '$push': {
                    'destination': destination_edge,
                },
                '$setOnInsert': {
                    'source': []
                },
                '$set': {
                    'model_type': model_type,
                    'node_type': node_type
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
                node_collection.update({'node_id': dsir["_id"], }, {
                    '$setOnInsert': {
                        'destination': [],
                        'source': []
                    },
                    '$set': {
                        'model_type': dsir["metadata"]["model_type"],
                        'node_type': "model",
                    }
                }, upsert=True)
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
                        self.update_node(dsir['_id'],"intermediate", dsir['metadata']['model_type'], edge_name, '',
                                         node_collection)
                        self.update_node(child_id, "model",dsir['metadata']['model_type'], '', edge_name,
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
                            self.update_node(dsir['_id'], "model",dsir['metadata']['model_type'], edge_name, '',
                                             node_collection)
                            # self.update_node(dsir_descendant_id, dsir_descendant['metadata']['model_type'], '',
                            #                  edge_name, node_collection)
                            node_collection.update({'node_id': dsir_descendant_id}, {
                                '$push': {
                                    'destination': edge_name,
                                },
                                '$setOnInsert': {
                                    'source': []
                                },
                                '$set': {
                                    'model_type': dsir_descendant['metadata']['model_type'],
                                }
                            }, upsert=True)

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
                        self.update_node(dsir['_id'], "model",dsir['metadata']['model_type'], edge_name, '',
                                         node_collection)
                        # self.update_node(dsir_descendant_id,dsir_descendant['metadata']['model_type'], '',
                        #                  edge_name, node_collection)
                        node_collection.update({'node_id': dsir_descendant_id}, {
                            '$push': {
                                'destination': edge_name,
                            },
                            '$setOnInsert': {
                                'source': []
                            },
                            '$set': {
                                'model_type': dsir_descendant['metadata']['model_type'],
                            }
                        }, upsert=True)

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

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return graph

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

    def generate_node_label(self):
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            node_list = node_collection.find({"model_type": "flood"})
            for node in node_list:
                backward_edges = edge_collection.find({"destination": node["node_id"]})
                b_flag = False
                for edge in backward_edges:
                    backward_node = node_collection.find_one({"node_id": edge["source"]})
                    if backward_node["model_type"] != node["model_type"]:
                        if len(node["source"]) == 0:
                            node_collection.update_one({"node_id": node["node_id"]},
                                                       {"$set": {"node_type": "model"}})
                        else:
                            node_collection.update_one({"node_id": node["node_id"]}, {"$set": {"node_type": "intermediate"}})
                        b_flag = True
                        break

                if not b_flag:
                    node_collection.update_one({"node_id": node["node_id"]}, {"$set": {"node_type": "model"}})

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()
