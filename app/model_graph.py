from bson.objectid import ObjectId


class ModelGraph:
    def __init__(self, mongo_client, graph_client, workflow_id):
        self.config = None
        self.OPTIMAL_PATHS = list()
        self.EDGE_COUNT = 0
        self.MONGO_CLIENT = mongo_client
        self.GRAPH_CLIENT = graph_client
        # TODO: Hardcoded the model_dependency here and workflow_id here
        self.model_dependency_list = ['flood', 'hurricane', 'human_mobility']
        self.config = self.MONGO_CLIENT["ds_config"]["workflows"].find_one({"_id": workflow_id})
        self.workflow_id = workflow_id

    def create_node(self, dsir):
        """Function to convert the DSIR to a  node for the flow_graph_path"""
        node = dict()
        node['name'] = dsir['metadata']['model_type'] + "_" + str(dsir["_id"])
        node['periods'] = [dict({'id': str(dsir['_id']), 'end': int(dsir['metadata']['temporal']['end']) * 1000, 'connector': list()})]

        if dsir['created_by'] != 'PostSynchronizationManager':
            node['periods'][0]['start'] = int(dsir['metadata']['temporal']['begin']) * 1000
        return node

    def store_edge(self, source, destination, edge_name):
        try:
            edge_collection = self.GRAPH_CLIENT["model_graph"]["edge"]
            # inserting the edge into the database
            edge_collection.insert({
                'source': source,
                'destination': destination,
                'edge_name': edge_name
            })
        except Exception as e:
            raise e

    def update_node(self, node_id, node_type, model_type, source_edge, destination_edge):
        try:
            node_collection = self.GRAPH_CLIENT["model_graph"]["node"]
            if len(source_edge) > 0:
                node_collection.update({'node_id': node_id, }, {
                    '$push': {
                        'source': source_edge,
                    },
                    '$setOnInsert': {
                        'destination': [],
                        'model_type': model_type,
                        'node_type': node_type
                    }
                }, upsert=True)
            elif len(destination_edge) > 0:
                node_collection.update({'node_id': node_id}, {
                    '$push': {
                        'destination': destination_edge,
                    },
                    '$setOnInsert': {
                        'source': [],
                        'model_type': model_type,
                        'node_type': node_type
                    }
                }, upsert=True)
            else:
                node_collection.update({'node_id': node_id}, {
                    '$setOnInsert': {
                        'destination': [],
                        'source': [],
                        'model_type': model_type,
                        'node_type': node_type,
                    }
                }, upsert=True)
        except Exception as e:
            raise e

    def generate_edge(self, dsir, node):
        """
        We only insert the forward edge for a node. Edge insertion takes place as follows.

        If the current DSIR is created by the ‘JobGateway’, we look at its nearest descendant which is created by
        ‘PostSynchronizationManager’ of the present window or ‘JobGateway’ of the next window, and join them using an
        edge.

        If the current DSIR is created by the ‘PostSynchronizationManager’, we look at its nearest descendant which is
        created by the ‘JobGateway’ of the next window and join them using an edge.

        """
        try:
            dsir_collection = self.MONGO_CLIENT["ds_results"]["dsir"]
            # Base Case
            if 'children' not in dsir:
                self.update_node(dsir["_id"], "model", dsir["metadata"]["model_type"], "", "")
                return node

            if dsir["created_by"] == "JobGateway":
                for child_id in dsir["children"]:
                    dsir_child = dsir_collection.find_one({"_id": child_id})

                    if dsir_child["created_by"] == "PostSynchronizationManager":
                        # The dsir_child is part of the graph. Now, we generate edge between them
                        node["periods"][0]["connector"].append({"connectTo": str(child_id), "connectorType": "finish-start"})

                        # generating a edge_name
                        edge_name = "e" + str(self.EDGE_COUNT)
                        self.EDGE_COUNT = self.EDGE_COUNT + 1
                        # storing the edge to the database
                        self.store_edge(dsir["_id"], child_id, edge_name)
                        # Storing the dsir_child, its always a "model" node
                        self.update_node(child_id, "model", dsir_child["metadata"]["model_type"], "", edge_name)

                        if self.config["model"][dsir["metadata"]["model_type"]]["post_synchronization_settings"]["aggregation_strategy"] == "average":
                            # storing the dsir, its always a "intermediate" node
                            self.update_node(dsir["_id"], "intermediate", dsir["metadata"]["model_type"], edge_name, "")

                        else:
                            # storing the dsir, its always a "model" node
                            self.update_node(dsir["_id"], "model", dsir["metadata"]["model_type"], edge_name, "")
                    elif dsir_child["created_by"] == "AlignmentManager":
                        # Move forward to find the descendant DSIR which is created by the JobGateway
                        dsir_descendant_id_list = dsir_child['children']
                        for dsir_descendant_id in dsir_descendant_id_list:
                            node["periods"][0]["connector"].append({
                                "connectTo": str(dsir_descendant_id),
                                "connectorType": "finish-start"
                            })
                            # fetching the DSIR descendant
                            dsir_descendant = dsir_collection.find_one({'_id': dsir_descendant_id})
                            # generating a edge_name
                            edge_name = 'e' + str(self.EDGE_COUNT)
                            self.EDGE_COUNT = self.EDGE_COUNT + 1
                            # storing the edge to the database
                            self.store_edge(dsir['_id'], dsir_descendant_id, edge_name)
                            # storing the dsir, its always a "model" node
                            self.update_node(dsir["_id"], "model", dsir["metadata"]["model_type"], edge_name, "")
                            # storing the dsir_descendant
                            if self.config["model"][dsir_descendant["metadata"]["model_type"]] \
                                    ["post_synchronization_settings"]["aggregation_strategy"] == "average":
                                # The dsir_descendant is a "intermediate" node
                                self.update_node(dsir_descendant['_id'], "intermediate", dsir_descendant['metadata']['model_type'], "", edge_name)
                            else:
                                # The dsir_descendant is a "model" node
                                self.update_node(dsir_descendant['_id'], "model", dsir_descendant['metadata']['model_type'], "", edge_name)

            elif dsir["created_by"] == "PostSynchronizationManager":
                for child_id in dsir["children"]:
                    dsir_child = dsir_collection.find_one({"_id": child_id})  # This is a DSIR created by AlignmentManager
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
                        self.store_edge(dsir['_id'], dsir_descendant_id, edge_name)
                        # storing the dsir, its always a "model" node
                        self.update_node(dsir['_id'], "model", dsir['metadata']['model_type'], edge_name, '')
                        if self.config["model"][dsir_descendant["metadata"]["model_type"]] \
                                ["post_synchronization_settings"]["aggregation_strategy"] == "average":
                            # The dsir_descendant is a "intermediate" node
                            self.update_node(dsir_descendant['_id'], "intermediate", dsir_descendant['metadata']['model_type'], "", edge_name)
                        else:
                            # The dsir_descendant is a "model" node
                            self.update_node(dsir_descendant['_id'], "model", dsir_descendant['metadata']['model_type'], "", edge_name)
            return node

        except Exception as e:
            raise e

    def generate_model_graph(self):
        """Function to generate the flow graph. Its done by parsing the DSIRS. A DSIR created by the PSM becomes a node
        by default, but the DSIR created by JobGateway undergoes the following sanity checks.
        We check whether children DSIRs are created by PSM or not, if not, then the DSIR becomes a node"""

        try:
            dsir_collection = self.MONGO_CLIENT["ds_results"]["dsir"]
            dsir_list = list(dsir_collection.find({
                "created_by": {"$in": ["JobGateway", "PostSynchronizationManager"]},
                "workflow_id": self.workflow_id
            }))
            graph = list()
            for dsir in dsir_list:
                # TODO: check if more than a single DSIR exists for a single model on a particular window
                # creating the graph node
                node = self.create_node(dsir)
                # generating the forward edges
                node = self.generate_edge(dsir, node)
                # adding the node to the graph
                graph.append(node)
            # End of for
            response = {"min_time": self.config["simulation_context"]["temporal"]["begin"],
                        "max_time": self.config["simulation_context"]["temporal"]["end"], "graph": graph}
        except Exception as e:
            raise e
        return response

    def delete_data(self):
        """Deleting the data in model_graph database in GRAPH_CLIENT"""
        try:
            # removing all existing edge-node documents in edge_node collection in flow_graph_path database
            self.GRAPH_CLIENT['model_graph']['edge'].remove({})
            # removing all existing node-edge vertex pairs in flow_graph_path database
            self.GRAPH_CLIENT['model_graph']['node'].remove({})
        except Exception as e:
            raise e

    def generate_window_number(self):
        try:
            node_collection = self.GRAPH_CLIENT['model_graph']['node']
            edge_collection = self.GRAPH_CLIENT['model_graph']['edge']
            dsir_collection = self.MONGO_CLIENT["ds_results"]["dsir"]

            dsir_list = dsir_collection.find({"metadata.temporal.begin": self.config["simulation_context"]["temporal"]["begin"],
                                              "created_by": "JobGateway"})
            start_nodes = [dsir["_id"] for dsir in dsir_list]
            model_window_start = [self.config["simulation_context"]["temporal"]["begin"]] * len(self.model_dependency_list)
            model_window_width = [model_config["output_window"] for model_config in self.config["model"].values()]
            window = 1

            while len(start_nodes) > 0:
                node_list = []
                for node_id in start_nodes:
                    node = node_collection.find_one({'node_id': node_id})
                    # updating the window_num for the node
                    node_collection.update({'node_id': node_id}, {'$set': {'window_num': window}})

                    # pre-processing the node
                    if self.config["model"][node["model_type"]]["post_synchronization_settings"]["aggregation_strategy"] == "average":
                        # The forward-links which are connected to this node belongs to the same window_num
                        edge_list = edge_collection.find({'source': node["node_id"]})
                        for edge in edge_list:
                            # updating the window_num for the "model" node
                            node_collection.update({"node_id": edge["destination"]}, {"$set": {'window_num': window}})

                # Performing temporal comparison to find the nodes for the next window
                for i in range(len(self.model_dependency_list)):
                    begin = model_window_start[i] + model_window_width[i]
                    dsir_list = list(dsir_collection.find({"metadata.temporal.begin": begin, "metadata.model_type": self.model_dependency_list[i],
                                                           "created_by": "JobGateway"}))
                    dsir_id_list = [dsir["_id"] for dsir in dsir_list]
                    if len(dsir_id_list) > 0:
                        node_list.extend(dsir_id_list)
                        # Moving the temporal begin to next window begin
                        model_window_start[i] += model_window_width[i]

                start_nodes = node_list
                window += 1

        except Exception as e:
            raise e

    def get_dsfr(self, timestamp, dsir_id_list):
        try:
            dsfr_collection = self.MONGO_CLIENT["ds_results"]["dsfr"]
            return list(dsfr_collection.find({"parent": {"$in": dsir_id_list}, "timestamp": timestamp}))
        except Exception as e:
            raise e
