import numpy as np
from bson.objectid import ObjectId


class FlowGraphGenerator:

    def __init__(self):
        self.MONGO_CLIENT = None
        self.GRAPH_CLIENT = None
        self.EDGE_COUNT = 0

    def connect_db(self, db_connect):
        self.MONGO_CLIENT = db_connect.MONGO_CLIENT
        self.GRAPH_CLIENT = db_connect.GRAPH_CLIENT

    def disconnect_db(self):
        self.MONGO_CLIENT.close()
        self.GRAPH_CLIENT.close()

    def model_to_am(self):
        """ Function to generate the edge from model to wm. The dsar created by the AM contains a DSIR-LIST. Each DSIR
        contains a field 'parent' which points to the DSIRS from which it got sampled and aligned. Using one of this DSIR,
        find its DSAR, which should have been created by the window manager. Now perform another back-hop to move backward in
        the provenance to find the DSAR created by the PSM. These DSAR form the MODEL node, and the corresponding initial DSAR
        used form the AM node
        """

        database = self.MONGO_CLIENT['ds_results']
        dsar_collection = database['dsar']
        dsir_collection = database['dsir']

        flow_graph_path_database = self.GRAPH_CLIENT['flow_graph_path']
        edge_node_collection = flow_graph_path_database['edge_node']
        node_edge_collection = flow_graph_path_database['node_edge']

        flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
        edge_collection = flow_graph_constraint_database['edge']
        node_collection = flow_graph_constraint_database['node']

        dsar_list = dsar_collection.find({'created_by': 'AlignmentManager'})
        for dsar in dsar_list:
            dsir_list = dsar['dsir_list']
            # The initial DSAR of model_type = "undefined" will have an empty DSIR_LIST, and we ignore that DSAR.
            if len(dsir_list) > 0:
                source = list()
                dsir = dsir_collection.find_one({'_id': ObjectId(dsir_list[0])})
                parent_dsir_id_list = dsir['parents']
                parent_dsir = dsir_collection.find_one({'_id': ObjectId(parent_dsir_id_list[0])})
                wm_dsar = dsar_collection.find_one({'_id': ObjectId(parent_dsir['dsar'])})
                wm_dsir_list = wm_dsar['dsir_list']
                # fetching all the nodes of type model
                for dsir_id in wm_dsir_list:
                    dsir = dsir_collection.find_one({'_id': ObjectId(dsir_id)})
                    upstream_dsar = dsir['dsar']
                    node = dict()
                    node['node_id'] = upstream_dsar
                    node['type'] = 'dsar'
                    node['node_type'] = 'model'
                    node['model_type'] = dsir['metadata']['model_type']
                    source.append(node)
                    # inserting the node to the flow_graph_constraint collection
                    node_collection.update({'node_id': node['node_id']}, node, upsert=True)

                destination = dict()
                destination['node_id'] = dsar['_id']
                destination['type'] = 'dsar'
                destination['node_type'] = 'am'
                destination['model_type'] = dsar['metadata']['model_type']
                edge = dict({
                    'source': source,
                    'destination': destination,
                    'weight': list(np.random.rand(len(source))),
                    'edge_type': 'AND'
                })

                # inserting the destination node to the node collection in flow_graph_constraint database
                node_collection.update({'node_id': destination['node_id']}, destination, upsert=True)

                # inserting the edge to the edge collection in flow_graph_constraint database
                edge_collection.insert_one(edge)

                # creating edge name for flow_graph_path
                edge['edge_name'] = 'e' + str(self.EDGE_COUNT)
                self.EDGE_COUNT = self.EDGE_COUNT + 1

                # inserting a source-destination edge pair in edge_node collection in flow_graph_rep database
                edge_node_collection.insert_one(edge)

                # inserting the destination node in node_edge collection in flow_graph_rep database
                node_edge_collection.update({
                    'node_id': ObjectId(edge['destination']['node_id'])
                },
                    {
                        '$push': {'destination_edge': edge['edge_name']},
                        '$set': {
                            'type': 'dsar',
                            'node_type': 'am',
                            'model_type': edge['destination']['model_type']
                        }
                    }, upsert=True)

                # inserting the source node in node_edge collection in graph database
                for node in edge['source']:
                    node_edge_collection.update({
                        'node_id': ObjectId(node['node_id']),
                    },
                        {
                            '$push': {'source_edge': edge['edge_name']},
                            '$set': {
                                'type': 'dsar',
                                'node_type': 'model',
                                'model_type': node['model_type']
                            }

                        }, upsert=True)

        return

    def am_to_job(self):
        """Function to generate the edge from am to job. The jobs contains a DSAR list. There is one DSAR for each upstream
        model. These DSAR form the AM node. Each job outputs a single DSIR, which is stored in 'output_dsir' field of a job.
        This DSIR forms a JOB node. The edge weights are relevance scores"""

        database = self.MONGO_CLIENT['ds_results']
        jobs_collection = database['jobs']
        dsar_collection = database['dsar']

        flow_graph_path_database = self.GRAPH_CLIENT['flow_graph_path']
        edge_node_collection = flow_graph_path_database['edge_node']
        node_edge_collection = flow_graph_path_database['node_edge']

        flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
        edge_collection = flow_graph_constraint_database['edge']
        node_collection = flow_graph_constraint_database['node']

        jobs_cursor = jobs_collection.find({})
        for job in jobs_cursor:
            source = list()
            for input_dsar in job['input_dsars']:
                dsar = dsar_collection.find_one({'_id': ObjectId(input_dsar)})
                node = dict()
                node['node_id'] = input_dsar
                node['type'] = 'dsar'
                node['node_type'] = 'am'
                node['model_type'] = dsar['metadata']['model_type']
                source.append(node)

                # inserting the node to the flow_graph_constraint database
                node_collection.update({'node_id': node['node_id']}, node, upsert=True)

            destination = dict({
                'node_id': job['output_dsir'],
                'type': 'dsir',
                'node_type': 'job',
                'model_type': job['model_type']
            })

            # inserting the node to the node collection in flow_graph_constraint database
            node_collection.update({'node_id': destination['node_id']}, destination, upsert=True)

            # edge weight is taken as relevance
            edge = dict({
                'source': source,
                'destination': destination,
                'weight': list(np.full(len(source), job['relevance'])),
                'edge_type': 'OR'
            })

            # inserting the edge to the edge collection in flow_graph_constraint database
            edge_collection.insert_one(edge)

            # creating edge name for flow_graph_path
            edge['edge_name'] = 'e' + str(self.EDGE_COUNT)
            self.EDGE_COUNT = self.EDGE_COUNT + 1

            # inserting a source-destination edge pair in edge_node collection in flow_graph_path database
            edge_node_collection.insert_one(edge)

            # inserting the destination node in node_edge collection in flow_graph_path database
            node_edge_collection.update({
                'node_id': ObjectId(edge['destination']['node_id'])
            },
                {
                    '$push': {'destination_edge': edge['edge_name']},
                    '$set': {
                        'type': 'dsir',
                        'node_type': 'job',
                        'model_type': edge['destination']['model_type']
                    }
                }, upsert=True)

            # inserting the source vertex in vertex_edge collection in graph database
            for node in edge['source']:
                node_edge_collection.update({
                    'node_id': ObjectId(node['node_id'])
                },
                    {
                        '$push': {'source_edge': edge['edge_name']},
                        '$set': {
                            'type': 'dsar',
                            'node_type': 'am',
                            'model_type': node['model_type']
                        }
                    }, upsert=True)

            # end of loop

        return

    def job_to_model(self):
        """Function to generate a edge from job to cluster_aggr and from cluster_aggr to model. For each DSIR created by the
        PostSynchronizationManager, its backward provenance is checked. If its parent DSIR which is created by the
        JobGateway lies in the same window, then it will serve as a source node. If no such a parent DSIR exists, then we
        create a dummy node containing the same DSIR and create a link between the two nodes.
        """

        database = self.MONGO_CLIENT['ds_results']
        dsar_collection = database['dsar']
        dsir_collection = database['dsir']

        flow_graph_path_database = self.GRAPH_CLIENT['flow_graph_path']
        edge_node_collection = flow_graph_path_database['edge_node']
        node_edge_collection = flow_graph_path_database['node_edge']

        flow_graph_constraint_database = self.GRAPH_CLIENT['flow_graph_constraint']
        edge_collection = flow_graph_constraint_database['edge']
        node_collection = flow_graph_constraint_database['node']

        dsar_list = dsar_collection.find({"created_by": "PostSynchronizationManager"})

        for dsar in dsar_list:
            dsir_list_id = dsar['dsir_list']
            vertex_list = list()
            # check for potential aggregation, check for clustering missing
            for dsir_id in dsir_list_id:
                dsir = dsir_collection.find_one({'_id': ObjectId(dsir_id)})
                # extracting the parents of the dsir
                dsir_parents_id = dsir['parents']
                temporal_begin = dsir['metadata']['temporal']['begin']
                temporal_end = dsir['metadata']['temporal']['end']
                model_type = dsir['metadata']['model_type']

                vertex_job_list = list()
                # finding the parent-dsirs(created_by: jobs) to create the link
                for dsir_parent_id in dsir_parents_id:
                    dsir_parent = dsir_collection.find_one({'_id': dsir_parent_id})
                    if dsir_parent['created_by'] == 'JobGateway' and \
                            dsir_parent['metadata']['temporal']['begin'] == temporal_begin and \
                            dsir_parent['metadata']['temporal']['end'] == temporal_end and \
                            dsir_parent['metadata']['model_type'] == model_type:
                        node = dict({
                            'node_id': dsir_parent_id,
                            'type': 'dsir',
                            'node_type': 'job',
                            'model_type': dsir_parent['metadata']['model_type']
                        })
                        vertex_job_list.append(node)
                        # inserting the node to the node collection in flow_graph_constraint database
                        node_collection.update({'node_id': node['node_id']}, node, upsert=True)

                # creating a dummy relationship from job to cluster_aggr if there is neither clustering nor aggregation
                if len(vertex_job_list) == 0:
                    node = dict({
                        'node_id': dsir_id,
                        'type': 'dsir',
                        'node_type': 'job',
                        'model_type': dsir['metadata']['model_type']
                    })
                    vertex_job_list.append(node)
                    # inserting the dummy node to the node collection in flow_graph_constraint database
                    node_collection.update({'node_id': node['node_id']}, node, upsert=True)

                destination = dict({
                    'node_id': dsir_id,
                    'type': 'dsir',
                    'node_type': 'cluster_aggr',
                    'model_type': dsir['metadata']['model_type']
                })
                # inserting the destination node to the node collection in flow_graph_constraint database
                node_collection.update({'node_id': destination['node_id'], 'node_type': 'cluster_aggr'}, destination,
                                       upsert=True)

                edge = dict({
                    'source': vertex_job_list,
                    'destination': destination,
                    'weight': list(np.random.rand(len(vertex_job_list))),
                    'edge_type': 'OR'
                })

                # inserting the edge to the edge collection in flow_graph_constraint database
                edge_collection.insert_one(edge)

                # creating edge name for flow_graph_path
                edge['edge_name'] = 'e' + str(self.EDGE_COUNT)
                self.EDGE_COUNT = self.EDGE_COUNT + 1

                # inserting a source-destination edge pair in edge_vertex collection in graph database
                edge_node_collection.insert_one(edge)

                # inserting the destination node in node_edge collection in graph database
                node_edge_collection.update({
                    'node_id': ObjectId(edge['destination']['node_id']),
                    'node_type': 'cluster_aggr'
                },
                    {
                        '$push': {'destination_edge': edge['edge_name']},
                        '$set': {
                            'type': 'dsir',
                            'node_type': 'cluster_aggr',
                            'model_type': destination['model_type']
                        }

                    }, upsert=True)

                # inserting the source nodes in node_edge collection in graph database
                for node in edge['source']:
                    node_edge_collection.update({
                        'node_id': ObjectId(node['node_id']),
                        'node_type': 'job'
                    },
                        {
                            '$push': {'source_edge': edge['edge_name']},
                            '$set': {
                                'type': 'dsir',
                                'node_type': 'job',
                                'model_type': node['model_type']
                            }
                        }, upsert=True)

                vertex_list.append(destination)
                # End of loop

            # creating a model node to link the cluster_aggr node
            destination = dict({
                'node_id': dsar['_id'],
                'type': 'dsar',
                'node_type': 'model',
                'model_type': dsar['metadata']['model_type']
            })

            # inserting the node to the node collection in flow_graph_constraint database
            node_collection.update({'node_id': destination['node_id']}, destination, upsert=True)

            edge = dict({
                'source': vertex_list,
                'destination': destination,
                'weight': list(np.random.rand(len(vertex_list))),
                'edge_type': 'AND'
            })

            # inserting the edge to the edge collection in flow_graph_constraint database
            edge_collection.insert_one(edge)

            # creating edge name for flow_graph_path
            edge['edge_name'] = 'e' + str(self.EDGE_COUNT)
            self.EDGE_COUNT = self.EDGE_COUNT + 1

            # inserting a source-destination edge pair in edge_node collection in flow_graph_path database
            edge_node_collection.insert_one(edge)

            # inserting the destination node in node_edge collection in flow_graph_path database
            node_edge_collection.update({
                'node_id': ObjectId(edge['destination']['node_id']),
                'node_type': 'model'
            },
                {
                    '$push': {'destination_edge': edge['edge_name']},
                    '$set': {
                        'type': 'dsar',
                        'node_type': 'model',
                        'model_type': destination['model_type']
                    }
                }, upsert=True)

            # inserting the source node in node_edge collection in flow_graph_path database
            for node in vertex_list:
                node_edge_collection.update({
                    'node_id': ObjectId(node['node_id']),
                    'node_type': 'cluster_aggr'
                },
                    {
                        '$push': {'source_edge': edge['edge_name']},
                        '$set': {
                            'type': 'dsir',
                            'node_type': 'cluster_aggr',
                            'model_type': node['model_type']
                        }
                    }, upsert=True)

        return

    def delete_data(self):
        # removing all existing edge-node documents in edge_node collection in flow_graph_path database
        self.GRAPH_CLIENT['flow_graph_path']['edge_node'].remove({})
        # removing all existing node-edge vertex pairs in flow_graph_path database
        self.GRAPH_CLIENT['flow_graph_path']['node_edge'].remove({})
        # removing all existing edges in edge collection in flow_graph_constraint database
        self.GRAPH_CLIENT['flow_graph_constraint']['edge'].remove({})
        # removing all existing nodes in node collection in flow_graph_constraint database
        self.GRAPH_CLIENT['flow_graph_constraint']['node'].remove({})

    def generate_flow_graph(self):
        self.delete_data()
        self.model_to_am()
        self.am_to_job()
        self.job_to_model()
