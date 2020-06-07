from app.db_connect import DBConnect
from bson.objectid import ObjectId


class PathFinder(DBConnect):
    def __init__(self):
        DBConnect.__init__(self)
        self.model_dependency_list = ['hurricane', 'flood', 'human_mobility']
        self.timelines = dict()
        for model in self.model_dependency_list:
            self.timelines[model] = dict()
            self.timelines[model]['window_num'] = 1
            self.timelines[model]['timelines'] = list()

    def extract_timelines_node(self, node):
        timelines = self.timelines[node['model_type']]['timelines']
        node_timelines = list()
        for timeline in timelines:
            for t_node in timeline:
                if t_node['_id'] == node['node_id']:
                    node_timelines.append(timeline)

        return node_timelines

    def get_node_index_timeline(self, node, timeline):
        for i in range(len(timeline)):
            if timeline[i]['_id'] == node['node_id']:
                return i

        return None

    def merge_timelines(self, timeline1, timeline2):
        merged_timeline = []
        for t2_node in timeline2:
            for t1_node in timeline1:
                if t1_node['_id'] == t2_node['_id']:
                    t1_des = set(t1_node['destination'])
                    t2_des = set(t2_node['destination'])
                    destination = list(t1_des.union(t2_des))
                    merged_timeline.append({'name': t1_node['name'], '_id': t1_node['_id'], 'destination': destination})
                    break

        for t1_node in timeline1:
            t_flag = False
            for m_node in merged_timeline:
                if t1_node['_id'] == m_node['_id']:
                    t_flag = True
                    break

            if not t_flag:
                merged_timeline.append(t1_node)

        for t2_node in timeline2:
            t_flag = False
            for m_node in merged_timeline:
                if t2_node['_id'] == m_node['_id']:
                    t_flag = True
                    break

            if not t_flag:
                merged_timeline.append(t2_node)

        return merged_timeline

    def stitch_timeline(self, prev_timeline, simulation_context, node_collection, edge_collection):
        # print('simulation_context : ' + str(simulation_context) + '\n')
        # print('timeline : ' + str(prev_timeline) + '\n')
        timelines = [prev_timeline]
        for sc_node in simulation_context:
            node = node_collection.find_one({'node_id': sc_node['_id']})

            if len(node['destination']) == 0:
                for i in range(len(timelines)):
                    t_flag = False

                    for t_node in timelines[i]:
                        if t_node['_id'] == sc_node['_id']:
                            sc_des = set(sc_node['destination'])
                            t_des = set(t_node['destination'])
                            destination = list(sc_des.union(t_des))
                            t_node['destination'] = destination
                            t_flag = True
                            break

                    if not t_flag:
                        timelines[i].append(sc_node)
            else:
                for edge_name in node['destination']:
                    edge = edge_collection.find_one({'edge_name': edge_name})
                    source = node_collection.find_one({'node_id': edge['source']})

                    # Find the back-edge linking to the previous window
                    if source['model_type'] == sc_node['name']:
                        if source["node_type"] == "model":
                            del_timelines_index = []
                            merged_timelines = list()
                            for i in range(len(timelines)):
                                index = self.get_node_index_timeline(source, timelines[i])
                                if index is not None:
                                    destination_list = timelines[i][index]['destination']
                                    for d_node_id in destination_list:
                                        d_node = node_collection.find_one({'node_id': d_node_id})
                                        # check for multiple forward edges to the same model_type of the node
                                        if d_node['model_type'] == node['model_type']:
                                            timelines[i][index]['destination'].remove(d_node_id)

                                    timelines[i][index]['destination'].append(sc_node['_id'])
                                    timelines[i].append(sc_node)
                                else:
                                    ext_timelines = self.extract_timelines_node(source)
                                    del_timelines_index.append(i)
                                    for ext_timeline in ext_timelines:
                                        merged_timelines.append(self.merge_timelines(timelines[i], ext_timeline))

                            for i in range(len(del_timelines_index)):
                                del timelines[i]

                            if len(merged_timelines) > 0:
                                timelines += merged_timelines
                        else:
                            for i in range(len(timelines)):
                                timelines[i].append(sc_node)
                            break
        return timelines

    def generate_timelines(self):
        node_list = self.find_start_nodes()
        self.generate_window_number(node_list)
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            # A set to keep track of visited nodes
            visited_nodes = set()

            tmp_timelines = dict()
            del_timelines = dict()
            for model_type in self.model_dependency_list:
                tmp_timelines[model_type] = dict()
                tmp_timelines[model_type]['timelines'] = list()
                del_timelines[model_type] = dict()
                del_timelines[model_type]['timelines_index'] = set()

            while len(node_list) > 0:
                # finding all the nodes part of current simulation_context
                simulation_context_node_id = node_list[0]
                simulation_context_node = node_collection.find_one({'node_id': simulation_context_node_id})
                del node_list[0]

                if simulation_context_node_id in visited_nodes:
                    continue

                visited_nodes.add(simulation_context_node_id)
                context_node_list = [simulation_context_node_id]
                simulation_context_visited = [simulation_context_node_id]
                simulation_context = [
                    dict({
                        'name': simulation_context_node['model_type'],
                        '_id': simulation_context_node['node_id'],
                        'destination': []
                    })
                ]

                while len(context_node_list) > 0:
                    node_id = context_node_list[0]
                    del context_node_list[0]

                    # Adding all the upstream models to the simulation_context
                    node = node_collection.find_one({'node_id': node_id})
                    for edge_name in node['destination']:
                        edge = edge_collection.find_one({'edge_name': edge_name})
                        source = node_collection.find_one({'node_id': edge['source']})

                        if node['model_type'] != source['model_type'] or \
                                (node['model_type'] == source['model_type'] and source['node_type'] == 'intermediate'):

                            if source['node_id'] not in simulation_context_visited:
                                # Adding the node to the simulation_context_visited
                                simulation_context_visited.append(source['node_id'])
                                # Adding the node to the simulation_context
                                simulation_context.append(dict({
                                    'name': source['model_type'],
                                    '_id': source['node_id'],
                                    'destination': [node_id]
                                }))
                                # Adding the node ot the context_node
                                context_node_list.append(source['node_id'])
                            else:
                                for sc_node in simulation_context:
                                    if sc_node['_id'] == source['node_id']:
                                        sc_node['destination'].append(node_id)
                                        break

                # print(str(simulation_context_node['node_id']) + ': ' + str(simulation_context))

                # Stitching the simulation_context to the timelines
                timelines = self.timelines[simulation_context_node['model_type']]['timelines']
                window_num = self.timelines[simulation_context_node['model_type']]['window_num']

                #  Base-case, adding the timeline for the first-window
                if window_num == simulation_context_node['window_num']:
                    self.timelines[simulation_context_node['model_type']]['timelines'].append(simulation_context)

                # Now store the tmp_timeline to timeline variable, since we moved to the window_num + 2
                elif window_num + 1 < simulation_context_node['window_num']:
                    model_type = simulation_context_node['model_type']

                    # Deleting the timelines which got evolved
                    for index in del_timelines[model_type]['timelines_index']:
                        del self.timelines[model_type]['timelines'][index]

                    # Adding the new timelines
                    self.timelines[model_type]['timelines'] += tmp_timelines[model_type]['timelines']
                    self.timelines['window_num'] += 1

                if window_num < simulation_context_node['window_num']:
                    visited_scr_nodes = set()
                    for edge_name in simulation_context_node['destination']:
                        edge = edge_collection.find_one({'edge_name': edge_name})
                        source = node_collection.find_one({'node_id': edge['source']})

                        if source['model_type'] == simulation_context_node['model_type']:
                            if source["node_type"] == "intermediate":
                                # Need to move backward to find a node_type of "model"
                                for src_edge_name in source["destination"]:
                                    src_edge = edge_collection.find_one({"edge_name": src_edge_name})
                                    src_node = node_collection.find_one({"node_id": src_edge["source"]})
                                    if src_node["model_type"] == simulation_context_node["model_type"]:
                                        # Search for the timeline in which the source is present and merge the
                                        # simulation_context
                                        if src_node["node_id"] not in visited_scr_nodes:
                                            visited_scr_nodes.add(src_node["node_id"])
                                            for i in range(0, len(timelines)):
                                                for t_node in timelines[i]:
                                                    if src_node['node_id'] == t_node['_id']:
                                                        new_timelines = self.stitch_timeline(timelines[i].copy(),
                                                                                             simulation_context.copy(),
                                                                                             node_collection,
                                                                                             edge_collection)
                                                        # Adding to the del_timeline
                                                        del_timelines[simulation_context_node['model_type']][
                                                            'timelines_index'] \
                                                            .add(i)
                                                        # saving the new_timeline
                                                        for new_timeline in new_timelines:
                                                            tmp_timelines[simulation_context_node['model_type']] \
                                                                ['timelines'].append(new_timeline)
                        else:
                            # Search for the timeline in which the source is present and merge the
                            # simulation_context
                            for i in range(0, len(timelines)):
                                for t_node in timelines[i]:
                                    if source['node_id'] == t_node['_id']:
                                        new_timelines = self.stitch_timeline(timelines[i].copy(),
                                                                             simulation_context.copy(),
                                                                             node_collection, edge_collection)
                                        # Adding to the del_timeline
                                        del_timelines[simulation_context_node['model_type']]['timelines_index'] \
                                            .add(i)
                                        # saving the new_timeline
                                        for new_timeline in new_timelines:
                                            tmp_timelines[simulation_context_node['model_type']]['timelines'] \
                                                .append(new_timeline)

                # Finding the nodes for next simulation_context
                # TODO: Need a check for intermediate nodes
                for edge_name in simulation_context_node['source']:
                    edge = edge_collection.find_one({'edge_name': edge_name})
                    destination = node_collection.find_one({'node_id': edge['destination']})
                    if destination['model_type'] == simulation_context_node['model_type']:
                        if destination["node_type"] == "model":
                            if destination["node_id"] not in visited_nodes:
                                node_list.append(destination["node_id"])
                        else:
                            # Need to find a node_type "model".
                            # All intermediate nodes have a single source edge
                            des_edge_name = destination["source"][0]
                            des_edge = edge_collection.find_one({"edge_name": des_edge_name})
                            if des_edge["destination"] not in visited_nodes:
                                node_list.append(des_edge["destination"])

                # End of loop
            for model_type in self.model_dependency_list:
                # Deleting the timelines which got evolved
                for index in del_timelines[model_type]['timelines_index']:
                    del self.timelines[model_type]['timelines'][index]

                # Adding the timelines which evolved to next window
                if len(tmp_timelines[model_type]['timelines']) > 0:
                    self.timelines[model_type]['timelines'] += tmp_timelines[model_type]['timelines']
                    self.timelines[model_type]['window_num'] += 1

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

    def find_start_nodes(self):
        """Function to generate all the timelines"""
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            nodes = node_collection.find({})
            start_nodes = list()

            for node in nodes:
                if len(node['destination']) == 0 and node['node_type'] == 'model':
                    start_nodes.append(node['node_id'])
                else:
                    flag = True
                    for edge_name in node['destination']:
                        edge = edge_collection.find_one({'edge_name': edge_name})
                        source = node_collection.find_one({'node_id': edge['source']})
                        if source['model_type'] == node['model_type']:
                            flag = False
                            break

                    if flag:
                        if node['node_type'] == 'model':
                            start_nodes.append(node['node_id'])
                        else:
                            # the intermediate node is only connected to the nodes of the same model_type
                            for edge_name in node['source']:
                                edge = edge_collection.find_one({'edge_name': edge_name})
                                if edge['destination'] not in start_nodes:
                                    start_nodes.append(edge['destination'])

            ordered_start_nodes = []
            for model in self.model_dependency_list:
                for node_id in start_nodes:
                    node = node_collection.find_one({'node_id': node_id})
                    if node['model_type'] == model:
                        ordered_start_nodes.append(node_id)

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return ordered_start_nodes

    def generate_window_number(self, start_nodes):
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            window = 1

            while len(start_nodes) > 0:
                node_list = []

                for node_id in start_nodes:
                    node = node_collection.find_one({'node_id': node_id})
                    # updating the window_num for the node
                    node_collection.update({'node_id': node_id}, {'$set': {'window_num': window}})

                    # pre-processing the node
                    # finding the back-links which are connected to intermediate nodes of the same model-type
                    for edge_name in node['destination']:
                        edge = edge_collection.find_one({'edge_name': edge_name})
                        source = node_collection.find_one({'node_id': edge['source']})
                        if source['model_type'] == node['model_type'] and source['node_type'] == 'intermediate':
                            # updating the window_num for the intermediate node
                            node_collection.update({'node_id': edge['source']}, {'$set': {'window_num': window}})

                    # Performing DFS to find the nodes for the next window
                    for edge_name in node['source']:
                        edge = edge_collection.find_one({'edge_name': edge_name})
                        destination = node_collection.find_one({'node_id': edge['destination']})
                        if destination['model_type'] == node['model_type']:
                            if destination['node_type'] == 'model':
                                node_list.append(destination['node_id'])
                            else:
                                # Updating the window_no for the nodes of node_type 'intermediate'
                                # Move forward to the node of node_type 'model'
                                for destination_edge_name in destination['source']:
                                    destination_edge = edge_collection.find_one({'edge_name': destination_edge_name})
                                    node_list.append(destination_edge['destination'])

                start_nodes = node_list
                window += 1

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

    def get_timeline(self):
        """
        Function to generate the time line for a node from the model_graph.
        Assumption: the node with node_type model is always input as parameter
        """

        # TODO: Have to change the recursive dfs to an iterative one

        try:
            self.generate_timelines()

        except Exception as e:
            raise e

        return self.timelines
