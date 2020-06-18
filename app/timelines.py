from app.db_connect import DBConnect
from bson.objectid import ObjectId
from copy import deepcopy
import itertools
from sortedcontainers import SortedSet


class Timelines(DBConnect):
    def __init__(self):
        DBConnect.__init__(self)
        self.model_dependency_list = ['hurricane', 'flood', 'human_mobility']
        self.model_window_width = [172780, 43200, 10800]
        self.model_window_start = [1530403220, 1530403220, 1530403220]
        self.model_state = {"hurricane": "stateful", "flood": "stateful", "human_mobility": "stateless"}
        # TODO: Hard coded no of windows
        self.window_num = {"hurricane": 1, "flood": 5, "human_mobility": 6}
        self.paths = {"hurricane": [], "flood": [], "human_mobility": []}
        self.timelines = []

    def generate_timeline(self):
        start_nodes = self.find_start_model_nodes()
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            top_5_score = SortedSet()

            # Perform DFS on each "stateful" model_type
            for model in self.model_dependency_list:
                if self.model_state[model] == "stateful":
                    node_list = [node_id for node_id in start_nodes
                                 if node_collection.find_one({"node_id": node_id})["model_type"] == model]
                    for node_id in node_list:
                        self.dfs(node_id, model, node_collection, edge_collection, [])

            # Generating timelines
            # TODO: Hardcoded the stateful models here
            for candidate_timeline in itertools.product(self.paths["hurricane"], self.paths["flood"]):
                # Flatten the list
                timeline = [j for sub in candidate_timeline for j in sub]
                score = self.find_score(timeline, node_collection, edge_collection)
                timeline_set = set(timeline)

                for model in self.model_dependency_list:
                    if self.model_state[model] == "stateless":
                        max_window_num = self.window_num[model]
                        w_no = 1
                        while w_no < max_window_num:
                            candidate_node_list = list(
                                node_collection.find({"window_num": w_no, "node_type": "model", "model_type": model}))
                            max_score = 0
                            node_id_add = None

                            for candidate_node in candidate_node_list:
                                model_score = 0
                                backward_edge_list = edge_collection.find({"destination": candidate_node["node_id"]})
                                for backward_edge in backward_edge_list:
                                    source_id = backward_edge["source"]
                                    if source_id in timeline_set:
                                        model_score += 1
                                if max_score <= model_score:
                                    node_id_add = candidate_node["node_id"]
                                    max_score = model_score

                            if node_id_add is not None:
                                timeline.append(node_id_add)
                                score += max_score
                                timeline_set.add(node_id_add)
                            # else:
                            #     print("something went wrong")

                            w_no += 1
                # End of loop
                self.timelines.append(dict({"score": score, "nodes": timeline}))

                # print(score)
                # print(timeline)
                # print()

            # End of loop

        except Exception as e:
            raise e
        finally:
            self.disconnect_db()

    def dfs(self, node_id, model, node_collection, edge_collection, path):
        # Adding node_id to the path
        path.append(node_id)
        node = node_collection.find_one({"node_id": node_id})

        if node["window_num"] == self.window_num[model]:
            self.paths[model].append(path)
        # elif len(node["source"]) == 0 and node["window_num"] < self.window_num[model]:
        #     candidate_node_list = node_collection.find({"window_num": node["window_num"] + 1, "node_type": "model", "model_type": model})
        #     for candidate_node in candidate_node_list:
        #         self.dfs(candidate_node["node_id"], model, node_collection, edge_collection, path.copy())
        else:
            visited = set()
            for edge_name in node["source"]:
                edge = edge_collection.find_one({"edge_name": edge_name})
                destination = node_collection.find_one({"node_id": edge["destination"]})
                if destination["node_type"] == "model":
                    self.dfs(destination["node_id"], model, node_collection, edge_collection, path.copy())
                else:
                    # NOTE: There will be only one edge out of intermediate node, because it will only contribute to one
                    # cluster
                    destination_edge_name = destination["source"][0]
                    destination_edge = edge_collection.find_one({"edge_name": destination_edge_name})
                    candidate_node_id = destination_edge["destination"]
                    if candidate_node_id not in visited:
                        visited.add(candidate_node_id)
                        self.dfs(candidate_node_id, model, node_collection, edge_collection, path.copy())

    def find_score(self, timeline, node_collection, edge_collection):
        timeline_set = set(timeline)
        score = 0

        for t_node_id in timeline_set:
            t_node = node_collection.find_one({"node_id": t_node_id})
            backward_edges = edge_collection.find({"destination": t_node_id})
            t_flag = False
            for edge in backward_edges:
                source = node_collection.find_one({"node_id": edge["source"]})
                if source["model_type"] == t_node["model_type"] and source["node_type"] == "intermediate":
                    # We have to do only for one "intermediate" node
                    edge_list = edge_collection.find({"destination": source["node_id"]})
                    node_id_list = [edge["source"] for edge in edge_list]
                    for node_id in node_id_list:
                        if node_id in timeline_set:
                            score += 1

                    t_flag = True
                    break

            if not t_flag:
                for edge in backward_edges:
                    node_id = edge["source"]
                    if node_id in timeline_set:
                        score += 1

        return score

    def find_start_nodes(self):
        """Function to find the start nodes of the model_graph"""
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            database = self.MONGO_CLIENT["ds_results"]
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            dsir_collection = database["dsir"]
            # TODO: Hardcoded the start_time here
            dsir_list = dsir_collection.find({
                "metadata.temporal.begin": 1530403220,
                'created_by': {'$in': ['JobGateway', 'PostSynchronizationManager']}
            })
            start_nodes = list()

            for dsir in dsir_list:
                node = node_collection.find_one({"node_id": dsir["_id"]})
                if len(node['destination']) == 0 and node['node_type'] == 'model':
                    start_nodes.append(node['node_id'])
                else:
                    if node["node_type"] == "intermediate":
                        start_nodes.append(node["node_id"])
                    else:
                        source_edges = edge_collection.find({"destination": node["node_id"]})
                        t_flag = False
                        for edge in source_edges:
                            source = node_collection.find_one({"node_id": edge["source"]})
                            if source["model_type"] == node["model_type"]:
                                t_flag = True
                                break

                        if not t_flag:
                            start_nodes.append(node["node_id"])

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

    def find_start_model_nodes(self):
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            node_collection = model_graph_database['node']
            node_list = node_collection.find({"window_num": 1, "node_type": "model"})
            start_nodes = [node["node_id"] for node in node_list]
            return start_nodes

        except Exception as e:
            raise e
        finally:
            self.disconnect_db()

    def generate_window_number(self, start_nodes):
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT['model_graph']
            database = self.MONGO_CLIENT["ds_results"]
            node_collection = model_graph_database['node']
            edge_collection = model_graph_database['edge']
            dsir_collection = database["dsir"]
            config = model_graph_database["config"]
            window = 1

            while len(start_nodes) > 0:
                node_list = []
                for node_id in start_nodes:
                    node = node_collection.find_one({'node_id': node_id})
                    # updating the window_num for the node
                    node_collection.update({'node_id': node_id}, {'$set': {'window_num': window}})

                    # pre-processing the node
                    # finding the forward-links which are connected to model nodes of the same model-type
                    for edge_name in node['source']:
                        edge = edge_collection.find_one({'edge_name': edge_name})
                        destination = node_collection.find_one({'node_id': edge['destination']})
                        if destination['model_type'] == node['model_type'] and destination['node_type'] == 'model':
                            # updating the window_num for the model node
                            node_collection.update({'node_id': destination["node_id"]},
                                                   {'$set': {'window_num': window}})

                # Performing temporal comparison to find the nodes for the next window
                for i in range(len(self.model_dependency_list)):
                    begin = self.model_window_start[i] + self.model_window_width[i]
                    dsir_list = list(dsir_collection.find({
                        "metadata.temporal.begin": begin,
                        "metadata.model_type": self.model_dependency_list[i],
                        "created_by": "JobGateway"
                    }))
                    dsir_id_list = [dsir["_id"] for dsir in dsir_list]

                    if len(dsir_id_list) > 0:
                        node_list.extend(dsir_id_list)
                        # Moving the temporal begin to next window begin
                        self.model_window_start[i] += self.model_window_width[i]

                start_nodes = node_list
                window += 1

            # Updating the config with no of windows
            config.update_one({}, {"$set": {"no_of_simulations": window}})

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

    def get_timeline(self, node_id):
        self.generate_timeline()
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT["model_graph"]
            node_collection = model_graph_database["node"]
            edge_collection = model_graph_database["edge"]
            node = node_collection.find_one({"node_id": node_id})
            ret_timelines_list = []
            if node["node_type"] == "model":
                # Finding timelines for the node
                for timeline in self.timelines:
                    timeline_set = set(timeline["nodes"])
                    if node_id in timeline_set:
                        node_links = dict()
                        ret_timeline = []
                        # Adding all the intermediate nodes to the timeline set
                        for t_node_id in timeline["nodes"]:
                            backward_edges = edge_collection.find({"destination": t_node_id})
                            t_node = node_collection.find_one({"node_id": t_node_id})
                            node_links[t_node_id] = []
                            for edge in backward_edges:
                                backward_node = node_collection.find_one({"node_id": edge["source"]})
                                if backward_node["model_type"] == t_node["model_type"] and backward_node["node_type"] == "intermediate":
                                    timeline_set.add(backward_node["node_id"])
                                    node_links[backward_node["node_id"]] = []

                        # Now generating the timeline in Visualization format
                        for t_node_id in timeline_set:
                            forward_edges = edge_collection.find({"source": t_node_id})
                            adjacent_nodes = [edge["destination"] for edge in forward_edges]
                            for adj_node_id in adjacent_nodes:
                                if adj_node_id in timeline_set:
                                    node_links[t_node_id].append(adj_node_id)

                        for t_node_id, destination in node_links.items():
                            t_node = node_collection.find_one({"node_id": t_node_id})
                            if t_node_id == node_id:
                                ret_timeline.append({
                                    "name": t_node["model_type"],
                                    "_id": str(t_node_id),
                                    "destination": [str(des) for des in destination],
                                    "selected": True
                                })
                            else:
                                ret_timeline.append({
                                    "name": t_node["model_type"],
                                    "_id": str(t_node_id),
                                    "destination": [str(des) for des in destination],
                                })

                        ret_timelines_list.append({
                            "score": timeline["score"],
                            "links": ret_timeline
                        })

        except Exception as e:
            raise e

        finally:
            self.disconnect_db()

        return ret_timelines_list

    def find_node(self, node_id):
        try:
            self.connect_db()
            model_graph_database = self.GRAPH_CLIENT["model_graph"]
            node_collection = model_graph_database["node"]
            edge_collection = model_graph_database["edge"]
            node = node_collection.find_one({"node_id": node_id})
            if node["node_type"] == "model":
                return node["node_id"]
            else:
                forward_edges = edge_collection.find({"source": node_id})
                return forward_edges[0]["destination"]

        except Exception as e:
            raise e
        finally:
            self.disconnect_db()
