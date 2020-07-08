from bson.objectid import ObjectId
from collections import defaultdict
import heapq
import itertools
import math


class Timelines:
    def __init__(self, mongo_client, graph_client):
        self.MONGO_CLIENT = mongo_client
        self.GRAPH_CLIENT = graph_client
        # TODO: Hard coded no of windows for each model
        self.model_dependency_list = {"hurricane": [], "flood": [0], "human_mobility": [0, 1]}
        self.window_count = {"hurricane": 1, "flood": 5, "human_mobility": 6}
        self.model_paths = {"hurricane": [], "flood": [], "human_mobility": []}
        self.timelines = []
        self.config = self.MONGO_CLIENT["ds_config"]["workflows"].find_one({"_id": ObjectId("5ee5ad3820c7f46abb64a069")})

    def get_top_k_timelines(self, k):
        try:
            node_collection = self.GRAPH_CLIENT['model_graph']['node']
            edge_collection = self.GRAPH_CLIENT["model_graph"]["edge"]
            timelines_index_list = self.generate_timelines(k)

            # TODO: Hardcoded here
            model_type_list = ["hurricane", "flood", "human_mobility"]
            timelines_list = []

            for timelines_index in timelines_index_list:
                timeline = []
                score = timelines_index[0]
                index_list = timelines_index[1]

                # Adding all the intermediate nodes to the stateful nodes path
                for i in range(len(model_type_list)):
                    if self.config["model"][model_type_list[i]]["post_synchronization_settings"]["aggregation_strategy"] == "average":
                        model_path = self.model_paths[model_type_list[i]][index_list[i]]
                        new_model_path = []
                        for node_id in model_path:
                            node = node_collection.find_one({"node_id": node_id})
                            # We are not processing nor adding "intermediate" nodes present in the model_path directly, but we are adding them
                            # through adjacent_node_id_list
                            if node["node_type"] == "model":
                                # All the forward adjacent nodes are "intermediate" nodes
                                adjacent_node_id_list = map(lambda x: x["destination"], edge_collection.find({"source": node_id}))
                                new_model_path.append(node_id)
                                new_model_path.extend(adjacent_node_id_list)
                        # Adding the new_model_path to the self.model_paths
                        self.model_paths[model_type_list[i]][index_list[i]] = new_model_path

                timeline_node_set = set.union(*map(set, [self.model_paths[model_type_list[i]][index_list[i]] for i in range(len(index_list))]))
                for i in range(len(index_list)):
                    model_type = model_type_list[i]
                    model_path = self.model_paths[model_type][index_list[i]]
                    for node_id in model_path:
                        node = {"name": model_type, "_id": str(node_id), "destination": []}
                        adjacent_node_id_list = map(lambda x: x["destination"], edge_collection.find({"source": node_id}))
                        for adjacent_node_id in adjacent_node_id_list:
                            if adjacent_node_id in timeline_node_set:
                                node["destination"].append(str(adjacent_node_id))

                        timeline.append(node)
                        # End of for
                    # End of for
                # End of for
                timelines_list.append({"score": score, "links": timeline})
        except Exception as e:
            raise e

        return timelines_list

    def generate_timelines(self, k):
        try:
            node_collection = self.GRAPH_CLIENT['model_graph']['node']
            edge_collection = self.GRAPH_CLIENT["model_graph"]["edge"]
            stateful_models = [model_name for model_name, model_config in self.config["model"].items() if model_config["stateful"]]
            stateless_models = [model_name for model_name, model_config in self.config["model"].items() if not model_config["stateful"]]

            # Perform DFS on each "stateful" model_type
            for model_type in stateful_models:
                visited = set()
                if self.config["model"][model_type]["post_synchronization_settings"]["aggregation_strategy"] == "average":
                    node_list = node_collection.find({"window_num": 1, "node_type": "intermediate", "model_type": model_type})
                    for node in node_list:
                        forward_edges = edge_collection.find({"source": node["node_id"]})
                        destination_list = [edge["destination"] for edge in forward_edges]
                        for destination_id in destination_list:
                            destination = node_collection.find_one({"node_id": destination_id})
                            if destination["model_type"] == model_type and destination_id not in visited:
                                visited.add(destination_id)
                                self.dfs(node, model_type, [])
                else:
                    node_list = node_collection.find({"window_num": 1, "node_type": "model", "model_type": model_type})
                for node in node_list:
                    self.dfs(node, model_type, [])

            # Finding the most compatible paths on each "stateless" model_type
            for model_type in stateless_models:
                if self.config["model"][model_type]["post_synchronization_settings"]["aggregation_strategy"] == "average":
                    node_list = node_collection.find({"window_num": 1, "node_type": "intermediate", "model_type": model_type})
                else:
                    node_list = node_collection.find({"window_num": 1, "node_type": "model", "model_type": model_type})

                for node in node_list:
                    self.find_most_compatible_path(node, model_type, [])

            # Generating the doc_list for Fagin's algorithm
            doc_list = []
            for model_type in self.model_paths:
                lst_score = []
                for i in range(len(self.model_paths[model_type])):
                    # Finding the score for the path
                    score = self.find_model_path_score(self.model_paths[model_type][i], model_type)
                    lst_score.append({i: score})
                doc_list.append(lst_score)

            doc_list = [sorted(lst, key=lambda k: list(k.values()), reverse=True) for lst in doc_list]
            max_doc_list_len = max([len(lst) for lst in doc_list])

            # Preprocessing doc_list so that all lists have equal length
            # for lst in doc_list:
            #     lst_len = len(lst)
            #     if lst_len < max_doc_list_len:
            #         lst.extend([{lst_len + 1: 0}] * (max_doc_list_len - len(lst)))

            # Running NRA
            top_k_timelines = []
            heapq.heapify(top_k_timelines)
            k_score = 0

            for row_idx in range(max_doc_list_len):
                candidate_doc_list = [col[: row_idx + 1] if row_idx < len(col) else col[:] for col in doc_list]
                for candidate_timeline in itertools.product(*candidate_doc_list):
                    count = len([True for i in range(len(candidate_timeline)) if candidate_doc_list[i].index(candidate_timeline[i]) < row_idx])
                    # Base Case, We had already visited this timeline
                    if count == len(doc_list):
                        continue
                    else:
                        compatibility, score = self.check_compatiblity(candidate_timeline)
                        if compatibility and (k > len(top_k_timelines) or score > k_score):
                            if 0 < len(top_k_timelines) == k:
                                heapq.heappop(top_k_timelines)

                            heapq.heappush(top_k_timelines, (score, [list(path.keys())[-1] for path in candidate_timeline]))
                            k_score = heapq.nsmallest(1, top_k_timelines)[0][0]
                            if len(top_k_timelines) == k:
                                return top_k_timelines
                # End of for
            # End of for
        except Exception as e:
            raise e
        # No of timelines is less than k
        return top_k_timelines

    def check_compatiblity(self, timeline):
        # TODO: Hardcoded here
        model_type_list = ["hurricane", "flood", "human_mobility"]
        score = 0
        for i in range(len(timeline)):
            model_type = model_type_list[i]
            model_path_index = list(timeline[i].keys())[-1]
            score += list(timeline[i].values())[-1]
            for j in self.model_dependency_list[model_type]:
                up_model_path_index = list(timeline[j].keys())[-1]
                up_model_type = model_type_list[j]
                no_of_edges = self.find_no_of_edges(self.model_paths[up_model_type][up_model_path_index],
                                                    self.model_paths[model_type][model_path_index])
                if no_of_edges < math.floor(self.window_count[model_type] / 2):
                    return False, -1
            # End of for
        # End of for
        return True, score

    def dfs(self, node, model_type, path):
        """Function to perform dfs
        NOTE: We don't maintain visited list because the graph is acyclic. Cycles occur during clustering and there is a local visited set
        maintained"""
        try:
            node_collection = self.GRAPH_CLIENT["model_graph"]["node"]
            edge_collection = self.GRAPH_CLIENT["model_graph"]["edge"]

            # Adding node_id to the path and visited
            path.append(node["node_id"])

            if node["window_num"] == self.window_count[model_type] and node["node_type"] == "model":
                # Adding the path to the self.model_path
                self.model_paths[model_type].append(path)
            else:
                forward_edges = edge_collection.find({"source": node["node_id"]})
                visited = set()
                for edge in forward_edges:
                    candidate_node = node_collection.find_one({"node_id": edge["destination"]})
                    if candidate_node["model_type"] == model_type:
                        if candidate_node["node_type"] == "model" and candidate_node["node_id"] not in visited:
                            visited.add(candidate_node["node_id"])
                            self.dfs(candidate_node, model_type, path.copy())
                        elif candidate_node["node_type"] == "intermediate":
                            # From an "intermediate" node there is only one outgoing edge to the "model" node of the same model_type
                            desc_edge_list = edge_collection.find({"source": candidate_node["node_id"]})
                            desc_nodes_id_list = [desc_edge["destination"] for desc_edge in desc_edge_list]
                            for desc_node_id in desc_nodes_id_list:
                                if desc_node_id not in visited:
                                    visited.add(desc_node_id)
                                    self.dfs(candidate_node, model_type, path.copy())

                if node["node_type"] == "model" and len(list(forward_edges)) == 0:
                    # Finding a node in the next window based on parametric compatibility
                    if self.config["model"][model_type]["post_synchronization_settings"]["aggregation_strategy"] == "average":
                        candidate_node_list = node_collection.find({"window_num": node["window_num"] + 1, "model_type": model_type,
                                                                    "node_type": "intermediate"})
                    else:
                        candidate_node_list = node_collection.find({"window_num": node["window_num"] + 1, "model_type": model_type,
                                                                    "node_type": "model"})
                    # Ranking the nodes based on parametric compatibility
                    job_collection = self.MONGO_CLIENT["ds_results"]["jobs"]
                    dsir_collection = self.MONGO_CLIENT["ds_results"]["dsir"]
                    dsir1 = dsir_collection.find_one({"_id": node["node_id"]})
                    job1 = job_collection.find_one({"_id": dsir1["metadata"]["job_id"]})
                    most_compatible_node = None
                    max_score = 0
                    for candidate_node in candidate_node_list:
                        job2 = job_collection.find_one({"output_dsir": candidate_node["node_id"]})
                        score = self.compute_compatibility(job1, job2)

                        if max_score <= score:
                            max_score = score
                            most_compatible_node = candidate_node

                    self.dfs(most_compatible_node, model_type, path.copy())
        except Exception as e:
            raise e

    def find_most_compatible_path(self, node, model_type, path):
        node_collection = self.GRAPH_CLIENT["model_graph"]["node"]
        edge_collection = self.GRAPH_CLIENT["model_graph"]["edge"]
        job_collection = self.MONGO_CLIENT["ds_results"]["jobs"]

        # Adding node_id to the path
        path.append(node["node_id"])

        # Adding the path to the self.model_path
        if node["window_num"] == self.window_count[model_type] and node["node_type"] == "model":
            self.model_paths[model_type].append(path)
        else:
            if node["node_type"] == "intermediate":
                # We have to add the model_node in the same window
                edge_list = edge_collection.find({"source": node["node_id"]})
                model_node_id_list = [edge["destination"] for edge in edge_list]
                path.extend(model_node_id_list)

            # Finding a arbitrary node in the next window
            if self.config["model"][model_type]["post_synchronization_settings"]["aggregation_strategy"] == "average":
                candidate_node_list = node_collection.find({"window_num": node["window_num"] + 1, "model_type": model_type,
                                                            "node_type": "intermediate"})
            else:
                candidate_node_list = node_collection.find({"window_num": node["window_num"] + 1, "model_type": model_type, "node_type": "model"})

            # Ranking the nodes based on parametric compatibility
            job1 = job_collection.find_one({"output_dsir": node["node_id"]})
            most_compatible_node = None
            max_score = 0
            for candidate_node in candidate_node_list:
                job2 = job_collection.find_one({"output_dsir": candidate_node["node_id"]})
                score = self.compute_compatibility(job1, job2)

                if max_score <= score:
                    max_score = score
                    most_compatible_node = candidate_node
            return self.find_most_compatible_path(most_compatible_node, model_type, path)

    def compute_compatibility(self, job1, job2):
        # TODO: Hardcoded here for bin, Need to check with Dr.Candan
        threshold = self.config["rules"]["bin"]
        match_counter = 0
        total_counter = len(job1["variables"])
        for key in job1["variables"]:
            parameter_1 = job1["variables"][key]
            parameter_2 = job2["variables"][key]
            if abs(parameter_1 - parameter_2) <= threshold[key]:
                match_counter = match_counter + 1
        return match_counter / total_counter

    def find_no_of_edges(self, path_1, path_2):
        no_of_edges = 0
        path_2_set = set(path_2)
        edge_collection = self.GRAPH_CLIENT["model_graph"]["edge"]
        for node_id in path_1:
            edge_list = edge_collection.find({"source": node_id})
            candidate_node_id_list = [edge["destination"] for edge in edge_list]
            for candidate_node_id in candidate_node_id_list:
                if candidate_node_id in path_2_set:
                    no_of_edges += 1
        return no_of_edges

    def find_model_path_score(self, path, model_type):
        job_collection = self.MONGO_CLIENT["ds_results"]["jobs"]
        score = 0
        # TODO: Need to integrate WM score
        for node_id in path:
            job = job_collection.find_one({"output_dsir": node_id})
            if job is not None:
                score += job["relevance"]

        normalized_score = score / self.window_count[model_type]
        return normalized_score
