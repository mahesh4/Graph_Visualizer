from bson.objectid import ObjectId
from collections import defaultdict, OrderedDict
from itertools import chain, combinations
import heapq
import itertools
import math
from app import utils


class Timelines:
    def __init__(self, mongo_client, graph_client, workflow_id):
        self.MONGO_CLIENT = mongo_client
        self.GRAPH_CLIENT = graph_client
        self.window_count = self.GRAPH_CLIENT["model_graph"]["workflows"].find_one({"workflow_id": ObjectId(workflow_id)})["window_count"]
        self.model_paths = {model: [] for model in self.window_count.keys()}
        self.timelines = []
        self.config = self.MONGO_CLIENT["ds_config"]["workflows"].find_one({"_id": ObjectId(workflow_id)})
        self.model_list = [model_config["name"] for model_config in self.config["model_settings"].values()]
        # self.model_dependency_list = {"hurricane": [], "flood": [0], "human_mobility": [0, 1]}
        self.model_dependency_list = {}
        for model_config in self.config["model_settings"].values():
            self.model_dependency_list[model_config["name"]] = [self.model_list.index(upstream_model) for upstream_model in model_config["upstream_models"].values()]
        self.workflow_id = workflow_id

    def remove_nodes_overlap(self, timelines_index_list):
        # TODO: Work in progress
        try:
            dsir_collection = self.MONGO_CLIENT["ds_results"]["dsir"]
            for timelines_index in timelines_index_list:
                index_list = timelines_index[2]
                for i in range((len(index_list))):
                    model_type = self.model_list[i]
                    model_path = self.model_paths[model_type][index_list[i][0]]
                    job_list = []
                    for node_id in model_path:
                        dsir = dsir_collection.find_one({"_id": node_id})
                        start = dsir["metadata"]["temporal"]["begin"]
                        end = dsir["metadata"]["temporal"]["end"]
                        weight = 0
                        # Calculating the no of edges the node has in the timeline
                        for j in self.model_dependency_list[model_type]:
                            up_model_type = self.model_list[j]
                            up_model_path = self.model_paths[up_model_type][index_list[j][0]]
                            weight += self.find_no_of_edges(up_model_path, model_path)
                        job_list.append({"start": start, "end": end, "_id": dsir["_id"], "weight": weight})
                    # Sorting the jobs based on end
                    job_list = sorted(job_list, key=lambda k: k["end"], reverse=True)
                    # Running Activity Scheduling Problem
                    profit_list = [job_list[0]["weight"]]
                    max_profit_job_list = [0]

                    for idx in range(len(job_list)):
                        comp_job_idx = None

                        # TODO: Need to use Binary search instead of linear search to speed up search
                        for comp_idx in reversed(range(idx)):
                            if job_list[comp_idx]["end"] <= job_list[idx]["start"]:
                                comp_job_idx = comp_idx
                                break

                        if comp_job_idx is not None:
                            if profit_list[comp_job_idx] + job_list[idx]["weight"] > profit_list[idx - 1]:
                                profit_list[idx] = profit_list[comp_job_idx] + job_list[idx]["weight"]
                                max_profit_job_list[idx] = comp_job_idx
                            else:
                                profit_list[idx] = profit_list[idx - 1]
                                max_profit_job_list[idx] = idx - 1

                    print(max_profit_job_list)
        except Exception as e:
            raise e

    def get_top_k_timelines(self, k):
        try:
            timelines_index_list = self.generate_top_k_timelines(k)
            timelines_list = self.format_timelines_to_output(timelines_index_list)
            return timelines_list
        except Exception as e:
            raise e

    def format_timelines_to_output(self, timelines_index_list):
        try:
            node_collection = self.GRAPH_CLIENT["model_graph"]["node"]
            edge_collection = self.GRAPH_CLIENT["model_graph"]["edge"]
            timelines_list = []

            for timelines_index in timelines_index_list:
                timeline = []
                score = -timelines_index[0]
                index_list = []
                for model in self.model_list:
                    for index, score, model_type in timelines_index[2]:
                        if model == model_type:
                            index_list.append(index)
                            break

                # Adding all the intermediate nodes to the stateful nodes path
                for i in range(len(self.model_list)):
                    model_info = utils.access_model_by_name(self.config, self.model_list[i])
                    if model_info["psm_settings"]["psm_strategy"] == "cluster":
                        model_path = self.model_paths[self.model_list[i]][index_list[i]]
                        new_model_path = []
                        for node_id in model_path:
                            node = node_collection.find_one({"node_id": node_id})
                            # We are not processing nor adding "intermediate" nodes present in the model_path directly, but we are adding them
                            # through adjacent_node_id_list
                            if node["node_type"] == "model":
                                # All the backward adjacent nodes are "intermediate" nodes
                                adjacent_node_id_list = map(lambda x: x["source"], edge_collection.find({"destination": node_id,
                                                                                                         "workflow_id": self.workflow_id}))
                                new_model_path.append(node_id)
                                new_model_path.extend(adjacent_node_id_list)
                        # Adding the new_model_path to the self.model_paths
                        self.model_paths[self.model_list[i]][index_list[i]] = new_model_path

                timeline_node_set = set.union(*map(set, [self.model_paths[self.model_list[i]][index_list[i]] for i in range(len(index_list))]))
                for i in range(len(index_list)):
                    model_type = self.model_list[i]
                    model_path = self.model_paths[model_type][index_list[i]]
                    for node_id in model_path:
                        node = {"name": model_type, "_id": str(node_id), "destination": [], "source": []}
                        adjacent_node_id_list = list(map(lambda x: x["destination"], edge_collection.find({"source": node_id,
                                                                                                           "workflow_id": self.workflow_id})))
                        upstream_node_id_list = list(map(lambda x: x["source"], edge_collection.find({"destination": node_id,
                                                                                                      "workflow_id": self.workflow_id})))

                        for adjacent_node_id in adjacent_node_id_list:
                            if adjacent_node_id in timeline_node_set:
                                node["destination"].append(str(adjacent_node_id))

                        for upstream_node_id in upstream_node_id_list:
                            if upstream_node_id in timeline_node_set:
                                node["source"].append(str(upstream_node_id))

                        timeline.append(node)
                        # End of for
                    # End of for
                # End of for
                timelines_list.append({"score": score, "links": timeline})
        except Exception as e:
            raise e
        return timelines_list

    def generate_top_k_timelines(self, k):
        try:
            node_collection = self.GRAPH_CLIENT['model_graph']['node']
            edge_collection = self.GRAPH_CLIENT["model_graph"]["edge"]
            stateful_models = [model_config["name"] for model_name, model_config in self.config["model_settings"].items() if model_config["stateful"]]
            stateless_models = [model_config["name"] for model_name, model_config in self.config["model_settings"].items() if not model_config["stateful"]]
            # Perform DFS on each "stateful" model_type
            for model_type in stateful_models:
                visited = set()
                model_info = utils.access_model_by_name(self.config, model_type)
                if model_info["psm_settings"]["psm_strategy"] == "average":
                    node_list = node_collection.find({"window_num": 1, "node_type": "intermediate", "model_type": model_type,
                                                      "workflow_id": self.workflow_id})
                    for node in node_list:
                        forward_edges = edge_collection.find({"source": node["node_id"], "workflow_id": self.workflow_id})
                        destination_list = [edge["destination"] for edge in forward_edges]
                        for destination_id in destination_list:
                            destination = node_collection.find_one({"node_id": destination_id, "workflow_id": self.workflow_id})
                            if destination["model_type"] == model_type and destination_id not in visited:
                                visited.add(destination_id)
                                self.dfs(node, model_type, [])
                else:
                    node_list = node_collection.find({"window_num": 1, "node_type": "model", "model_type": model_type,
                                                      "workflow_id": self.workflow_id})
                for node in node_list:
                    self.dfs(node, model_type, [])

            # Finding the most compatible paths on each "stateless" model_type
            for model_type in stateless_models:
                model_info = utils.access_model_by_name(self.config, model_type)
                if model_info["psm_settings"]["psm_strategy"] == "average":
                    node_list = node_collection.find({"window_num": 1, "node_type": "intermediate", "model_type": model_type,
                                                      "workflow_id": self.workflow_id})
                else:
                    node_list = node_collection.find({"window_num": 1, "node_type": "model", "model_type": model_type,
                                                      "workflow_id": self.workflow_id})

                for node in node_list:
                    self.find_most_compatible_path(node, model_type, [])

            # Getting the top K timelines index list
            timelines_index_list = self.run_nra(k)
            return timelines_index_list

        except Exception as e:
            raise e

    def run_nra(self, k):
        try:
            model_type_list = self.model_list
            # Generating the doc_list for NRA algorithm
            doc_list = []
            for model_type in self.model_paths:
                lst_score = []
                for i in range(len(self.model_paths[model_type])):
                    # Finding the score for the path
                    score = round(self.find_model_path_score(self.model_paths[model_type][i], model_type), 3)
                    lst_score.append({"index": i, "score": score, "model_type": model_type})
                doc_list.append(lst_score)

            doc_list = [sorted(lst, key=lambda k: k["score"], reverse=True) for lst in doc_list]
            max_doc_list_len = max([len(lst) for lst in doc_list])
            # Running NRA
            top_k_timelines = []
            heapq.heapify(top_k_timelines)
            no_of_models = len(model_type_list)
            for row_idx in range(max_doc_list_len):
                candidate_doc_list = [col[: row_idx + 1] if row_idx < len(col) else col[:] for col in doc_list]

                # Updating the up_score for all the rows in the top_k_timelines
                del_list = []
                for i, top_timeline in enumerate(top_k_timelines):
                    top_timeline_model_type_list = [model_type for index, score, model_type in top_timeline[2]]
                    new_up_score = 0
                    if len(top_timeline_model_type_list) == no_of_models:
                        continue
                    else:
                        del_flag = False
                        for j in range(no_of_models):
                            if model_type_list[j] not in top_timeline_model_type_list:
                                if row_idx < len(candidate_doc_list[j]):
                                    new_up_score += candidate_doc_list[j][row_idx]["score"]
                                else:
                                    del top_k_timelines[i]
                                    del_flag = True
                                    break
                            else:
                                index = top_timeline_model_type_list.index(model_type_list[j])
                                new_up_score += top_timeline[2][index][1]

                        if not del_flag:
                            top_timeline[1] = round(-new_up_score, 3)

                for col_idx in range(len(candidate_doc_list)):
                    iter_doc_list = list(candidate_doc_list)
                    iter_doc_list[col_idx] = [candidate_doc_list[col_idx][row_idx] if row_idx < len(candidate_doc_list[col_idx]) else []]

                    for idx in range(col_idx):
                        iter_doc_list[idx] = iter_doc_list[idx][: row_idx]

                    if len(iter_doc_list[col_idx][0]) == 0:
                        continue

                    for candidate_timeline in itertools.product(*iter_doc_list):
                        power_set = list(self.power_set(candidate_timeline))
                        power_set.reverse()
                        up_score = round(sum([x["score"] for x in candidate_timeline]), 3)
                        for candidate_timeline_subset in power_set:
                            # All subset of the candidate_timeline has been added to top_k_timelines
                            compatibility, low_score = self.check_timeline_compatibility(candidate_timeline_subset)
                            if compatibility:
                                top_k_flag = False
                                for i in range(len(top_k_timelines)):
                                    top_timeline = top_k_timelines[i]
                                    timeline = [(c_path["index"], c_path["score"], c_path["model_type"]) for c_path in candidate_timeline_subset]
                                    if self.check_set_equal(timeline, top_timeline[2]):
                                        top_k_flag = True
                                        break
                                # End of loop

                                if not top_k_flag:
                                    heapq.heappush(top_k_timelines, [-low_score, -up_score, [(c_path["index"], c_path["score"], c_path["model_type"])
                                                                                             for c_path in candidate_timeline_subset]])
                        # End of loop
                # End of loop

                heapq.heapify(top_k_timelines)

                # Criteria for ending the NRA early
                if k < len(top_k_timelines):
                    comp_timeline_count = 0
                    timelines = heapq.nsmallest(len(top_k_timelines), top_k_timelines)
                    for t_timeline in timelines[:k]:
                        if len(t_timeline[2]) == no_of_models:
                            comp_timeline_count += 1
                        else:
                            break

                    if comp_timeline_count == k:
                        terminate = True
                        for t_timeline in timelines[k:]:
                            if t_timeline[1] < timelines[k - 1][0]:
                                print("not terminating at ", row_idx, t_timeline, timelines[k - 1])
                                terminate = False
                                break
                        if terminate:
                            return timelines[:k]
        except Exception as e:
            raise e

        # No of timelines is less than k or there are some incompatible timelines in top_k_timelines
        compatible_timelines = []
        comp_timeline_count = k
        while top_k_timelines:
            t_timeline = heapq.heappop(top_k_timelines)
            if len(t_timeline[2]) == no_of_models:
                compatible_timelines.append(t_timeline)
                comp_timeline_count -= 1
                if comp_timeline_count == 0:
                    break
        return compatible_timelines

    def merge_subtimelines(self, timeline1, timeline2):
        """merge timeline1 with timeline2"""
        timeline1_model_set = set([t1_node["model"] for t1_node in timeline1])
        merged_timeline = timeline1
        for t2_node in timeline2:
            if t2_node["model"] not in timeline1_model_set:
                timeline1.append(t2_node)

    def check_subset_model(self, subset_list, set_list):
        for e in subset_list:
            if e not in set_list:
                return False
        return True

    def check_subset(self, subset_list, set_list):
        for subset_index, subset_model in subset_list:
            flag = False
            for set_index, set_model in set_list:
                if set_index == subset_index and set_model == subset_model:
                    flag = True
                    break
            if not flag:
                return False
        # End of loop
        return True

    def check_set_equal(self, subset_list, set_list):
        if len(subset_list) != len(set_list):
            return False
        for subset_index, subset_score, subset_model in subset_list:
            flag = False
            for set_index, set_score, set_model in set_list:
                if set_index == subset_index and set_model == subset_model:
                    flag = True
                    break
            # End of loop

            if not flag:
                return False
        # End of loop
        return True

    def check_timeline_compatibility(self, timeline):
        model_type_list = self.model_list
        score = sum([timeline[i]["score"] for i in range(len(timeline))])
        total_edges = 0
        for i in range(len(timeline)):
            model_type = timeline[i]["model_type"]
            model_path_index = timeline[i]["index"]
            for j in self.model_dependency_list[model_type]:
                up_model_type = model_type_list[j]
                up_model_path_index = None
                for t_model_path in timeline:
                    if t_model_path["model_type"] == up_model_type:
                        up_model_path_index = t_model_path["index"]
                        break

                if up_model_path_index is not None:
                    no_of_edges = self.find_no_of_edges(self.model_paths[up_model_type][up_model_path_index],
                                                        self.model_paths[model_type][model_path_index])
                    # if no_of_edges < math.floor(self.window_count[model_type] / 2):
                    #     return False, score
                    total_edges += no_of_edges
            # End of for
        # End of for
        if total_edges == 0:
            return False, 0
        else:
            return True, round(score, 3)

    def dfs(self, node, model_type, path):
        """Function to perform dfs. The dfs path consist of both "intermediate" and "model" nodes
        NOTE: We don't maintain visited list because the graph is acyclic. Cycles occur during clustering and there is a local visited set
        maintained"""
        try:
            node_collection = self.GRAPH_CLIENT["model_graph"]["node"]
            edge_collection = self.GRAPH_CLIENT["model_graph"]["edge"]
            model_info = utils.access_model_by_name(self.config, model_type)
            # Adding node_id to the path and visited
            path.append(node["node_id"])

            if node["window_num"] == self.window_count[model_type] and node["node_type"] == "model":
                # Adding the path to the self.model_path
                self.model_paths[model_type].append(path)
                return
            else:
                forward_edges = list(edge_collection.find({"source": node["node_id"], "workflow_id": self.workflow_id}))
                visited = set()
                for edge in forward_edges:
                    candidate_node = node_collection.find_one({"node_id": edge["destination"], "workflow_id": self.workflow_id})
                    if candidate_node["model_type"] == model_type:
                        if candidate_node["node_type"] == "model" and candidate_node["node_id"] not in visited:
                            visited.add(candidate_node["node_id"])
                            self.dfs(candidate_node, model_type, path.copy())
                        elif candidate_node["node_type"] == "intermediate":
                            # From an "intermediate" node there is only one outgoing edge to the "model" node of the same model_type
                            desc_edge_list = edge_collection.find({"source": candidate_node["node_id"], "workflow_id": self.workflow_id})
                            desc_nodes_id_list = [desc_edge["destination"] for desc_edge in desc_edge_list]
                            for desc_node_id in desc_nodes_id_list:
                                if desc_node_id not in visited:
                                    visited.add(desc_node_id)
                                    self.dfs(candidate_node, model_type, path.copy())

                if node["node_type"] == "model" and len(forward_edges) == 0:
                    # Finding a node in the next window based on parametric compatibility
                    if model_info["psm_settings"]["psm_strategy"] == "average":
                        candidate_node_list = node_collection.find({"window_num": node["window_num"] + 1, "model_type": model_type,
                                                                    "node_type": "intermediate", "workflow_id": self.workflow_id})
                    else:
                        candidate_node_list = node_collection.find({"window_num": node["window_num"] + 1, "model_type": model_type,
                                                                    "node_type": "model", "workflow_id": self.workflow_id})
                    # Ranking the nodes based on parametric compatibility
                    job_collection = self.MONGO_CLIENT["ds_results"]["jobs"]
                    dsir_collection = self.MONGO_CLIENT["ds_results"]["dsir"]
                    dsir1 = dsir_collection.find_one({"_id": node["node_id"], "workflow_id": self.workflow_id})
                    job1 = job_collection.find_one({"_id": dsir1["metadata"]["job_id"], "workflow_id": self.workflow_id})
                    most_compatible_node = None
                    max_score = 0
                    for candidate_node in candidate_node_list:
                        job2 = job_collection.find_one({"output_dsir": candidate_node["node_id"], "workflow_id": self.workflow_id})
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
        model_info = utils.access_model_by_name(self.config, model_type)
        # Adding node_id to the path
        path.append(node["node_id"])

        # Adding the path to the self.model_path
        if node["window_num"] == self.window_count[model_type] and node["node_type"] == "model":
            self.model_paths[model_type].append(path)
        else:
            if node["node_type"] == "intermediate":
                # We have to add the model_node in the same window
                edge_list = edge_collection.find({"source": node["node_id"], "workflow_id": self.workflow_id})
                model_node_id_list = [edge["destination"] for edge in edge_list]
                path.extend(model_node_id_list)

            # Finding a arbitrary node in the next window
            if model_info["psm_settings"]["psm_strategy"] == "average":
                candidate_node_list = node_collection.find({"window_num": node["window_num"] + 1, "model_type": model_type,
                                                            "node_type": "intermediate", "workflow_id": self.workflow_id})
            else:
                candidate_node_list = node_collection.find({"window_num": node["window_num"] + 1, "model_type": model_type, "node_type": "model",
                                                            "workflow_id": self.workflow_id})

            # Ranking the nodes based on parametric compatibility
            job1 = job_collection.find_one({"output_dsir": node["node_id"], "workflow_id": self.workflow_id})
            most_compatible_node = None
            max_score = 0
            for candidate_node in candidate_node_list:
                job2 = job_collection.find_one({"output_dsir": candidate_node["node_id"], "workflow_id": self.workflow_id})
                score = self.compute_compatibility(job1, job2)

                if max_score <= score:
                    max_score = score
                    most_compatible_node = candidate_node
            return self.find_most_compatible_path(most_compatible_node, model_type, path)

    def compute_compatibility(self, job1, job2):
        threshold = {var_config["name"]: float(var_config["bin"]) for model_config in self.config["model_settings"].values() for var_id, var_config in model_config["sampled_variables"].items()}
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
            edge_list = edge_collection.find({"source": node_id, "workflow_id": self.workflow_id})
            candidate_node_id_list = [edge["destination"] for edge in edge_list]
            for candidate_node_id in candidate_node_id_list:
                if candidate_node_id in path_2_set:
                    no_of_edges += 1
        return no_of_edges

    def find_model_path_score(self, path, model_type):
        """This is intra-actor compatibility. Score is calculated as the sum of relevance scores"""
        job_collection = self.MONGO_CLIENT["ds_results"]["jobs"]
        score = 0
        # TODO: Need to integrate WM score
        for node_id in path:
            job = job_collection.find_one({"output_dsir": node_id, "workflow_id": self.workflow_id})
            if job is not None:
                score += job["relevance"]

        normalized_score = score / self.window_count[model_type]
        return normalized_score

    def power_set(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))

    def generate_timelines(self):
        """Function to generate all the model paths in the complete execution context"""
        try:
            node_collection = self.GRAPH_CLIENT["model_graph"]["node"]
            edge_collection = self.GRAPH_CLIENT["model_graph"]["edge"]
            stateful_models = [model_config["name"] for model_id, model_config in self.config["model_settings"].items() if model_config["stateful"]]
            stateless_models = [model_config["name"] for model_id, model_config in self.config["model_settings"].items() if not model_config["stateful"]]

            # Perform DFS on each "stateful" model_type
            for model_type in stateful_models:
                visited = set()
                model_info = utils.access_model_by_name(self.config, model_type)
                if model_info["psm_settings"]["psm_strategy"] == "average":
                    node_list = node_collection.find({"window_num": 1, "node_type": "intermediate", "model_type": model_type,
                                                      "workflow_id": self.workflow_id})
                    for node in node_list:
                        forward_edges = edge_collection.find({"source": node["node_id"], "workflow_id": self.workflow_id})
                        destination_list = [edge["destination"] for edge in forward_edges]
                        for destination_id in destination_list:
                            destination = node_collection.find_one({"node_id": destination_id, "workflow_id": self.workflow_id})
                            if destination["model_type"] == model_type and destination_id not in visited:
                                visited.add(destination_id)
                                self.dfs(node, model_type, [])
                else:
                    node_list = node_collection.find({"window_num": 1, "node_type": "model", "model_type": model_type,
                                                      "workflow_id": self.workflow_id})
                for node in node_list:
                    self.dfs(node, model_type, [])

            # Finding the most compatible paths on each "stateless" model_type
            for model_type in stateless_models:
                model_info = utils.access_model_by_name(self.config, model_type)
                if model_info["psm_settings"]["psm_strategy"] == "average":
                    node_list = node_collection.find({"window_num": 1, "node_type": "intermediate", "model_type": model_type,
                                                      "workflow_id": self.workflow_id})
                else:
                    node_list = node_collection.find({"window_num": 1, "node_type": "model", "model_type": model_type,
                                                      "workflow_id": self.workflow_id})

                for node in node_list:
                    self.find_most_compatible_path(node, model_type, [])

        except Exception as e:
            raise e

    def get_top_k_dsir_timelines(self, dsir_id, k):
        """Function to get the top K timelines for a particular dsir_id"""
        try:
            dsir_collection = self.MONGO_CLIENT["ds_results"]["dsir"]
            dsir = dsir_collection.find_one({"_id": dsir_id, "workflow_id": self.workflow_id})
            model_type = dsir["metadata"]["model_type"]
            self.generate_timelines()

            # Stripping the model_paths for the particular model_type which doesn't consist of the dsir_id
            new_model_path = []
            for index, model_path in enumerate(self.model_paths[model_type]):
                if dsir_id in model_path:
                    new_model_path.append(model_path)

            self.model_paths[model_type] = new_model_path
            # Getting the top K timelines
            timelines_index_list = self.run_nra(k)
            timelines_list = self.format_timelines_to_output(timelines_index_list)
            return timelines_list
        except Exception as e:
            raise e

    def get_top_k_timelines_test(self, K):
        try:
            doc_list = []
            self.generate_timelines()
            for model_type in self.model_paths:
                lst_score = []
                for i in range(len(self.model_paths[model_type])):
                    # Finding the score for the path
                    score = round(self.find_model_path_score(self.model_paths[model_type][i], model_type), 3)
                    lst_score.append({"index": i, "score": score, "model_type": model_type})
                doc_list.append(lst_score)

            doc_list = [sorted(lst, key=lambda k: k["score"], reverse=True) for lst in doc_list]
            top_k_timelines = []
            top_k_score = 0

            for candidate_timeline in itertools.product(*doc_list):
                compatibility, low_score = self.check_timeline_compatibility(candidate_timeline)
                if compatibility:
                    score = sum([model_path["score"] for model_path in candidate_timeline])
                    if len(top_k_timelines) < K:
                        top_k_timelines.append([score, [(model_path["index"], model_path["model_type"]) for model_path in candidate_timeline]])
                        top_k_timelines = sorted(top_k_timelines, key=lambda k: k[0], reverse=True)
                        top_k_score = top_k_timelines[-1][0]
                    elif score > top_k_score:
                        del top_k_timelines[-1]
                        top_k_timelines.append([score, [(model_path["index"], model_path["model_type"]) for model_path in candidate_timeline]])
                        top_k_timelines = sorted(top_k_timelines, key=lambda k: k[0], reverse=True)
                        top_k_score = top_k_timelines[-1][0]

            top_k_timelines = sorted(top_k_timelines, key=lambda k: k[0], reverse=True)

        except Exception as e:
            raise e
