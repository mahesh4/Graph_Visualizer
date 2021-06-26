import traceback

import bson
import numpy
from bson.objectid import ObjectId
from collections import defaultdict
from itertools import chain, combinations
import pathlib
import sys
from datetime import datetime
import math
import itertools

MODULES_PATH = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.append(str(MODULES_PATH))
from app.timeline_visualizer import utils
from app.timeline_visualizer.function_H import parametric_similarity
from app.timeline_visualizer.function_H import provenance_criteria


class Timelines:
    def __init__(self, mongo_client, workflow_id):
        self.MONGO_CLIENT = mongo_client
        self.window_count = self.MONGO_CLIENT["model_graph"]["workflows"].find_one({"workflow_id": ObjectId(workflow_id)})["window_count"]
        self.model_paths = {model: [] for model in self.window_count.keys()}
        self.timelines = []
        self.start_time = datetime.now()
        self.end_time = None
        self.config = self.MONGO_CLIENT["ds_config"]["workflows"].find_one({"_id": ObjectId(workflow_id)})
        self.model_list = list(self.model_paths.keys())
        self.model_dependency_list = {}
        for model_config in self.config["model_settings"].values():
            self.model_dependency_list[model_config["name"]] = []
            if "upstream_models" in model_config:
                self.model_dependency_list[model_config["name"]] = [self.model_list.index(upstream_model) for upstream_model in
                                                                    model_config["upstream_models"].values()]
        # End of loop
        self.workflow_id = workflow_id
        self.DSIR_DB = self.MONGO_CLIENT["ds_results"]["dsir"]
        self.JOB_DB = self.MONGO_CLIENT["ds_results"]["jobs"]
        self.DS_CONFIG = self.MONGO_CLIENT["ds_config"]["collection"].find_one({})

    def format_timelines_to_output(self, top_k_timelines_list):
        try:
            node_collection = self.MONGO_CLIENT["model_graph"]["node"]
            edge_collection = self.MONGO_CLIENT["model_graph"]["edge"]
            timelines_list = []

            for top_timeline in top_k_timelines_list:
                timeline = []
                score = top_timeline["causal_edges"]

                # Adding all the intermediate nodes to the stateful nodes path
                for model in self.model_list:
                    model_info = utils.access_model_by_name(self.config, model)
                    if model_info["psm_settings"]["psm_strategy"] == "cluster":
                        model_path = top_timeline["model_paths"][model]["instance_list"]
                        new_model_path = []
                        for node_id in model_path:
                            node = node_collection.find_one({"node_id": ObjectId(node_id)})
                            # We are not processing nor adding "intermediate" nodes present in the model_path directly, but we are adding them
                            # through adjacent_node_id_list
                            if node["node_type"] == "model":
                                # All the backward adjacent nodes are "intermediate" nodes
                                adjacent_node_id_list = map(lambda x: x["source"], edge_collection.find({"destination": ObjectId(node_id),
                                                                                                         "workflow_id": self.workflow_id}))
                                new_model_path.append(node_id)
                                new_model_path.extend(adjacent_node_id_list)
                        # Adding the new_model_path to the self.model_paths
                        top_timeline["model_paths"][model]["instance_list"] = new_model_path

                timeline_node_set = set.union(*map(set, [top_timeline["model_paths"][i]["instance_list"] for i in self.model_list]))
                for model_type in self.model_list:
                    model_path = top_timeline["model_paths"][model_type]["instance_list"]
                    for node_id in model_path:
                        node = {"name": model_type, "_id": str(node_id), "destination": [], "source": []}
                        adjacent_node_id_list = list(map(lambda x: x["destination"], edge_collection.find({"source": ObjectId(node_id),
                                                                                                           "workflow_id": self.workflow_id})))
                        upstream_node_id_list = list(map(lambda x: x["source"], edge_collection.find({"destination": ObjectId(node_id),
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

    def check_causal_edge(self, parent_dsir, dsir):
        provenenace = self.MONGO_CLIENT["ds_provenance"]["provenance"].find_one({"dsir_id": dsir["_id"]})
        parent_dsir_end = str(int(parent_dsir["metadata"]["temporal"]["end"]))
        model = parent_dsir["metadata"]["model_type"]
        score = 0
        if parent_dsir_end in provenenace["provenance"][model] and parent_dsir["_id"] in provenenace["provenance"][model][parent_dsir_end]:
            score = 1
        return score

    def function_H(self, dsir1, dsir2):
        """
        @param
            dsir1 (dict): dsir
            dsir2 (dict): dsir
        @return:
        """
        # TODO: To add spatial_compatibility_score
        parametric_similarity_score = self.compute_parametric_similarity([dsir1["_id"], dsir2["_id"]])
        temporal_context = {"end": dsir2["metadata"]["temporal"]["end"]}
        provenance_similarity_score = provenance_criteria.find_provenance_similarity(dsir1["_id"], dsir2["_id"], temporal_context, self.MONGO_CLIENT)
        causal_edge = self.check_causal_edge(dsir1, dsir2)
        return numpy.mean([parametric_similarity_score, provenance_similarity_score, causal_edge])

    def compute_parametric_similarity(self, dsir_id_list):
        """
        Returns parametric similarity score of the windowing_set
        @param dsir_id_list: A list of dsir_ids
        @return score: parametric similarity score
        """
        if self.DS_CONFIG["compatibility_settings"]["parametric_mode"] == "conjunction":
            parametric_compatibility_function = parametric_similarity.conjunction
        elif self.DS_CONFIG["compatibility_settings"]["parametric_mode"] == "union":
            parametric_compatibility_function = parametric_similarity.union
        else:
            raise Exception("Illegal parametric compatibility")

        parameter_dict = defaultdict(list)
        for dsir_id in dsir_id_list:
            dsir = self.DSIR_DB.find_one({"_id": dsir_id})
            if dsir["created_by"] == "PostSynchronizationManager":
                parent_dsir_list = dsir["parents"]
                jobs_list = list(self.JOB_DB.find({"output_dsir": {"$in": parent_dsir_list}}))
            else:
                jobs_list = list([self.JOB_DB.find_one({"output_dsir": dsir_id})])

            for job in jobs_list:
                for variable, value in job["variables"].items():
                    parameter_dict[variable].append(value)
                # End of loop
        # End of loop

        # computing parametric similarity score
        compatibility, score = parametric_compatibility_function(parameter_dict, self.DS_CONFIG)
        return score

    def construct_model_graph(self, model_type):
        """
        Function to build the model_graph for a particular model. The model-graph is a directed graph, where each entry in the adj_dict is a list 2
        elements which are destination node and edge_weight respectively.
        Note: the terminal nodes won't be present in adj_dict, since they don't have any adjacent nodes in the successive windows
        @param model_type: The current_model on which the model_graph has to be constructed
        @rtype: adj_dict: Adjacency matrix
        """
        adj_dict = defaultdict(list)
        node_collection = self.MONGO_CLIENT["model_graph"]["node"]
        dsir_collection = self.MONGO_CLIENT["ds_results"]["dsir"]
        prev_nodes = node_collection.find({"node_type": "model", "model_type": model_type, "workflow_id": self.workflow_id, "window_num": 1})
        prev_node_list_dsir = list(dsir_collection.find({"_id": {"$in": [p_node["node_id"] for p_node in prev_nodes]}}))
        for prev_node_dsir in prev_node_list_dsir:
            adj_dict[str(prev_node_dsir["_id"])] = []

        window_num = 2
        while window_num <= self.window_count[model_type]:
            window_nodes = node_collection.find({"node_type": "model", "model_type": model_type, "workflow_id": self.workflow_id,
                                                 "window_num": window_num})
            window_num += 1
            # BaseCase
            curr_node_list_dsir = list()
            for w_node in window_nodes:
                w_node_dsir = dsir_collection.find_one({"_id": w_node["node_id"]})
                curr_node_list_dsir.append(w_node_dsir)
                for p_node_dsir in prev_node_list_dsir:
                    delta = self.function_H(p_node_dsir, w_node_dsir)
                    # There exist an edge between p_node_dsir and w_node_dsir only when delta > 0
                    if delta > 0:
                        adj_dict[str(p_node_dsir["_id"])].append([str(w_node_dsir["_id"]), delta])
                # End of loop
            # End of loop
            prev_node_list_dsir = curr_node_list_dsir
        # End of loop
        return adj_dict

    def generate_model_paths(self, model_type, penalty, max_model_path):
        """
        Function to generate model_paths
        @param model_type: The current_model on which the model_paths has to be extracted
        @param penalty:
        @param min_score:
        @return: model_paths: A list of model_paths
        """
        node_collection = self.MONGO_CLIENT["model_graph"]["node"]
        model_paths = list()
        unique_idx = 0
        # Base Case
        if self.window_count[model_type] == 1:
            terminal_node_list = node_collection.find({"node_type": "model", "model_type": model_type, "workflow_id": self.workflow_id,
                                                       "window_num": 1})
            for terminal_node in terminal_node_list:
                # Since they don't have any internal edges, we give score as 0
                # inserting model-paths
                self.MONGO_CLIENT["model_graph"]["model_paths"].insert_one(
                    {"workflow_id": self.workflow_id, "model_path": [str(terminal_node["node_id"])],
                     "model_type": model_type, "internal_causal_edges": 0, "unique_idx": unique_idx})
                unique_idx += 1
                if unique_idx == max_model_path:
                    break
            # End of loop
            return model_paths

        # build the adjacent matrix
        adj_dict = self.construct_model_graph(model_type)
        visited_nodes = set()
        while True:
            window_num = 1
            dp_dict = defaultdict(int)
            path_dict = dict()
            # since terminal nodes are not present in adj_dict, we iterate till (self.window_count[model_type] - 1) only
            while window_num < self.window_count[model_type]:
                candidate_node_list = node_collection.find({"node_type": "model", "model_type": model_type, "workflow_id": self.workflow_id,
                                                            "window_num": window_num})
                for candidate_node in candidate_node_list:
                    candidate_node_id = str(candidate_node["node_id"])
                    for adj_node, weight in adj_dict[candidate_node_id]:
                        if dp_dict[adj_node] < dp_dict[candidate_node_id] + weight:
                            if window_num == 1:
                                if candidate_node_id in visited_nodes:
                                    if dp_dict[candidate_node_id] + weight - penalty > 0:
                                        dp_dict[adj_node] = dp_dict[candidate_node_id] + weight - penalty
                                        path_dict[adj_node] = [candidate_node_id, adj_node]
                                else:
                                    dp_dict[adj_node] = dp_dict[candidate_node_id] + weight
                                    path_dict[adj_node] = [candidate_node_id, adj_node]
                            elif candidate_node_id in path_dict:
                                path_dict[adj_node] = list(path_dict[candidate_node_id])
                                path_dict[adj_node].append(adj_node)
                                dp_dict[adj_node] = dp_dict[candidate_node_id] + weight
                    # End of loop
                # End of loop
                window_num += 1
            # End of loop

            terminal_node_list = node_collection.find({"node_type": "model", "model_type": model_type, "workflow_id": self.workflow_id,
                                                       "window_num": self.window_count[model_type]})
            max_score = 0
            max_path = list()
            for terminal_node in terminal_node_list:
                if dp_dict[str(terminal_node["node_id"])] > max_score:
                    max_score = dp_dict[str(terminal_node["node_id"])]
                    max_path = path_dict[str(terminal_node["node_id"])]
            # End of loop

            if len(max_path) == 0:
                break

            if len([1 for _id in max_path if _id in visited_nodes]) < len(max_path):
                # Computing internal-causal edges
                causal_edges_count = self.compute_internal_causal_edges(max_path, model_type)
                # inserting model-paths
                self.MONGO_CLIENT["model_graph"]["model_paths"].insert_one(
                    {"workflow_id": self.workflow_id, "model_path": max_path, "model_type": model_type,
                     "internal_causal_edges": causal_edges_count, "unique_idx": unique_idx})
                unique_idx += 1
                if unique_idx == max_model_path:
                    break

            visited_nodes.update(max_path)

            # Introducing penalties on the visited edges
            for path_node in max_path:
                if path_node in adj_dict:
                    del_list = list()
                    for idx, edge in enumerate(adj_dict[path_node]):
                        adj_dsir_id, delta = edge
                        if adj_dsir_id in max_path:
                            adj_dict[path_node][idx][1] = max(0, delta - penalty)
                            if adj_dict[path_node][idx][1] == 0:
                                del_list.append(idx)
                    # End of loop
                    for del_idx in reversed(sorted(del_list)):
                        del adj_dict[path_node][del_idx]
                    # End of loop
        # End of loop

    def compute_internal_causal_edges(self, max_path, model):
        """
        Function to compute the internal-edges of the max_path
        @param max_path: A model_path with a list of dsir_ids.
        Note: The instances in the model_paths are ordered in the increasing order of the windows
        @param model: The model_type of the max_path
        """
        internal_causal_edges_count = 0
        psm_strategy = utils.access_model_by_name(self.config, model)["psm_settings"]["psm_strategy"]
        # The last-window nodes won't have any outgoing causal edges
        for dsir_id in max_path[:-1]:
            node = self.MONGO_CLIENT["model_graph"]["node"].find_one({"node_id": ObjectId(dsir_id)})
            forward_edges = node["source"]
            forward_nodes = list(self.MONGO_CLIENT["model_graph"]["node"].find({"destination": {"$in": forward_edges},
                                                                                "model_type": model}))
            if psm_strategy == "cluster":
                descendant_edges = [edge for fr_node in forward_nodes for edge in fr_node["source"]]
                descendant_nodes = list(self.MONGO_CLIENT["model_graph"]["node"].find({"destination": {"$in": descendant_edges},
                                                                                       "model_type": model}))
                for ds_node in descendant_nodes:
                    if str(ds_node["node_id"]) in max_path:
                        denom = len(
                            list(self.MONGO_CLIENT["model_graph"]["node"].find({"source": {"$in": ds_node["destination"]}, "model_type": model})))
                        internal_causal_edges_count += len(set(ds_node["destination"]).intersection(set(descendant_edges))) / denom
            else:
                for fr_node in forward_nodes:
                    if str(fr_node["node_id"]) in max_path:
                        internal_causal_edges_count += 1
        return internal_causal_edges_count

    def causal_edges_between_model_paths(self, model_path1_info, model_path2_info):
        """
        Function to compute external causal edges between two model_paths.

        @param model_path1_info: A dictionary consisting of three fields
            model_path: A list of simulation instances which are part of the model_path
            model_type: The model_type of the model_path

        @param model_path2_info: A dictionary consisting of three fields
            model_path: A list of simulation instances which are part of the model_path
            model_type: The model_type of the model_path
        """
        model_path_1 = model_path1_info["model_path"]
        # This belongs to upstream model
        model_path_2 = model_path2_info["model_path"]
        causal_edges = 0
        psm_strategy = utils.access_model_by_name(self.config, model_path1_info["model_type"])["psm_settings"]["psm_strategy"]
        for dsir_id in model_path_2:
            node = self.MONGO_CLIENT["model_graph"]["node"].find_one({"node_id": ObjectId(dsir_id)})
            forward_edges = node["source"]
            forward_nodes = self.MONGO_CLIENT["model_graph"]["node"].find(
                {"destination": {"$in": forward_edges}, "model_type": model_path1_info["model_type"]})
            if psm_strategy == "cluster":
                descendant_edges = [edge for fr_node in forward_nodes for edge in fr_node["source"]]
                descendant_nodes = list(self.MONGO_CLIENT["model_graph"]["node"].find({"destination": {"$in": descendant_edges},
                                                                                       "model_type": model_path1_info["model_type"]}))
                for ds_node in descendant_nodes:
                    if str(ds_node["node_id"]) in model_path_1:
                        intermedidate_nodes = list(self.MONGO_CLIENT["model_graph"]["node"].find({"source": {"$in": ds_node["destination"]}}))
                        up_edges = [up_edge for it_node in intermedidate_nodes for up_edge in it_node["destination"]]
                        denom = len(list(self.MONGO_CLIENT["model_graph"]["node"].find({'source': {"$in": up_edges},
                                                                                        "model_type": model_path2_info["model_type"]})))
                        causal_edges += len(set(ds_node["destination"]).intersection(set(descendant_edges))) / denom
            else:
                for fr_node in forward_nodes:
                    if str(fr_node["node_id"]) in model_path_1:
                        # denom = len([1 for e in fr_node["destination"] if
                        #              self.MONGO_CLIENT["model_graph"]["node"].find_one({"source": {"$in": [e]},
                        #                                                                 "model_type": model_path2_info["model_type"]}) is not None])
                        # causal_edges += len(set(fr_node["destination"]).intersection(set(forward_edges))) / denom
                        # causal_edges += len(set(fr_node["destination"]).intersection(set(forward_edges)))
                        causal_edges += 1
        return causal_edges

    def join_causal_pair(self, model_1, model_2):
        """"""
        model_paths_1 = list(self.MONGO_CLIENT["model_graph"]["model_paths"].find({"model_type": model_1}))
        model_paths_2 = list(self.MONGO_CLIENT["model_graph"]["model_paths"].find({"model_type": model_2}))
        for model_path_1 in model_paths_1:
            for model_path_2 in model_paths_2:
                external_causal_edges = self.causal_edges_between_model_paths(model_path_2, model_path_1)
                self.MONGO_CLIENT["model_graph"]["causal_pairs"].insert({
                    model_path_1["model_type"]: {
                        "_id": model_path_1["_id"]
                    },
                    model_path_2["model_type"]: {
                        "_id": model_path_2["_id"]
                    },
                    "external_causal_edges": external_causal_edges
                })

    def generate_model_paths_all(self, penalty, max_model_path):
        for model_type in self.model_list:
            utils.set_model(model_type)
            self.generate_model_paths(model_type, penalty, max_model_path)

    def generate_timelines_via_A_star_topology(self, K, diversity):
        """"""
        self.start_time = datetime.now()
        # Generating a topological_order
        level_ordered_models = self.MONGO_CLIENT["ds_state"]["runtime"].find_one({})["level_order"]
        joins = 0
        no_of_models = len(level_ordered_models)
        timelines_scores = list()
        count = 0
        total_causal_pairs = 0
        visited_dict = defaultdict(bool)
        model_visited_dict = defaultdict(int)
        cache = defaultdict(dict)
        # We want a timeline can't have more #diversity re-used model-paths
        model_path_count = defaultdict(int)
        top_k_timelines = list()
        for model in level_ordered_models:
            model_path_count[model] = len(list(self.MONGO_CLIENT["model_graph"]["model_paths"].find({"model_type": model})))

        window_count = self.MONGO_CLIENT["model_graph"]["workflows"].find_one({"workflow_id": self.workflow_id})["window_count"]
        max_external_causal_edges = 0
        for model_id, model_config in self.DS_CONFIG["model_settings"].items():
            if "downstream_models" in model_config:
                total_causal_pairs += len(model_config["downstream_models"])
                for down_stream_model in model_config["downstream_models"].values():
                    max_external_causal_edges += window_count[down_stream_model]

        while count < K:
            # Finding the highest ranked sub_timeline
            candidate_sub_timeline = list(
                self.MONGO_CLIENT["model_graph"]["timelines"].find({"no_of_models": {"$lt": no_of_models}}).sort(
                    [("heuristic_score", -1), ("no_of_models", -1)]).limit(1))
            extended_sub_timelines = list()
            if not candidate_sub_timeline:
                if count > 0:
                    break
                # creating a new sub_timeline
                for model_path in self.MONGO_CLIENT["model_graph"]["model_paths"].find({"model_type": level_ordered_models[0]}):
                    extended_sub_timelines.append({
                        "causal_edges": model_path["internal_causal_edges"],
                        "reused": 0,
                        "internal_causal_edges": model_path["internal_causal_edges"],
                        "external_causal_edges": 0,
                        "heuristic_score": model_path["internal_causal_edges"] * no_of_models + max_external_causal_edges,
                        "causal_pairs": 0,
                        "no_of_models": 1,
                        "total_joins": 0,
                        "model_paths": {
                            level_ordered_models[0]: {
                                "instance_list": model_path["model_path"],
                                "_id": model_path["_id"]
                            }
                        }})
                # End of loop
                self.MONGO_CLIENT["model_graph"]["timelines"].insert(extended_sub_timelines)
            else:
                candidate_sub_timeline = candidate_sub_timeline[0]
                next_model_idx = candidate_sub_timeline["no_of_models"]
                next_model = level_ordered_models[next_model_idx]
                model_config = utils.access_model_by_name(self.config, next_model)

                # Updating causal_pairs
                if "upstream_models" in model_config:
                    candidate_sub_timeline["causal_pairs"] += len(model_config["upstream_models"])

                n = candidate_sub_timeline["no_of_models"]
                c = 0
                for idx, model in enumerate(reversed(level_ordered_models[n:])):
                    if model_visited_dict[model] == model_path_count[model]:
                        c += 1

                for model_path in self.MONGO_CLIENT["model_graph"]["model_paths"].find({"model_type": next_model}):
                    reused = candidate_sub_timeline["reused"]
                    # A timeline can't have more than #diversity visited model-paths
                    if visited_dict[str(model_path["_id"])]:
                        if candidate_sub_timeline["reused"] >= diversity - c:
                            continue
                        else:
                            reused += 1

                    joins += 1
                    internal_causal_edges = candidate_sub_timeline["internal_causal_edges"] + model_path["internal_causal_edges"]
                    external_causal_edges = candidate_sub_timeline["external_causal_edges"]
                    # Adding the external causal edges
                    if "upstream_models" in model_config:
                        for upstream_model_type in model_config["upstream_models"].values():
                            key1 = str(candidate_sub_timeline["model_paths"][upstream_model_type]["_id"])
                            key2 = str(model_path["_id"])
                            if key1 not in cache or key2 not in cache[key1]:
                                upstream_model_path = candidate_sub_timeline["model_paths"][upstream_model_type]["instance_list"]
                                cache[key1][key2] = self.causal_edges_between_model_paths(model_path, {"model_path": upstream_model_path,
                                                                                                       "model_type": upstream_model_type})
                            external_causal_edges += cache[key1][key2]
                        # End of loop
                    # Extending the candidate_timeline
                    causal_edges = internal_causal_edges + external_causal_edges
                    internal_causal_edges_avg = math.ceil(internal_causal_edges / (next_model_idx + 1))
                    heuristic_score = causal_edges
                    if candidate_sub_timeline["causal_pairs"] > 0:
                        external_causal_edges_avg = math.ceil(external_causal_edges / candidate_sub_timeline["causal_pairs"])
                        heuristic_score += internal_causal_edges_avg * (no_of_models - (next_model_idx + 1)) + \
                                           external_causal_edges_avg * (total_causal_pairs - candidate_sub_timeline["causal_pairs"])
                        # End of loop
                    else:
                        heuristic_score += max_external_causal_edges + internal_causal_edges_avg * len(level_ordered_models[next_model_idx + 1:])

                    new_candidate_sub_timeline = dict({
                        "_id": bson.objectid.ObjectId(),
                        "causal_edges": causal_edges,
                        "model_paths": dict(candidate_sub_timeline["model_paths"]),
                        "internal_causal_edges": internal_causal_edges,
                        "external_causal_edges": external_causal_edges,
                        "causal_pairs": candidate_sub_timeline["causal_pairs"],
                        "heuristic_score": heuristic_score,
                        "reused": reused,
                        "no_of_models": next_model_idx + 1})
                    new_candidate_sub_timeline["model_paths"][next_model] = dict(
                        {"instance_list": model_path["model_path"], "_id": model_path["_id"]})
                    extended_sub_timelines.append(new_candidate_sub_timeline)
                # End of loop

                self.MONGO_CLIENT["model_graph"]["timelines"].delete_one({"_id": candidate_sub_timeline["_id"]})

                if next_model_idx + 1 == no_of_models and len(extended_sub_timelines) > 0:
                    if diversity < no_of_models:
                        # retaining the top-timeline
                        top_extended_timeline = {"heuristic_score": -1, "model_paths": dict()}
                        for timeline in extended_sub_timelines:
                            if timeline["heuristic_score"] > top_extended_timeline["heuristic_score"]:
                                top_extended_timeline = timeline
                        # End of loop

                        top_k_timelines.append(top_extended_timeline)
                        top_extended_timeline["insert_time"] = datetime.now()
                        for model_type, model_path in top_extended_timeline["model_paths"].items():
                            if not visited_dict[str(model_path["_id"])]:
                                visited_dict[str(model_path["_id"])] = True
                                model_visited_dict[model_type] += 1
                                self.MONGO_CLIENT["model_graph"]["timelines"].update_many({"model_paths." + model_type + "._id": model_path["_id"],
                                                                                           "no_of_models": {"$lt": no_of_models}},
                                                                                          {"$inc": {"reused": 1}})
                        # End of loop

                        timelines_scores.append(top_extended_timeline["causal_edges"])
                        count += 1

                        # Removing, where reused is greater than diversity
                        self.MONGO_CLIENT["model_graph"]["timelines"].remove({"reused": {"$gt": diversity}, "no_of_models": {"$lt": no_of_models}})

                        # cleaning up-subtrees which can't grow further
                        max_reused = diversity
                        for idx, model in enumerate(reversed(level_ordered_models)):
                            # Check if the model has all model_paths visited
                            if model_visited_dict[model] == model_path_count[model]:
                                self.MONGO_CLIENT["model_graph"]["timelines"].remove({"reused": {"$gt": max_reused - 1}, "no_of_models": {"$lte": idx}})
                                max_reused -= 1

                        top_extended_timeline["workflow_id"] = self.workflow_id
                        top_extended_timeline["total_joins"] = joins
                        self.MONGO_CLIENT["model_graph"]["timelines"].insert(top_extended_timeline)
                    else:
                        # retaining all the timelines
                        top_k_timelines.extend(extended_sub_timelines)
                        for top_timeline in extended_sub_timelines:
                            top_timeline["insert_time"] = datetime.now()
                            top_timeline["workflow_id"] = self.workflow_id
                            top_timeline["total_joins"] = joins
                            self.MONGO_CLIENT["model_graph"]["timelines"].insert(top_timeline)
                            count += 1
                        # End of loop
                elif len(extended_sub_timelines) > 0:
                    # Adding the new_sub_timelines to mongoDB
                    self.MONGO_CLIENT["model_graph"]["timelines"].insert(extended_sub_timelines)
        # End of loop

        self.end_time = datetime.now()
        self.MONGO_CLIENT["ds_config"]["workflows"].update({"_id": self.workflow_id}, {"$set": {"start_time": self.start_time,
                                                                                                "end_time": self.end_time,
                                                                                                }})
        # topK = Timeline(probability, self.workflow_id, self.MONGO_CLIENT)
        # topK.find_topK_skylines()
        # skyline_indices = topK.skyline_timeline_object_indices
        # top_k_timelines = [top_k_timelines[s_idx] for s_idx in skyline_indices]
        # data = self.format_timelines_to_output(top_k_timelines)
        # # Need to delete the generated timelines
        # self.MONGO_CLIENT["model_graph"]["timelines"].remove({"workflow_id": self.workflow_id})
        # return data
        # self.MONGO_CLIENT["model_graph"]["topology_timelines"].insert(top_k_timelines)
        return

    def generate_timelines_via_joins(self, K, diversity):
        start = datetime.now()
        cache = defaultdict(dict)
        model_list = self.MONGO_CLIENT["ds_state"]["runtime"].find_one({})["level_order"]
        model_paths_dict = defaultdict(list)
        model_paths_dict_indices = dict()
        causal_pairs = list()
        total_timelines_count = 1
        for model in model_list:
            model_paths = list(self.MONGO_CLIENT["model_graph"]["model_paths"].find({"model_type": model}))
            model_paths_dict[model] = model_paths
            model_paths_dict_indices[model] = list(range(0, len(model_paths)))
            model_config = utils.access_model_by_name(self.config, model)
            causal_pairs.extend([(upstream_model, model) for upstream_model in model_config["upstream_models"].values()])
            total_timelines_count *= len(model_paths)
        # End of loop

        timeline_idx_scores = numpy.zeros((total_timelines_count, 2))
        for unique_idx, timeline_mp_idx in enumerate(list(itertools.product(*model_paths_dict_indices.values()))):
            int_causal_edges = 0
            ext_causal_edges = 0
            timeline = {str(model_list[model_idx]): model_paths_dict[model_list[model_idx]][model_path_idx] for model_idx, model_path_idx in
                        enumerate(
                            timeline_mp_idx)}
            int_causal_edges = sum([model_path["internal_causal_edges"] for model_path in timeline.values()])
            for upstream_model, model in causal_pairs:
                key1 = timeline[upstream_model]["_id"]
                key2 = timeline[model]["_id"]
                if key1 not in cache or key2 not in cache[key1]:
                    cache[key1][key2] = self.causal_edges_between_model_paths(timeline[model], timeline[upstream_model])
                ext_causal_edges += cache[key1][key2]
            # End of loop

            # saving the timeline
            timeline_object = dict()
            timeline_object["_id"] = unique_idx
            timeline_object["internal_causal_edges"] = int_causal_edges
            timeline_object["external_causal_edges"] = ext_causal_edges
            timeline_object["causal_edges"] = int_causal_edges + ext_causal_edges
            timeline_object["model_paths"] = timeline
            self.MONGO_CLIENT["model_graph"]["timelines_all"].insert_one(timeline_object)
            timeline_idx_scores[unique_idx] = [unique_idx, timeline_object["causal_edges"]]

        # sorting the timelines
        timeline_idx_scores = timeline_idx_scores[numpy.argsort(-timeline_idx_scores[:, 1])]
        # finding timelines
        topK_timelines = list()
        topK_timelines_scores = list()
        mp_in_top_K = set()
        count = 0
        for timeline_idx, causal_edges in timeline_idx_scores:
            timeline = self.MONGO_CLIENT["model_graph"]["timelines_all"].find_one({"_id": timeline_idx})
            model_paths_id_set = set([str(model_path["_id"]) for model_path in timeline["model_paths"].values()])

            if len(model_paths_id_set.intersection(mp_in_top_K)) <= diversity:
                topK_timelines.append(timeline)
                topK_timelines_scores.append(timeline["causal_edges"])
                count += 1
                mp_in_top_K.update(model_paths_id_set)
                if count == K:
                    break
        # End of loop
        self.MONGO_CLIENT["model_graph"]["topK_joins"].insert(topK_timelines)
        end_time = datetime.now()
        return (end_time - start).total_seconds()

    def check_correctness(self):
        timelines = self.MONGO_CLIENT["model_graph"]["topK_joins"].find()
        mp_in_topK = set()
        for timeline in timelines:
            max_reused = 0
            model_paths_id_list = [str(model_path["_id"]) for model_path in timeline["model_paths"].values()]
            for model_path_id in model_paths_id_list:
                if str(model_path_id) in mp_in_topK:
                    max_reused += 1

            mp_in_topK.update(model_paths_id_list)
            timeline["reused"] = max_reused
            self.MONGO_CLIENT["model_graph"]["topK_joins"].save(timeline)
        # End of loop

    def generate_timelines_via_A_star_no_diverse(self, K):
        """"""
        self.start_time = datetime.now()
        # Generating a topological_order
        level_ordered_models = self.MONGO_CLIENT["ds_state"]["runtime"].find_one({})["level_order"]
        joins = 0
        no_of_models = len(level_ordered_models)
        count = 0
        total_causal_pairs = 0
        cache = defaultdict(dict)
        top_k_timelines = list()
        self.MONGO_CLIENT["model_graph"]["timelines"].remove({})
        window_count = self.MONGO_CLIENT["model_graph"]["workflows"].find_one({"workflow_id": self.workflow_id})["window_count"]
        max_external_causal_edges = 0

        for model_id, model_config in self.DS_CONFIG["model_settings"].items():
            if "downstream_models" in model_config:
                total_causal_pairs += len(model_config["downstream_models"])
                for down_stream_model in model_config["downstream_models"].values():
                    max_external_causal_edges += window_count[down_stream_model]

        while count < K:
            # Finding the highest ranked sub timeline
            candidate_sub_timeline = list(
                self.MONGO_CLIENT["model_graph"]["timelines"].find({"no_of_models": {"$lt": no_of_models}}).sort(
                    [("heuristic_score", -1), ("no_of_models", -1)]).limit(1))
            extended_sub_timelines = list()
            if not candidate_sub_timeline:
                if count > 0:
                    break
                # creating a new sub_timeline
                for model_path in self.MONGO_CLIENT["model_graph"]["model_paths"].find({"model_type": level_ordered_models[0]}):
                    extended_sub_timelines.append({
                        "causal_edges": model_path["internal_causal_edges"],
                        "reused": 0,
                        "internal_causal_edges": model_path["internal_causal_edges"],
                        "external_causal_edges": 0,
                        "heuristic_score": model_path["internal_causal_edges"] * no_of_models + max_external_causal_edges,
                        "causal_pairs": 0,
                        "no_of_models": 1,
                        "total_joins": 0,
                        "model_paths": {
                            level_ordered_models[0]: {
                                "instance_list": model_path["model_path"],
                                "_id": model_path["_id"]
                            }
                        }})
                # End of loop
                self.MONGO_CLIENT["model_graph"]["timelines"].insert(extended_sub_timelines)
            else:
                candidate_sub_timeline = candidate_sub_timeline[0]
                next_model_idx = candidate_sub_timeline["no_of_models"]
                next_model = level_ordered_models[next_model_idx]
                model_config = utils.access_model_by_name(self.config, next_model)

                if "upstream_models" in model_config:
                    candidate_sub_timeline["causal_pairs"] += len(model_config["upstream_models"])

                for model_path in self.MONGO_CLIENT["model_graph"]["model_paths"].find({"model_type": next_model}):
                    joins += 1
                    internal_causal_edges = candidate_sub_timeline["internal_causal_edges"] + model_path["internal_causal_edges"]
                    external_causal_edges = candidate_sub_timeline["external_causal_edges"]
                    # Adding the external causal edges
                    if "upstream_models" in model_config:
                        for upstream_model_type in model_config["upstream_models"].values():
                            key1 = str(candidate_sub_timeline["model_paths"][upstream_model_type]["_id"])
                            key2 = str(model_path["_id"])
                            if key1 not in cache or key2 not in cache[key1]:
                                upstream_model_path = candidate_sub_timeline["model_paths"][upstream_model_type]["instance_list"]
                                cache[key1][key2] = self.causal_edges_between_model_paths(model_path, {"model_path": upstream_model_path,
                                                                                                       "model_type": upstream_model_type})
                            external_causal_edges += cache[key1][key2]
                        # End of loop
                    # Extending the candidate_timeline
                    causal_edges = internal_causal_edges + external_causal_edges
                    internal_causal_edges_avg = math.ceil(internal_causal_edges / (next_model_idx + 1))
                    heuristic_score = causal_edges
                    if candidate_sub_timeline["causal_pairs"] > 0:
                        external_causal_edges_avg = math.ceil(external_causal_edges / candidate_sub_timeline["causal_pairs"])
                        heuristic_score += internal_causal_edges_avg * (no_of_models - (next_model_idx + 1)) + \
                                           external_causal_edges_avg * (total_causal_pairs - candidate_sub_timeline["causal_pairs"])
                        # End of loop
                    else:
                        heuristic_score += max_external_causal_edges + internal_causal_edges_avg * len(level_ordered_models[next_model_idx + 1:])

                    new_candidate_sub_timeline = dict({
                        "_id": bson.objectid.ObjectId(),
                        "causal_edges": causal_edges,
                        "model_paths": dict(candidate_sub_timeline["model_paths"]),
                        "internal_causal_edges": internal_causal_edges,
                        "external_causal_edges": external_causal_edges,
                        "causal_pairs": candidate_sub_timeline["causal_pairs"],
                        "heuristic_score": heuristic_score,
                        "no_of_models": next_model_idx + 1})
                    new_candidate_sub_timeline["model_paths"][next_model] = dict(
                        {"instance_list": model_path["model_path"], "_id": model_path["_id"]})
                    extended_sub_timelines.append(new_candidate_sub_timeline)
                # End of loop

                self.MONGO_CLIENT["model_graph"]["timelines"].delete_one({"_id": candidate_sub_timeline["_id"]})

                if next_model_idx + 1 == no_of_models and len(extended_sub_timelines) > 0:
                    # retaining all the timelines
                    top_k_timelines.extend(extended_sub_timelines)
                    for top_timeline in extended_sub_timelines:
                        top_timeline["insert_time"] = datetime.now()
                        top_timeline["workflow_id"] = self.workflow_id
                        self.MONGO_CLIENT["model_graph"]["timelines"].insert(top_timeline)
                        count += 1
                    # End of loop
                elif len(extended_sub_timelines) > 0:
                    # Adding the new_sub_timelines to mongoDB
                    self.MONGO_CLIENT["model_graph"]["timelines"].insert(extended_sub_timelines)

        # End of loop

        self.end_time = datetime.now()
        self.MONGO_CLIENT["ds_config"]["workflows"].update({"_id": self.workflow_id}, {"$set": {"start_time": self.start_time,
                                                                                                "end_time": self.end_time,
                                                                                                }})
        return

    def generate_timelines_via_A_star_dfs(self, K, diversity):
        """"""
        self.start_time = datetime.now()
        # Generating a topological_order
        level_ordered_models = self.MONGO_CLIENT["ds_state"]["runtime"].find_one({})["dfs_order"]
        joins = 0
        no_of_models = len(level_ordered_models)
        timelines_scores = list()
        count = 0
        total_causal_pairs = 0
        visited_dict = defaultdict(bool)
        model_visited_dict = defaultdict(int)
        cache = defaultdict(dict)
        # We want a timeline can't have more #diversity re-used model-paths
        model_path_count = defaultdict(int)
        top_k_timelines = list()
        for model in level_ordered_models:
            model_path_count[model] = len(list(self.MONGO_CLIENT["model_graph"]["model_paths"].find({"model_type": model})))

        window_count = self.MONGO_CLIENT["model_graph"]["workflows"].find_one({"workflow_id": self.workflow_id})["window_count"]
        max_external_causal_edges = 0
        for model_id, model_config in self.DS_CONFIG["model_settings"].items():
            if "downstream_models" in model_config:
                total_causal_pairs += len(model_config["downstream_models"])
                for down_stream_model in model_config["downstream_models"].values():
                    max_external_causal_edges += window_count[down_stream_model]

        while count < K:
            # Finding the highest ranked sub_timeline
            candidate_sub_timeline = list(
                self.MONGO_CLIENT["model_graph"]["timelines"].find({"no_of_models": {"$lt": no_of_models}}).sort(
                    [("heuristic_score", -1), ("no_of_models", -1)]).limit(1))
            extended_sub_timelines = list()
            if not candidate_sub_timeline:
                if count > 0:
                    break
                # creating a new sub_timeline
                for model_path in self.MONGO_CLIENT["model_graph"]["model_paths"].find({"model_type": level_ordered_models[0]}):
                    extended_sub_timelines.append({
                        "causal_edges": model_path["internal_causal_edges"],
                        "reused": 0,
                        "internal_causal_edges": model_path["internal_causal_edges"],
                        "external_causal_edges": 0,
                        "heuristic_score": model_path["internal_causal_edges"] * no_of_models + max_external_causal_edges,
                        "causal_pairs": 0,
                        "no_of_models": 1,
                        "total_joins": 0,
                        "model_paths": {
                            level_ordered_models[0]: {
                                "instance_list": model_path["model_path"],
                                "_id": model_path["_id"]
                            }
                        }})
                # End of loop
                self.MONGO_CLIENT["model_graph"]["timelines"].insert(extended_sub_timelines)
            else:
                candidate_sub_timeline = candidate_sub_timeline[0]
                next_model_idx = candidate_sub_timeline["no_of_models"]
                next_model = level_ordered_models[next_model_idx]
                model_config = utils.access_model_by_name(self.config, next_model)

                # Updating causal_pairs
                if "upstream_models" in model_config:
                    candidate_sub_timeline["causal_pairs"] += len(model_config["upstream_models"])

                n = candidate_sub_timeline["no_of_models"]
                c = 0
                for idx, model in enumerate(reversed(level_ordered_models[n:])):
                    if model_visited_dict[model] == model_path_count[model]:
                        c += 1

                for model_path in self.MONGO_CLIENT["model_graph"]["model_paths"].find({"model_type": next_model}):
                    reused = candidate_sub_timeline["reused"]
                    # A timeline can't have more than #diversity visited model-paths
                    if visited_dict[str(model_path["_id"])]:
                        if candidate_sub_timeline["reused"] >= diversity - c:
                            continue
                        else:
                            reused += 1

                    joins += 1
                    internal_causal_edges = candidate_sub_timeline["internal_causal_edges"] + model_path["internal_causal_edges"]
                    external_causal_edges = candidate_sub_timeline["external_causal_edges"]
                    # Adding the external causal edges
                    if "upstream_models" in model_config:
                        for upstream_model_type in model_config["upstream_models"].values():
                            if upstream_model_type in candidate_sub_timeline["model_paths"]:
                                key1 = str(candidate_sub_timeline["model_paths"][upstream_model_type]["_id"])
                                key2 = str(model_path["_id"])
                                if key1 not in cache or key2 not in cache[key1]:
                                    upstream_model_path = candidate_sub_timeline["model_paths"][upstream_model_type]["instance_list"]
                                    cache[key1][key2] = self.causal_edges_between_model_paths(model_path, {"model_path": upstream_model_path,
                                                                                                           "model_type": upstream_model_type})
                                external_causal_edges += cache[key1][key2]
                        # End of loop
                        for downstream_model_type in model_config["downstream_models"].values():
                            if downstream_model_type in candidate_sub_timeline["model_paths"]:
                                key1 = str(candidate_sub_timeline["model_paths"][downstream_model_type]["_id"])
                                key2 = str(model_path["_id"])
                                if key1 not in cache or key2 not in cache[key1]:
                                    upstream_model_path = candidate_sub_timeline["model_paths"][downstream_model_type]["instance_list"]
                                    cache[key1][key2] = self.causal_edges_between_model_paths({"model_path": upstream_model_path,
                                                                                               "model_type": downstream_model_type}, model_path)
                                external_causal_edges += cache[key1][key2]
                        # End of loop

                    # Extending the candidate_timeline
                    causal_edges = internal_causal_edges + external_causal_edges
                    internal_causal_edges_avg = math.ceil(internal_causal_edges / (next_model_idx + 1))
                    heuristic_score = causal_edges
                    if candidate_sub_timeline["causal_pairs"] > 0:
                        external_causal_edges_avg = math.ceil(external_causal_edges / candidate_sub_timeline["causal_pairs"])
                        heuristic_score += internal_causal_edges_avg * (no_of_models - (next_model_idx + 1)) + \
                                           external_causal_edges_avg * (total_causal_pairs - candidate_sub_timeline["causal_pairs"])
                        # End of loop
                    else:
                        heuristic_score += max_external_causal_edges + internal_causal_edges_avg * len(level_ordered_models[next_model_idx + 1:])

                    new_candidate_sub_timeline = dict({
                        "_id": bson.objectid.ObjectId(),
                        "causal_edges": causal_edges,
                        "model_paths": dict(candidate_sub_timeline["model_paths"]),
                        "internal_causal_edges": internal_causal_edges,
                        "external_causal_edges": external_causal_edges,
                        "causal_pairs": candidate_sub_timeline["causal_pairs"],
                        "heuristic_score": heuristic_score,
                        "reused": reused,
                        "no_of_models": next_model_idx + 1})
                    new_candidate_sub_timeline["model_paths"][next_model] = dict(
                        {"instance_list": model_path["model_path"], "_id": model_path["_id"]})
                    extended_sub_timelines.append(new_candidate_sub_timeline)
                # End of loop

                self.MONGO_CLIENT["model_graph"]["timelines"].delete_one({"_id": candidate_sub_timeline["_id"]})

                if next_model_idx + 1 == no_of_models and len(extended_sub_timelines) > 0:
                    if diversity < no_of_models:
                        # retaining the top-timeline
                        top_extended_timeline = {"heuristic_score": -1, "model_paths": dict()}
                        for timeline in extended_sub_timelines:
                            if timeline["heuristic_score"] > top_extended_timeline["heuristic_score"]:
                                top_extended_timeline = timeline
                        # End of loop

                        top_k_timelines.append(top_extended_timeline)
                        top_extended_timeline["insert_time"] = datetime.now()
                        for model_type, model_path in top_extended_timeline["model_paths"].items():
                            if not visited_dict[str(model_path["_id"])]:
                                visited_dict[str(model_path["_id"])] = True
                                model_visited_dict[model_type] += 1
                                self.MONGO_CLIENT["model_graph"]["timelines"].update_many({"model_paths." + model_type + "._id": model_path["_id"],
                                                                                           "no_of_models": {"$lt": no_of_models}},
                                                                                          {"$inc": {"reused": 1}})
                        # End of loop

                        timelines_scores.append(top_extended_timeline["causal_edges"])
                        count += 1

                        # Removing, where reused is greater than diversity
                        self.MONGO_CLIENT["model_graph"]["timelines"].remove({"reused": {"$gt": diversity}, "no_of_models": {"$lt": no_of_models}})

                        # cleaning up-subtrees which can't grow further
                        max_reused = diversity
                        for idx, model in enumerate(reversed(level_ordered_models)):
                            # Check if the model has all model_paths visited
                            if model_visited_dict[model] == model_path_count[model]:
                                self.MONGO_CLIENT["model_graph"]["timelines"].remove({"reused": {"$gt": max_reused - 1}, "no_of_models": {"$lte": idx}})
                                max_reused -= 1

                        top_extended_timeline["workflow_id"] = self.workflow_id
                        top_extended_timeline["total_joins"] = joins
                        self.MONGO_CLIENT["model_graph"]["timelines"].insert(top_extended_timeline)
                    else:
                        # retaining all the timelines
                        top_k_timelines.extend(extended_sub_timelines)
                        for top_timeline in extended_sub_timelines:
                            top_timeline["insert_time"] = datetime.now()
                            top_timeline["workflow_id"] = self.workflow_id
                            top_timeline["total_joins"] = joins
                            self.MONGO_CLIENT["model_graph"]["timelines"].insert(top_timeline)
                            count += 1
                        # End of loop
                elif len(extended_sub_timelines) > 0:
                    # Adding the new_sub_timelines to mongoDB
                    self.MONGO_CLIENT["model_graph"]["timelines"].insert(extended_sub_timelines)
        # End of loop

        self.end_time = datetime.now()
        self.MONGO_CLIENT["ds_config"]["workflows"].update({"_id": self.workflow_id}, {"$set": {"start_time": self.start_time,
                                                                                                "end_time": self.end_time,
                                                                                                }})

        return
