import numpy as np
from app.skyline.timeline_object import TimelineObject
from app.skyline.instance_object import InstanceObject
from heapq import heapify, heappush, heappop
from rtree import index
from collections import defaultdict
import sys
from bson.objectid import ObjectId

class Timeline:
    def __init__(self, timelines_id_list, min_probability, workflow_id, mongo_client):
        self.timeline_objects = self.create_timeline_objects(timelines_id_list)
        self.timeline_db = None
        self.minHeap = []
        # stores timeline_object_indices with values from Umin
        self.global_rtree = index.Rtree('globalRtree')
        self.min_probability = min_probability
        self.non_skyline_timeline_object_indices = set()
        self.skyline_timeline_object_indices = set()
        self.mongo_client = mongo_client
        self.config = self.mongo_client["ds_config"]["workflows"].find_one({"_id": ObjectId(workflow_id)})
        heapify(self.minHeap)

    def create_timeline_objects(self, timelines_id_list):
        """

        :param timelines:
        :return:
        """
        # for timeline_id in timelines:
        # TODO: Need to create a post-process the data to divide the parameter into equal segment sizes
        # TODO: Hardcoded interest_parameters
        interest_parameters = {"rainfall": "max", "windspeed": "min"}
        segment_size = 43200
        for timeline_id in timelines_id_list:
            timeline = self.mongo_client["ds_visualize"]["timeline"].find_one({"_id": ObjectId(timeline_id)})
            for model, model_path in timeline["data"]:
                values = defaultdict(lambda: 0)
                end = 0
                for dsir_id in model_path:
                    dsir = self.mongo_client["ds_results"]["dsir"].find_one({"_id": ObjectId(timeline_id)})
                    end = end + dsir["metadata"]["temporal"]["end"]
                    # This dsir did not get clustered
                    if dsir["created_by"] == "JobGateway":
                        jobs = self.mongo_client["ds_results"]["jobs"].find_one({"output_dsir": dsir_id})
                    else:
                        jobs = self.mongo_client["ds_results"]["jobs"].find({"output_dsir": {"$in": dsir["parents"]}})
                    for job in jobs:
                        for parameter in interest_parameters.keys():
                            values[parameter] += job["variables"][parameter]
                        # End of loop
                    # End of loop
                    if end >= segment_size:
                        end = end - segment_size
                        # creating an instance for the timeline





        return list()

    def find_topK_skylines(self):
        """

        :param objects:
        :return:
        """
        # Finding the minimum-bounding region
        while len(self.minHeap):
            key, timeline_object_index, instance_index = heappop(self.minHeap)
            instance = self.timeline_objects[timeline_object_index].instances[instance_index]
            # Check if the instance is a virtual instance
            if instance_index == self.timeline_objects[timeline_object_index].virtual_instance_min:
                dominating_timeline_object_indices = self.find_dominating_timeline_object_indices(timeline_object_index, instance_index)
                probability = self.compute_probability(instance, dominating_timeline_object_indices)
                self.timeline_objects[timeline_object_index].instances[instance_index].probability = probability
                if probability >= self.min_probability:
                    self.timeline_objects[timeline_object_index].partition_instances_to_layers()
                else:
                    self.non_skyline_timeline_object_indices.add(timeline_object_index)

            else:
                dominating_timeline_object_indices = self.find_dominating_timeline_object_indices(timeline_object_index, instance_index)
                probability = self.compute_probability(instance, dominating_timeline_object_indices)
                self.timeline_objects[timeline_object_index].instances[instance_index].probability = probability
                self.timeline_objects[timeline_object_index].prob_lower_limit += (1 / self.timeline_objects[timeline_object_index].instances_count) \
                                                                                 * probability

                is_last_instance_in_layer, layer_idx = self.timeline_objects[timeline_object_index].is_last_instance_in_layer(instance_index)

                if is_last_instance_in_layer:
                    # update Upmax
                    self.timeline_objects[timeline_object_index].update_Pmax(layer_idx)

                self.timeline_objects[timeline_object_index].prob_upper_limit = self.timeline_objects[timeline_object_index].prob_lower_limit + \
                                                                                self.timeline_objects[timeline_object_index].UPmax * \
                                                                                (self.timeline_objects[timeline_object_index].instances_count -
                                                                                 instance_index + 1) / self.timeline_objects[timeline_object_index].instances_count

                if ((instance_index + 1) / self.timeline_objects[timeline_object_index].instances_count) * \
                        self.timeline_objects[timeline_object_index].UPmin < self.min_probability:
                    self.prune_other_timeline_objects(timeline_object_index, instance_index)

                if self.timeline_objects[timeline_object_index].prob_lower_limit >= self.min_probability:
                    self.skyline_timeline_object_indices.add(timeline_object_index)

                if self.timeline_objects[timeline_object_index].prob_upper_limit >= self.min_probability:
                    next_instance_index = self.timeline_objects[timeline_object_index].find_next_instance_in_layer(layer_idx, instance_index)
                    if next_instance_index is not None:
                        heappush(self.minHeap, [sum(self.timeline_objects[timeline_object_index][instance_index + 1].value), timeline_object_index,
                                     instance_index+1])
                    else:
                        # TODO: Need to delete the timeline_object and update the Rtree
                        self.non_skyline_timeline_object_indices.add(timeline_object_index)
        # End of loop
        return

    def find_dominating_timeline_object_indices(self, timeline_object_index, instance_index):
        """
        Function to compute the dominating_timeline_object ids for a given virtual instance
        :param timeline_object_index:
        :param instance_index:
        :return:
        """
        Umax = self.timeline_objects[timeline_object_index].instances[instance_index]
        query = [0] * Umax.dim
        query.extend(Umax.value)
        dominating_timeline_object_indices = [candidate_timeline_object_index for candidate_timeline_object_index in
                                              self.global_rtree.intersection(query) if timeline_object_index != candidate_timeline_object_index]

        return [idx for idx in dominating_timeline_object_indices if idx not in self.non_skyline_timeline_object_indices]

    def find_virtual_max_instance(self, timeline_object_index, limit):
        Umax = [-sys.maxsize - 1] * self.timeline_objects[timeline_object_index].dim
        for instance in self.timeline_objects[timeline_object_index][:limit + 1]:
            for idx, value in enumerate(instance.value):
                Umax[idx] = max(Umax[idx], value)
        return Umax

    def prune_other_timeline_objects(self, timeline_object_index, instance_index):
        """
        :param instance:
        :param instances_list:
        :return:
        """
        Umax = self.find_virtual_max_instance(timeline_object_index, instance_index)
        query = [0] * len(Umax)
        query.extend(Umax)
        pruning_candidate_timeline_object_indices = self.global_rtree.intersection(query)
        self.non_skyline_timeline_object_indices.update(pruning_candidate_timeline_object_indices)
        # TODO: Need to delete those non-skyline timeline_objects in the global R-Tree

    def compute_probability(self, candidate_instance, dominating_timeline_object_indices):
        """
        Function to compute the probability of the instance
        :param candidate_instance:
        :param dominating_timeline_object_indices:
        :return:
        """
        key = np.sum(candidate_instance.value)
        query = [0] * candidate_instance.dim
        query.extend(candidate_instance.value)
        probability = 1
        for idx in dominating_timeline_object_indices:
            timeline_obj_instances_list = self.timeline_objects[idx].instances
            candidate_dominating_instances = [instance for instance in timeline_obj_instances_list if np.sum(instance.value) < key]
            # candidate_dominating_instances = self.find_dominating_candidate_instances(candidate_instance, candidate_dominating_instances)
            local_rtree = index.Rtree('localRtree')
            for idfr, candidate_instance in enumerate(candidate_dominating_instances):
                mbb = candidate_instance.values
                mbb.extend(candidate_instance.values)
                local_rtree.insert(idfr, mbb)

            dominating_instance_list = [candidate_dominating_instances[i] for i in local_rtree.intersection(query)]
            dominating_instances_count = len(dominating_instance_list)
            probability *= (1 - dominating_instances_count) / self.timeline_objects[idx].instances_count
        # End of loop
        return probability
