import numpy as np
from app.skyline.timeline_object import TimelineObject
from app.skyline.instance_object import InstanceObject
from heapq import heapify, heappush, heappop
from rtree import index
from collections import defaultdict
import sys
from bson.objectid import ObjectId

p = index.Property()


class Timeline:
    def __init__(self, timelines_id_list, min_probability, workflow_id, mongo_client):
        self.timeline_objects = list()
        self.create_timeline_objects(timelines_id_list)
        self.timeline_db = None
        # stores timeline_object_indices with values from Umin
        self.global_rtree = index.Rtree('globalRtree')
        self.min_probability = min_probability
        self.non_skyline_timeline_object_indices = set()
        self.skyline_timeline_object_indices = set()
        self.mongo_client = mongo_client
        self.config = self.mongo_client["ds_config"]["workflows"].find_one({"_id": ObjectId(workflow_id)})
        self.dim = 2
        p.dimension = self.dim
        self.minHeap = self.fill_minHeap()

    def create_timeline_objects(self, timelines_id_list):
        """
        Function to create timeline_objects (Multi-instance Objects)
        :param timelines_id_list: timeline_ids
        :return: None
        """
        # TODO: Need to integrate #causal edges and #interleaved time-instants
        # TODO: Hardcoded interest_parameters
        interest_parameters = {"rainfall": "max", "windspeed": "min"}
        segment_size = 43200
        for timeline_id in timelines_id_list:
            timeline = self.mongo_client["ds_visualize"]["timeline"].find_one({"_id": ObjectId(timeline_id)})
            instance_list = list()
            for model, model_path in timeline["data"]:
                values = defaultdict(list)
                end = 0
                begin = sys.maxsize
                for dsir_id in model_path:
                    dsir = self.mongo_client["ds_results"]["dsir"].find_one({"_id": ObjectId(dsir_id)})
                    end = end + dsir["metadata"]["temporal"]["end"]
                    begin = min(begin, dsir["metadata"]["temporal"]["end"])
                    # This dsir did not get clustered
                    if dsir["created_by"] == "JobGateway":
                        jobs = self.mongo_client["ds_results"]["jobs"].find_one({"output_dsir": dsir_id})
                    else:
                        jobs = self.mongo_client["ds_results"]["jobs"].find({"output_dsir": {"$in": dsir["parents"]}})
                    # TODO: Need to fix weights
                    weight = (dsir["metadata"]["temporal"]["end"] - dsir["metadata"]["temporal"]["begin"]) / abs(end - dsir["metadata"]["temporal"]["end"])
                    for job in jobs:
                        for parameter in interest_parameters.keys():
                            values[parameter].append(job["variables"][parameter] * weight)
                        # End of loop
                    # End of loop
                    if end - begin >= segment_size:
                        end = 0
                        begin = sys.maxsize
                        # creating an instance for the timeline
                        values = [sum(value_list) / len(value_list) if interest_parameters[key] == 'min' else 1 / (sum(value_list) / len(value_list)) for
                                  key, value_list in values.items()]
                        instance_list.append(values)
                        values = defaultdict(list)
            # End of loop
            self.timeline_objects.append(TimelineObject(instance_list, self.dim))
        # End of loop
        return

    def find_topK_skylines(self):
        """
        Function to compute the topK skylines
        """
        # Finding the minimum-bounding region
        while len(self.minHeap) > 0:
            key, timeline_object_index, instance_index = heappop(self.minHeap)
            instance = self.timeline_objects[timeline_object_index].instances[instance_index]
            # Check if the instance is a virtual instance
            if instance_index == self.timeline_objects[timeline_object_index].virtual_instance_min_idx:
                # Finding the dominating timeline_objects
                dominating_timeline_object_indices = self.find_dominating_timeline_object_indices(timeline_object_index, instance_index)
                probability = self.compute_probability(instance, dominating_timeline_object_indices)
                self.timeline_objects[timeline_object_index].instances[instance_index].probability = probability
                if probability >= self.min_probability:
                    self.timeline_objects[timeline_object_index].partition_instances_to_layers(self.dim)
                    next_instance_index = self.timeline_objects[timeline_object_index].layers[0][0]
                    heappush(self.minHeap, (sum(self.timeline_objects[timeline_object_index].instances[next_instance_index].value), timeline_object_index,
                                            next_instance_index))
                else:
                    self.non_skyline_timeline_object_indices.add(timeline_object_index)
                    self.remove_timeline_object_from_global_Rtree(timeline_object_index)

            else:
                dominating_timeline_object_indices = self.find_dominating_timeline_object_indices(timeline_object_index, instance_index)
                # Applying Pruning Rule 3
                if not self.check_complete_dominating_object_indices(timeline_object_index, instance_index, dominating_timeline_object_indices):
                    probability = self.compute_probability(instance, dominating_timeline_object_indices)
                    self.timeline_objects[timeline_object_index].instances[instance_index].probability = probability
                    self.timeline_objects[timeline_object_index].prob_lower_limit += (1 / self.timeline_objects[timeline_object_index].instances_count) * \
                                                                                     probability
                else:
                    # TODO: need to evaluate this case
                    probability = 0
                    self.timeline_objects[timeline_object_index].instances[instance_index].probability = 0

                is_last_instance_in_layer, layer_idx = self.timeline_objects[timeline_object_index].is_last_instance_in_layer(instance_index)
                if is_last_instance_in_layer:
                    # update Upmax
                    self.timeline_objects[timeline_object_index].update_Pmax(layer_idx)

                self.timeline_objects[timeline_object_index].processed_instance_count += 1
                self.timeline_objects[timeline_object_index].update_UPmin(probability)

                self.timeline_objects[timeline_object_index].prob_upper_limit = self.timeline_objects[timeline_object_index].prob_lower_limit + \
                                                                                self.timeline_objects[timeline_object_index].UPmax * \
                                                                                (self.timeline_objects[timeline_object_index].instances_count -
                                                                                 self.timeline_objects[timeline_object_index].processed_instance_count) / \
                                                                                self.timeline_objects[timeline_object_index].instances_count

                if (self.timeline_objects[timeline_object_index].processed_instance_count / self.timeline_objects[timeline_object_index].instances_count) * \
                        self.timeline_objects[timeline_object_index].UPmin < self.min_probability:
                    self.prune_other_timeline_objects(timeline_object_index, instance_index)

                if self.timeline_objects[timeline_object_index].prob_lower_limit >= self.min_probability:
                    self.skyline_timeline_object_indices.add(timeline_object_index)

                elif self.timeline_objects[timeline_object_index].prob_upper_limit >= self.min_probability:
                    next_instance_index = self.timeline_objects[timeline_object_index].find_next_instance_in_layer(layer_idx, instance_index)
                    if next_instance_index is not None:
                        heappush(self.minHeap, (sum(self.timeline_objects[timeline_object_index][next_instance_index].value), timeline_object_index,
                                                next_instance_index))
                    else:
                        # TODO: Need to delete the timeline_object and update the Rtree
                        self.non_skyline_timeline_object_indices.add(timeline_object_index)
        # End of loop

    def find_dominating_timeline_object_indices(self, timeline_object_index, instance_index):
        """
        Function to compute the dominating_timeline_object ids for a given virtual instance
        :param timeline_object_index:
        :param instance_index:
        :return:
        """
        Umax = self.timeline_objects[timeline_object_index].instances[instance_index]
        query = [0] * self.dim
        query.extend(Umax.value)
        dominating_timeline_object_indices = [candidate_timeline_object_index for candidate_timeline_object_index in
                                              self.global_rtree.intersection(query) if timeline_object_index != candidate_timeline_object_index]

        return [idx for idx in dominating_timeline_object_indices if idx not in self.non_skyline_timeline_object_indices]

    def find_virtual_max_instance(self, timeline_object_index, limit):
        Umax = [-sys.maxsize - 1] * self.dim
        for instance in self.timeline_objects[timeline_object_index][:limit + 1]:
            for idx, value in enumerate(instance.value):
                Umax[idx] = max(Umax[idx], value)
        return Umax

    def prune_other_timeline_objects(self, timeline_object_index, instance_index):
        """
        :param timeline_object_index:
        :param instance_index:
        :return:
        """
        Umax = self.find_virtual_max_instance(timeline_object_index, instance_index)
        query = [0] * self.dim
        query.extend(Umax)
        pruning_candidate_timeline_object_indices = self.global_rtree.intersection(query)
        self.non_skyline_timeline_object_indices.update(pruning_candidate_timeline_object_indices)
        for delete_timeline_object_index in pruning_candidate_timeline_object_indices:
            vmin_idx = self.timeline_objects[delete_timeline_object_index]
            mbb = list(self.timeline_objects[delete_timeline_object_index].instances[vmin_idx].value)
            mbb.extend(mbb)
            self.global_rtree.delete(delete_timeline_object_index, mbb)

    def compute_probability(self, candidate_instance, dominating_timeline_object_indices):
        """
        Function to compute the probability of the instance
        :param candidate_instance:
        :param dominating_timeline_object_indices:
        :return:
        """
        # Note: The dominating_timeline_object_indices won't have the timeline_object_index which the candidate_instance belongs to
        threshold = sum(candidate_instance.value)
        query = [0] * self.dim
        query.extend(candidate_instance.value)
        probability = 1
        for idx in dominating_timeline_object_indices:
            timeline_obj_instances_list = self.timeline_objects[idx].instances
            # TODO: Need to crosscheck less than relationship
            candidate_dominating_instances = [instance for instance in timeline_obj_instances_list if sum(instance.value) < threshold]
            # candidate_dominating_instances = self.find_dominating_candidate_instances(candidate_instance, candidate_dominating_instances)
            local_rtree = index.Rtree('localRtree', properties=p)
            for idfr, candidate_instance in enumerate(candidate_dominating_instances):
                mbb = list(candidate_instance.value)
                mbb.extend(candidate_instance.value)
                local_rtree.insert(idfr, mbb)

            dominating_instance_list = [candidate_dominating_instances[i] for i in local_rtree.intersection(query)]
            dominating_instances_count = len(dominating_instance_list)
            probability *= (1 - dominating_instances_count) / self.timeline_objects[idx].instances_count
        # End of loop
        return probability

    def check_complete_dominating_object_indices(self, timeline_object_index, instance_index, dominating_timeline_object_indices):
        """
        :param timeline_object_index:
        :param instance_index:
        :param dominating_timeline_object_indices:
        :return:
        """
        local_Rtree = index.Rtree('localRtree', properties=p)
        for dominating_timeline_object_index in dominating_timeline_object_indices:
            Vmax = list(self.timeline_objects[dominating_timeline_object_indices].virtual_instance_max)
            local_Rtree.insert(dominating_timeline_object_index, Vmax.extend(Vmax))

        query = [0] * self.dim
        query.extend(self.timeline_objects[timeline_object_index].instances[instance_index].value)
        if len(list(local_Rtree.intersection(query))) > 0:
            return True
        else:
            return False

    def fill_minHeap(self):
        """
        :return: minHeap
        """
        minHeap = list()
        for i, timeline_object in enumerate(self.timeline_objects):
            idx = timeline_object.virtual_instance_min_idx
            minHeap.append((sum(timeline_object.instances[idx].value), i, idx))
            mbb = timeline_object.instances[idx].value
            mbb.extend(timeline_object.instances[idx].value)
            self.global_rtree.insert(i, mbb)
        # End of loop
        heapify(minHeap)
        return minHeap

    def remove_timeline_object_from_global_Rtree(self, timeline_object_id):
        """

        :param timeline_object_id:
        :return:
        """
        vmin_idx = self.timeline_objects[timeline_object_id].virtual_instance_min_idx
        vmin = self.timeline_objects[timeline_object_id].instances[vmin_idx]
        mbb = vmin.extend(vmin)
        self.global_rtree.delete(timeline_object_id, mbb)
