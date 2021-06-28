import pathlib
import random
from heapq import heapify, heappush, heappop
from rtree import index
from collections import defaultdict
import sys
from bson.objectid import ObjectId
import pandas as pd
from datetime import datetime
import numpy as np

MODULES_PATH = pathlib.Path(__file__).resolve().parent.parent.absolute()
sys.path.append(str(MODULES_PATH))

p = index.Property()


class InstanceObject:
    def __init__(self, value):
        self.value = np.array(value)
        self.skyline = False
        self.probability = None


class TimelineObject:
    def __init__(self, instance_list, dim):
        self.instances = instance_list
        self.instances_count = len(self.instances)
        self.prob_lower_limit = 0
        self.prob_upper_limit = 1
        self.layers = list()
        self.UPmax = 1  # initial value
        self.UPmin = 1  # Visited(Processed) Subset Pmin
        self.virtual_instance_min = None
        self.virtual_instance_max = None
        self.compute_virtual_instances(dim)
        self.processed_instance_count = 0

    def check_dominating(self, dominating_instance, dominated_instance, dim):
        """
        :param dominating_instance:
        :param dominated_instance:
        :param dim:
        :return:
        """
        less_count, equal_count, greater_count = 0, 0, 0
        for i in range(dim):
            if dominating_instance.value[i] < dominated_instance.value[i]:
                less_count += 1
            elif dominating_instance.value[i] > dominated_instance.value[i]:
                greater_count += 1
            else:
                equal_count += 1

        if greater_count == 0 and less_count > 0:
            return 1
        elif less_count == 0 and greater_count > 0:
            return -1
        else:
            return 0

    def find_layer(self, instance, dim):
        """
        Function to compute the layer in which the instance should belong
        :param instance:
        :return:
        """
        beg = 0
        end = len(self.layers) - 1

        while beg <= end:
            mid = int(beg + (end - beg) / 2)
            update = False
            for layer_instances in self.layers[mid]:
                dominate = self.check_dominating(instance, self.instances[layer_instances], dim)
                if dominate == 1:
                    end = mid - 1
                    update = True
                    break
                elif dominate == -1:
                    beg = mid + 1
                    update = True
                    break
            # End of loop
            if not update:
                return mid
        # End of loop
        # TODO: Need to evaluate here
        return beg

    def partition_instances_to_layers(self, dim):
        """

        :param dim:
        :return:
        """
        # Note: We don't add the virtual instances into the layers
        self.instances.sort(key=lambda x: sum(x.value))
        for idx, instance in enumerate(self.instances):
            layer_index = self.find_layer(instance, dim)
            total_layers = len(self.layers)
            if layer_index < total_layers:
                self.layers[layer_index].append(idx)
            elif layer_index == total_layers:
                self.layers.append([idx])
            else:
                raise Exception("wrong layer found")
        # End of loop
        return

    def is_last_instance_in_layer(self, instance_idx, dim):
        layer_idx = self.find_layer(self.instances[instance_idx], dim)
        return (True, layer_idx) if self.layers[layer_idx][-1] == instance_idx else (False, layer_idx)

    def update_Pmax(self, layer_idx):
        self.UPmax = np.max([self.instances[instance_id].probability for instance_id in self.layers[layer_idx]])

    def find_next_instance_in_layer(self, layer_idx, instance_index):
        for index, layer_instance_idx in enumerate(self.layers[layer_idx]):
            if layer_instance_idx == instance_index:
                if index + 1 == len(self.layers[layer_idx]):
                    if layer_idx + 1 < len(self.layers):
                        return self.layers[layer_idx + 1][0]
                    else:
                        return None
                else:
                    return self.layers[layer_idx][index + 1]
        # End of loop
        return None

    def update_UPmin(self, probability):
        self.UPmin = min(self.UPmin, probability)

    def compute_virtual_instances(self, dim):
        """
        Function to compute the virtual_min and virtual_max instance
        :return:
        """
        virtual_min = [sys.maxsize] * dim
        virtual_max = [-1 - sys.maxsize] * dim
        for instance in self.instances:
            for idx, val in enumerate(instance.value):
                virtual_min[idx] = min(virtual_min[idx], val)
                virtual_max[idx] = max(virtual_max[idx], val)
            # End of loop
        # End of loop
        self.virtual_instance_min = InstanceObject(virtual_min)
        self.virtual_instance_max = virtual_max


class Timeline:
    def __init__(self, min_probability, timeline_objects, interest_parameters):
        self.timeline_objects = timeline_objects
        self.interest_parameters = interest_parameters
        self.dim = len(self.interest_parameters)
        p.dimension = self.dim
        p.overwrite = True
        self.global_rtree = index.Rtree(properties=p)
        self.timelines_list = list()
        # stores timeline_object_indices with values from Umin
        self.min_probability = min_probability
        self.non_skyline_timeline_object_indices = set()
        self.skyline_timeline_object_indices = set()
        self.minHeap = []
        heapify(self.minHeap)
        self.fill_minHeap()

    def find_topK_skylines(self):
        """
        Function to compute the topK skylines
        """
        # Finding the minimum-bounding region
        while len(self.minHeap) > 0:
            key, timeline_object_index, instance_index, virtual = heappop(self.minHeap)

            if timeline_object_index in self.non_skyline_timeline_object_indices:
                continue

            # Check if the instance is a virtual instance
            if virtual:
                # Finding the dominating timeline_objects
                instance = self.timeline_objects[timeline_object_index].virtual_instance_min
                dominating_timeline_object_indices = self.find_dominating_timeline_object_indices(timeline_object_index, instance_index)
                probability = self.compute_probability(instance, dominating_timeline_object_indices)
                self.timeline_objects[timeline_object_index].instances[instance_index].probability = probability
                if probability >= self.min_probability:
                    self.timeline_objects[timeline_object_index].partition_instances_to_layers(self.dim)
                    next_instance_index = self.timeline_objects[timeline_object_index].layers[0][0]
                    heappush(self.minHeap,
                             (sum(self.timeline_objects[timeline_object_index].instances[next_instance_index].value), timeline_object_index,
                              next_instance_index, False))
                else:
                    self.non_skyline_timeline_object_indices.add(timeline_object_index)
                    self.remove_timeline_object_from_global_Rtree(timeline_object_index)

            else:
                instance = self.timeline_objects[timeline_object_index].instances[instance_index]
                dominating_timeline_object_indices = self.find_dominating_timeline_object_indices(timeline_object_index, instance_index)
                # print(timeline_object_index, self.timeline_objects[timeline_object_index].instances[instance_index].value)
                # Applying Pruning Rule 3
                if not self.check_complete_dominating_object_indices(timeline_object_index, instance_index, dominating_timeline_object_indices):
                    probability = self.compute_probability(instance, dominating_timeline_object_indices)
                    # print(timeline_object_index, self.timeline_objects[timeline_object_index].instances[instance_index].value, probability)
                    self.timeline_objects[timeline_object_index].instances[instance_index].probability = probability
                    self.timeline_objects[timeline_object_index].prob_lower_limit += \
                        ((1 / self.timeline_objects[timeline_object_index].instances_count) * probability)
                else:
                    probability = 0
                    self.timeline_objects[timeline_object_index].instances[instance_index].probability = 0

                is_last_instance_in_layer, layer_idx = self.timeline_objects[timeline_object_index].is_last_instance_in_layer(instance_index,
                                                                                                                              self.dim)
                if is_last_instance_in_layer:
                    # update Upmax
                    self.timeline_objects[timeline_object_index].update_Pmax(layer_idx)

                self.timeline_objects[timeline_object_index].processed_instance_count += 1
                self.timeline_objects[timeline_object_index].update_UPmin(probability)

                self.timeline_objects[timeline_object_index].prob_upper_limit = self.timeline_objects[timeline_object_index].prob_lower_limit + \
                                                                                (self.timeline_objects[timeline_object_index].UPmax * \
                                                                                 (self.timeline_objects[timeline_object_index].instances_count -
                                                                                  self.timeline_objects[
                                                                                      timeline_object_index].processed_instance_count)) / \
                                                                                self.timeline_objects[timeline_object_index].instances_count

                # if ((self.timeline_objects[timeline_object_index].instances_count -
                #         self.timeline_objects[timeline_object_index].processed_instance_count) /
                #     self.timeline_objects[timeline_object_index].instances_count) * \
                #         self.timeline_objects[timeline_object_index].UPmin < self.min_probability:
                #     self.prune_other_timeline_objects(timeline_object_index, instance_index)

                if self.timeline_objects[timeline_object_index].prob_lower_limit >= self.min_probability:
                    self.skyline_timeline_object_indices.add(timeline_object_index)

                elif self.timeline_objects[timeline_object_index].prob_upper_limit >= self.min_probability:
                    next_instance_index = self.timeline_objects[timeline_object_index].find_next_instance_in_layer(layer_idx, instance_index)
                    if next_instance_index is not None:
                        heappush(self.minHeap, (sum(self.timeline_objects[timeline_object_index].instances[next_instance_index].value),
                                                timeline_object_index, next_instance_index, False))
                    else:
                        # TODO: Need to delete the timeline_object and update the Rtree
                        self.non_skyline_timeline_object_indices.add(timeline_object_index)
        # End of loop
        # topk_diverse_timelines = list()
        # for i in self.skyline_timeline_object_indices:
        #     topk_diverse_timelines.append(self.timelines_list[i])
        # # End of loop
        # return topk_diverse_timelines

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

        # return [idx for idx in dominating_timeline_object_indices if idx not in self.non_skyline_timeline_object_indices]
        return dominating_timeline_object_indices

    def find_virtual_max_instance(self, timeline_object_index, limit):
        Umax = [-sys.maxsize - 1] * self.dim
        for instance in self.timeline_objects[timeline_object_index].instances[:limit + 1]:
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
        pruning_candidate_timeline_object_indices = list(self.global_rtree.intersection(query))
        self.non_skyline_timeline_object_indices.update(pruning_candidate_timeline_object_indices)
        # for delete_timeline_object_index in pruning_candidate_timeline_object_indices:
        #     vmin_idx = self.timeline_objects[delete_timeline_object_index].virtual_instance_min_idx
        #     mbb = list(self.timeline_objects[delete_timeline_object_index].instances[vmin_idx].value)
        #     mbb.extend(mbb)
        # self.global_rtree.delete(delete_timeline_object_index, mbb)

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
            candidate_dominating_instances = [instance for instance in timeline_obj_instances_list if sum(instance.value) < threshold]
            # candidate_dominating_instances = self.find_dominating_candidate_instances(candidate_instance, candidate_dominating_instances)
            local_rtree = index.Rtree(properties=p)
            for idfr, dominating_instance in enumerate(candidate_dominating_instances):
                mbb = list(dominating_instance.value)
                mbb.extend(dominating_instance.value)
                local_rtree.insert(idfr, mbb)

            dominating_instance_list = list(local_rtree.intersection(query))
            dominating_instances_count = len(dominating_instance_list)
            probability *= (1 - (dominating_instances_count / self.timeline_objects[idx].instances_count))
        # End of loop
        return probability

    def check_complete_dominating_object_indices(self, timeline_object_index, instance_index, dominating_timeline_object_indices):
        """
        :param timeline_object_index:
        :param instance_index:
        :param dominating_timeline_object_indices:
        :return:
        """
        local_Rtree = index.Rtree(properties=p)
        for dominating_timeline_object_index in dominating_timeline_object_indices:
            Vmax = list(self.timeline_objects[dominating_timeline_object_index].virtual_instance_max)
            if sum(Vmax) >= sum(self.timeline_objects[timeline_object_index].instances[instance_index].value):
                continue
            Vmax.extend(Vmax)
            local_Rtree.insert(dominating_timeline_object_index, Vmax)

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
        for i, timeline_object in enumerate(self.timeline_objects):
            umin = timeline_object.virtual_instance_min
            heappush(self.minHeap, (sum(umin.value), i, -1, True))
            mbb = list(umin.value)
            mbb.extend(umin.value)
            self.global_rtree.insert(i, mbb)
        # End of loop

    def remove_timeline_object_from_global_Rtree(self, timeline_object_id):
        """

        :param timeline_object_id:
        :return:
        """
        # vmin_idx = self.timeline_objects[timeline_object_id].virtual_instance_min_idx
        # mbb = list(self.timeline_objects[timeline_object_id].instances[vmin_idx].value)
        # mbb.extend(mbb)
        # self.global_rtree.delete(timeline_object_id, mbb)
        pass

    def naive_algorithm(self):
        prob_dict = dict()
        for t_idx, timeline_object in enumerate(self.timeline_objects):
            probability = 0
            for idx, instance in enumerate(self.timeline_objects[t_idx].instances):
                dominating_timeline_object_indices = self.find_dominating_timeline_object_indices(t_idx, idx)
                p = self.compute_probability(instance, dominating_timeline_object_indices)
                probability += p
            probability = probability / self.timeline_objects[t_idx].instances_count
            prob_dict[t_idx] = probability

        skyline_points = list()
        for idx, prob in prob_dict.items():
            if prob >= self.min_probability:
                skyline_points.append(idx)


def generate_timelines_list(no_of_instances, no_of_sample_parameters, N):
    timelines_list = list()
    parameter_name = ["param_" + str(i) for i in range(0, no_of_sample_parameters)]
    for i in range(0, N):
        instance_list = list()
        for j in range(0, no_of_instances):
            instance = dict()
            for param_name in parameter_name:
                instance[param_name] = random.randint(0, 100)
            # End of loop
            instance["score"] = round(random.random(), 2)
            instance_object = InstanceObject(list(instance.values()))
            instance_list.append(instance_object)
         # End of loop
        timeline_object = TimelineObject(instance_list, no_of_sample_parameters + 1)
        timelines_list.append(timeline_object)
    return timelines_list


def scalability_no_of_timeline_object(iter):
    no_of_instances = 10
    no_of_sample_parameters = 5
    min_probability = 0.5
    # N_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300,
    #           2400, 2500, 2600, 2700, 2800, 2900, 3000]
    N_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    interest_parameters = {"param_" + str(i): "min" for i in range(0, no_of_sample_parameters)}
    interest_parameters["score"] = "max"
    compute_time_skyline = defaultdict(list)
    compute_time_naive = defaultdict(list)
    for N in N_list:
        for i in range(0, iter):
            timelines_list = generate_timelines_list(no_of_instances, no_of_sample_parameters, N)
            skyline = Timeline(min_probability, timelines_list, interest_parameters)
            start = datetime.now()
            skyline.find_topK_skylines()
            end = datetime.now()
            compute_time_skyline[N].append((end - start).total_seconds())
            start = datetime.now()
            skyline.naive_algorithm()
            end = datetime.now()
            compute_time_naive[N].append((end - start).total_seconds())
        # End of loop
        print("completed " + str(N))
    # End of loop

    b = np.zeros((len(N_list), iter))
    c = np.zeros((len(N_list), iter))
    for i, N in enumerate(N_list):
        b[i] = compute_time_skyline[N]
        c[i] = compute_time_naive[N]

    p = pd.DataFrame(b)
    q = pd.DataFrame(c)
    p.index = N_list
    q.index = N_list
    writer = pd.ExcelWriter("scalability-no-of-timeline_objects.xlsx", engine='xlsxwriter')
    p.to_excel(writer, sheet_name="compute time p-skyline")
    q.to_excel(writer, sheet_name="compute time naive")
    writer.save()
    writer.close()


def scalability_no_of_instances_object(iter, N):
    no_of_instances_list = [i for i in range(1, 20)]
    # no_of_instances_list = [2, 5]
    no_of_sample_parameters = 5
    min_probability = 0.5
    interest_parameters = {"param_" + str(i): "min" for i in range(0, no_of_sample_parameters)}
    interest_parameters["score"] = "max"
    compute_time_skyline = defaultdict(list)
    compute_time_naive = defaultdict(list)
    for no_of_instances in no_of_instances_list:
        for i in range(0, iter):
            timelines_list = generate_timelines_list(no_of_instances, no_of_sample_parameters, N)
            skyline = Timeline(min_probability, timelines_list, interest_parameters)
            start = datetime.now()
            skyline.find_topK_skylines()
            end = datetime.now()
            compute_time_skyline[no_of_instances].append((end - start).total_seconds())
            start = datetime.now()
            skyline.naive_algorithm()
            end = datetime.now()
            compute_time_naive[no_of_instances].append((end - start).total_seconds())
            # End of loop
        print("completed " + str(no_of_instances))
        # End of loop

    b = np.zeros((len(no_of_instances_list), iter))
    c = np.zeros((len(no_of_instances_list), iter))
    for i, N in enumerate(no_of_instances_list):
        b[i] = compute_time_skyline[N]
        c[i] = compute_time_naive[N]

    p = pd.DataFrame(b)
    q = pd.DataFrame(c)
    p.index = no_of_instances_list
    q.index = no_of_instances_list
    writer = pd.ExcelWriter("scalability-no-of-instances.xlsx", engine='xlsxwriter')
    p.to_excel(writer, sheet_name="compute time p-skyline")
    q.to_excel(writer, sheet_name="compute time naive")
    writer.save()
    writer.close()


def scalability_no_of_sample_parameters(iter, N, no_of_instances):
    no_of_sample_parameters_list = [i for i in range(1, 11)]
    # no_of_sample_parameters_list = [5, 9]
    min_probability = 0.5
    compute_time_skyline = defaultdict(list)
    compute_time_naive = defaultdict(list)
    for no_of_sample_parameters in no_of_sample_parameters_list:
        interest_parameters = {"param_" + str(i): "min" for i in range(0, no_of_sample_parameters)}
        interest_parameters["score"] = "max"
        for i in range(0, iter):
            timelines_list = generate_timelines_list(no_of_instances, no_of_sample_parameters, N)
            skyline = Timeline(min_probability, timelines_list, interest_parameters)
            start = datetime.now()
            skyline.find_topK_skylines()
            end = datetime.now()
            compute_time_skyline[no_of_sample_parameters].append((end - start).total_seconds())
            start = datetime.now()
            skyline.naive_algorithm()
            end = datetime.now()
            compute_time_naive[no_of_sample_parameters].append((end - start).total_seconds())
            # End of loop
        print("completed " + str(no_of_sample_parameters))
        # End of loop

    b = np.zeros((len(no_of_sample_parameters_list), iter))
    c = np.zeros((len(no_of_sample_parameters_list), iter))
    for i, N in enumerate(no_of_sample_parameters_list):
        b[i] = compute_time_skyline[N]
        c[i] = compute_time_naive[N]

    p = pd.DataFrame(b)
    q = pd.DataFrame(c)
    p.index = no_of_sample_parameters_list
    q.index = no_of_sample_parameters_list
    writer = pd.ExcelWriter("scalability-no-of-sample-parameters.xlsx", engine='xlsxwriter')
    p.to_excel(writer, sheet_name="compute time p-skyline")
    q.to_excel(writer, sheet_name="compute time naive")
    writer.save()
    writer.close()


if __name__ == '__main__':
    # scalability_no_of_timeline_object(30)
    # scalability_no_of_instances_object(30, 50)
    scalability_no_of_sample_parameters(30, 50, 10)
