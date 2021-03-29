import sys
import numpy as np
from app.skyline.instance_object import InstanceObject


class TimelineObject:
    def __init__(self, value_list, dim):
        self.instances = [InstanceObject(value) for value in value_list]
        self.instances_count = len(self.instances)
        self.prob_lower_limit = 0
        self.prob_upper_limit = 1
        self.layers = list()
        self.UPmax = 1  # initial value
        self.UPmin = 1  # Visited(Processed) Subset Pmin
        self.virtual_instance_min_idx = None
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
            elif dominating_instance.value[i] > dominated_instance.values[i]:
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
        end = len(self.layers)
        while beg <= end:
            mid = int(beg + (end - beg) / 2)
            update = False
            for layer_instance in self.layers[mid]:
                dominate = self.check_dominating(instance, layer_instance, dim)
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
            if idx == self.virtual_instance_min_idx:
                continue
            layer_index = self.find_layer(instance, dim)
            total_layers = len(self.layers)
            if layer_index < total_layers:
                self.layers[layer_index].append(idx)
            elif layer_index == total_layers:
                self.layers.append([idx])
            else:
                raise Exception("wrong layer found")
        # End of loop

    def is_last_instance_in_layer(self, instance_idx, dim):
        layer_idx = self.find_layer(self.instances[instance_idx], dim)
        return True, layer_idx if self.layers[layer_idx][-1] == instance_idx else False, layer_idx

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
        virtual_max = [-1-sys.maxsize] * dim
        for instance in self.instances:
            for idx, val in enumerate(instance.value):
                virtual_min[idx] = min(virtual_min[idx], val)
                virtual_max[idx] = max(virtual_max[idx], val)
            # End of loop
        # End of loop
        self.instances.append(InstanceObject(virtual_min))
        self.virtual_instance_min_idx = self.instances_count
        self.instances_count += 1
        self.virtual_instance_max = virtual_max



